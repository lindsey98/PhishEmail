import json
import wandb
import re
from transformers import AutoTokenizer
import random
import pandas as pd
import os
from datasets import load_from_disk  # for some reason load_dataset gives an error
import json
from wandb import Api
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForCausalLM
from types import SimpleNamespace
from transformers import get_cosine_schedule_with_warmup
from transformers import GenerationConfig
from tqdm.auto import tqdm
from pathlib import Path
from lib.sequence_generation.wb_log_dataset import load_jsonl, formatting_prompts_func, create_prompt_no_anwer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.integrations import WandbCallback
from datasets import load_dataset

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["WANDB_PROJECT"] = "spamarchieve_ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def param_count(m):
    params = sum([p.numel() for p in m.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.classify(inputs=tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})

def save_model(model, model_name, models_folder="models", log=False):
    """Save the model to wandb as an artifact
    Args:
        model (nn.Module): Model to save.
        model_name (str): Name of the model.
        models_folder (str, optional): Folder to save the model. Defaults to "models".
    """
    model_name = f"{wandb.run.id}_{model_name}"
    file_name = Path(f"{models_folder}/{model_name}")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(file_name, safe_serialization=True)
    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    tokenizer.save_pretrained(model_name)
    if log:
        at = wandb.Artifact(model_name, type="model")
        at.add_dir(file_name)
        wandb.log_artifact(at)



if __name__ == '__main__':

    '''Load dataset from W&B'''
    model_id = 'meta-llama/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    api = Api()
    artifact = api.artifact('lindsey98/spamarchieve_ft/spamarchieve_gpt_splitted:latest', type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    test_dataset = eval_dataset.map(create_prompt_no_anwer)
    #

    '''dataloader'''
    output_dir = "./output/"
    batch_size = 1
    gradient_accumulation_steps = 2
    num_train_epochs = 4
    max_seq_length = 4096

    '''Configurations'''
    total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)

    peft_config = LoraConfig(
        r=16,  # the rank of the LoRA matrices
        lora_alpha=16,  # the weight
        lora_dropout=0.1,  # dropout to add to the LoRA layers
        bias="none",  # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],  # the name of the layers to add LoRA
        modules_to_save=None,  # layers to unfreeze and train from the original pre-trained model
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        bf16=True, # Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training.
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=total_num_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        evaluation_strategy="steps",
        eval_steps=total_num_steps // num_train_epochs,
        # logging strategies
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=total_num_steps // num_train_epochs,
    )

    model_kwargs = dict(
        device_map='auto',
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        # use_flash_attention_2=True,
        use_cache=False,
    )

    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        packing=False,
        max_seq_length=max_seq_length,
        args=training_args,
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
    )
    '''Train'''
    # wandb.init(project="spamarchieve_ft", job_type='train')
    # wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
    # trainer.add_callback(wandb_callback)
    # trainer.train()
    # wandb.finish()

