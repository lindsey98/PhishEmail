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
from tqdm.auto import tqdm
from pathlib import Path
from lib.data.utils import prepare_prompt_batch, prepare_prompt, create_prompt_no_answer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, LlamaTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from functools import partial
from lib.data.utils import load_jsonl, prepare_prompt_no_output, remove_urls, pad_eos

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def param_count(m):
    params = sum([p.numel() for p in m.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params


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
    # model_id = 'meta-llama/Llama-2-7b-hf'
    # tokenizer = LlamaTokenizer.from_pretrained(model_id)

    model_id = 'meta-llama/Meta-Llama-3-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # dataset = "enron" #
    dataset = 'spamarchieve'
    os.environ["WANDB_PROJECT"] = f"{dataset}_ft"  # name your W&B project

    output_dir = "./output/"
    batch_size = 1
    gradient_accumulation_steps = 2
    num_train_epochs = 2
    max_seq_length = {'meta-llama/Llama-2-7b-hf': 2048,
                      'meta-llama/Meta-Llama-3-8B': 2048} # by right llama3 can reach 8k

    api = Api()
    artifact = api.artifact(f'lindsey98/{dataset}_ft/{dataset}_gpt_splitted:latest', type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    test_dataset = eval_dataset.map(partial(create_prompt_no_answer,
                                            tokenizer=tokenizer,
                                            max_seq_len=max_seq_length[model_id]),
                                    writer_batch_size=3_000)
    print('Length of training = {}'.format(len(train_dataset)))
    print('Length of evaluation = {}'.format(len(eval_dataset)))

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

    if model_id == 'meta-llama/Llama-2-7b-hf':
        response_template = "\n### Response:" # Note: For llama2, the response_template with context and w/o context are encoded differently. https://huggingface.co/docs/trl/en/sft_trainer#using-tokenids-directly-for-responsetemplate
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # ignore the \n part
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    else:
        response_template = "### Response:\n" # annoying
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model_id,
        model_init_kwargs=model_kwargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        formatting_func=partial(prepare_prompt_batch,
                                tokenizer=tokenizer,
                                max_seq_len=max_seq_length[model_id]),
        packing=False,
        max_seq_length=max_seq_length[model_id],
        args=training_args,
        peft_config=peft_config,
        dataset_batch_size=50,
    )

    '''Train'''
    wandb.init(project=os.getenv("WANDB_PROJECT"), job_type='train', name=model_id)
    wandb_callback = DecoderCallback(trainer, test_dataset, num_samples=25, max_new_tokens=100)
    trainer.add_callback(wandb_callback)
    trainer.train()
    wandb.finish()

