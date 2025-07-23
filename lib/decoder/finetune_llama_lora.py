import json
import wandb
import re
from transformers import AutoTokenizer
import os
import torch
from transformers import GenerationConfig
from tqdm.auto import tqdm
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.integrations import WandbCallback
from datasets import load_dataset

class DecoderPreprocesser:

    @staticmethod
    def remove_urls(text):
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(pattern, '', text)

    @staticmethod
    def prompt_input(row, max_seq_len=500):
        cleaned_input = DecoderPreprocesser.remove_urls(row['input'])  # Assuming 'input' is a key in the row dictionary
        cleaned_input = cleaned_input[:max_seq_len]
        return f"### Instruction:\n{row['instruction']}\n\n### Input:\n{cleaned_input}\n\n### Response:\n"

    @staticmethod
    def create_prompt_no_anwer(row):
        row["output"] = ""
        return {"text": DecoderPreprocesser.prompt_input(row)}


    @staticmethod
    def formatting_prompts_func(examples, max_seq_len=500):
        output_text = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            response = examples["output"][i]

            input_text = DecoderPreprocesser.remove_urls(input_text)  # Assuming 'input' is a key in the row dictionary
            words = input_text.split()  # Split the text into words
            input_text = ' '.join(words[:max_seq_len])  # Join the first 500 words back into a string
            input_text = input_text.strip()

            text = f'''Write a response that appropriately completes the request.
    
                    ### Instruction:
                    {instruction}
    
                    ### Input:
                    {input_text}
    
                    ### Response:
                    {response}
                    '''

            output_text.append(text)

        return output_text

    @staticmethod
    def pad_eos(ds):
        EOS_TOKEN = "</s>"
        return [f"{row['output']}{EOS_TOKEN}" for row in ds]

    @staticmethod
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
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
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



if __name__ == '__main__':

    '''Load dataset from W&B'''
    model_id = 'NousResearch/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token


    dataset_dir = "./datasets/llama_training/"
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    test_dataset = eval_dataset.map(DecoderPreprocesser.create_prompt_no_anwer)

    '''dataloader'''
    output_dir = "./checkpoints/llama2"
    batch_size = 1
    gradient_accumulation_steps = 2
    num_train_epochs = 4
    max_seq_length = 2048

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
        formatting_func=DecoderPreprocesser.formatting_prompts_func,
        peft_config=peft_config,
    )
    '''Train'''
    wandb.init(project="llama2", job_type='train')
    trainer.train()
    wandb.finish()
