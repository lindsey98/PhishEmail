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
from lib.llm.wb_finetune_lora import LLMSampleCB, param_count
from lib.llm.wb_log_dataset import load_jsonl, formatting_prompts_func, create_prompt_no_anwer
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["WANDB_PROJECT"] = "spamarchieve_ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints



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
    output_dir = "./output_6layer/"
    batch_size = 1
    gradient_accumulation_steps = 2
    num_train_epochs = 4
    max_seq_length = 4096

    '''Configurations'''
    total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)

    training_args = TrainingArguments(
        seed=1234,
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
        save_steps= total_num_steps // num_train_epochs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.model.layers[26:].parameters(): # only train last 6 layers
        param.requires_grad = True
    model.model.embed_tokens.weight.requires_grad_(False)
    model.gradient_checkpointing_enable()
    params, trainable_params = param_count(model)

    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        packing=False,
        max_seq_length=max_seq_length,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    '''Train'''
    wandb.init(project="spamarchieve_ft",
               job_type='train')
    wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
    trainer.add_callback(wandb_callback)
    trainer.train()
    wandb.finish()

