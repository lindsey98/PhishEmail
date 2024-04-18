import wandb
from transformers import AutoTokenizer
import os
from wandb import Api
import torch
from transformers import TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
from datasets import load_dataset
from functools import partial
import numpy as np
import evaluate

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

def param_count(m):
    params = sum([p.numel() for p in m.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == '__main__':

    '''Load dataset from W&B'''
    model_id = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    os.environ["WANDB_PROJECT"] = "mixed_classification_ft"  # name your W&B project

    output_dir = "./output/"
    batch_size = 4
    num_train_epochs = 4

    api = Api()
    artifact = api.artifact(f'lindsey98/mixed_classification_ft/mixed_classification_gpt_splitted:latest', type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("csv", data_dir=dataset_dir)
    tokenized_ds = ds.map(partial(preprocess_function, tokenizer=tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    '''Configurations'''
    total_num_steps = num_train_epochs * len(ds['train']) // (batch_size)

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=total_num_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
    )

    id2label = {0: "A.", 1: "B.", 2: "Unclear."}
    label2id = {"A.": 0, "B.": 1, "Unclear.": 2}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
        # device_map='auto',
        trust_remote_code=True,
    )
    param_count(model) # 125M

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        args=training_args,
    )

    '''Train'''
    wandb.init(project=os.getenv("WANDB_PROJECT"),
               job_type='train')
    trainer.train()
    wandb.finish()

