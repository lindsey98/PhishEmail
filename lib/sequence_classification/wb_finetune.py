from datasets import load_dataset
from wandb import Api
from transformers.integrations import WandbCallback
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import os
import evaluate # pip install evaluate
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def preprocess_function(examples, tokenizer):
    # Assuming 'text' is the key in your dataset that contains the input sentences
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)  # Set max_length as needed


def compute_metrics(eval_pred):
    accuracy = evaluate.combine(["accuracy", "f1", "precision", "recall"]) # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer

    def classify(self, text):
        tokenized_prompt = self.tokenizer(text, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            logits = self.model(tokenized_prompt).logits
            predicted_class_id = logits.argmax().item()
        return self.model.config.id2label[predicted_class_id]

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["text", "predicted class", "ground-truth class"])
        for example in tqdm(examples, leave=False):
            text = example["text"]
            gt_label = self.model.config.id2label[example["labels"]]
            pred_label = self.classify(text=text)
            records_table.add_data(text, pred_label, gt_label)
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})

def model_training():
    # Initialize wandb run
    wandb.init()

    # Retrieve sweep parameters
    sweep_config = wandb.config
    batch_size = sweep_config.batch_size
    learning_rate = sweep_config.lr
    num_train_epochs = sweep_config.epochs

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=1,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    api = Api()
    artifact = api.artifact('lindsey98/spamarchieve_instruction/spamarchieve_instruction_splitted_spamarchieve:latest', type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    model_id = "bert-base-uncased"
    output_dir = "./output_instruction"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = ds.map(partial(preprocess_function, tokenizer=tokenizer), batched=True)
    tokenized_datasets = tokenized_dataset.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    id2label = {0: "Non-instruction", 1: "Instruction"}
    label2id = {"Non-instruction": 0, "Instruction": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=id2label, label2id=label2id,
    )


    # wandb.init(project="spamarchieve_instruction", job_type='train')

    #
    # # Set up training arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     report_to="wandb",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=4,
    #     weight_decay=0.01,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     logging_strategy="steps",
    #     logging_steps=1,
    # )
    #
    # # Initialize Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    #
    # # Train the model
    # trainer.train()
    # # Finish the wandb run
    # wandb.finish()

    with wandb.init(project="spamarchieve_instruction"):

        # Create an artifact for the model
        artifact = wandb.Artifact('model_artifact', type='model')

        # Directory containing the checkpoints
        checkpoints_dir = './output_instruction/'

        # Add each checkpoint to the artifact
        for checkpoint_dir in os.listdir(checkpoints_dir):
            if checkpoint_dir.startswith('checkpoint-'):
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_dir)
                artifact.add_dir(checkpoint_path, name=checkpoint_dir)

        wandb.log_artifact(artifact)
