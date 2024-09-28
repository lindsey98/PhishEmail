import os
import datasets
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
from .preprocessing import tokenize_and_align_labels
from .trainer import BertTrainer_FocalLoss
from .evaluation import compute_token_classification_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# Create a mapping from labels to integers
def compute_metrics(p):
    return compute_token_classification_metrics(p, label_list)

if __name__ == '__main__':
    model_id = "google-bert/bert-large-uncased"
    dataset = 'NER'

    dataset_dir = f'./datasets/ner_adversarial_training/'
    label_list = [
        "O",
        "B-identity",
        "I-identity",
        "B-relation",
        "I-relation",
        "B-action",
        "I-action",
    ]

    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    # Load the dataset using the custom dataset class
    ds = datasets.load_dataset("json", data_files={"train": dataset_dir + "train.json",
                                                   "test": dataset_dir + "test.json"})
    train_dataset = ds["train"]
    test_dataset = ds["test"]
    print(train_dataset)
    print(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_wnut = ds.map(lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    '''Train'''
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=7, id2label=id_to_label, label2id=label_to_id
    )
    training_args = TrainingArguments(
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        output_dir="./checkpoints/identity_adversarial_training",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        weight_decay=0.01,
        seed=42,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        load_best_model_at_end=True,
        save_total_limit=2
    )
    #
    trainer = BertTrainer_FocalLoss(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
