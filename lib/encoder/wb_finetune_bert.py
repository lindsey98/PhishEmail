
import json
import os
import datasets
from transformers import AutoTokenizer
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import wandb
from transformers import pipeline
import torch
from tqdm import tqdm
import spacy
from spacy import displacy
import json
from pathlib import Path
import torch.nn.functional as F
from lib.model_utils.preprocessing import tokenize_and_align_labels
from lib.model_utils.callback import NERCallback
from lib.model_utils.postprocessing import *
from lib.model_utils.evaluation import compute_token_classification_metrics, compute_entity_overlap_metrics
from lib.model_utils.trainer import BertTrainer_FocalLoss
from functools import partial
from spacy.tokens import Doc, Span
import nltk
from collections import Counter
from nltk import ngrams

# Ensure you have the necessary nltk data
# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pretty_print_metrics(metrics):
    print("Evaluation Results:")
    print(f"Precision: {metrics['eval_precision']:.3f}")
    print(f"Recall: {metrics['eval_recall']:.3f}")
    print(f"F1 Score: {metrics['eval_f1']:.3f}")
    print("\nDetailed Classification Report:")
    print(metrics['eval_class_report'])

def extract_action_phrases(data):
    action_phrases = []

    for entry in data:
        tokens = entry['tokens']
        tags = entry['ner_tags']

        current_phrase = []
        for token, tag in zip(tokens, tags):
            if tag == 5:  # B-action
                if current_phrase:
                    action_phrases.append(' '.join(current_phrase))
                    current_phrase = []
                current_phrase.append(token)
            elif tag == 6:  # I-action
                if current_phrase:
                    current_phrase.append(token)
            else:
                if current_phrase:
                    action_phrases.append(' '.join(current_phrase))
                    current_phrase = []

        if current_phrase:  # Append any remaining phrase
            action_phrases.append(' '.join(current_phrase))

    return action_phrases

def extract_phrases(sentences, n=3):
    all_phrases = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if len(words) >= n:
            start_ngram = tuple(words[:n])
            all_phrases.append(start_ngram)
    return all_phrases


def summarize_patterns(sentences, n=3):
    phrases = extract_phrases(sentences, n)
    pattern_counter = Counter(phrases)
    return pattern_counter

if __name__ == '__main__':
    model_id = "google-bert/bert-large-uncased"
    dataset = 'NER'

    dataset_dir = f'./datasets/ner_training/'
    label_list = [
        "O",
        "B-organization",
        "I-organization",
        "B-relation",
        "I-relation",
        "B-action",
        "I-action",
    ]

    # Create a mapping from labels to integers
    def compute_metrics(p):
        return compute_token_classification_metrics(p, label_list)

    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    # Load the dataset using the custom dataset class
    ds = datasets.load_dataset("json", data_files={"train": dataset_dir + "train.json",
                                                   "test": dataset_dir + "test.json"})
    train_dataset = ds["train"]
    test_dataset = ds["test"]
    print(train_dataset)
    print(test_dataset)

    '''Observe the action pattern'''
    action_phrases = extract_action_phrases(train_dataset)
    pattern_counter = summarize_patterns(action_phrases, n=5)
    for phrase, count in pattern_counter.most_common():
        print(f"Phrase: {' '.join(phrase)}, Count: {count}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_wnut = ds.map(lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    '''Train'''
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=7, id2label=id_to_label, label2id=label_to_id
    )
    os.environ["WANDB_PROJECT"] = f"{dataset}_bert"  # name your W&B project
    training_args = TrainingArguments(
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        output_dir="./checkpoints/output_ner",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=20,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        load_best_model_at_end=True,
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

    wandb.init(project=os.getenv("WANDB_PROJECT"), job_type='train', name=model_id)
    wandb_callback = NERCallback(trainer, test_dataset, num_samples=30)
    trainer.add_callback(wandb_callback)
    trainer.train()
    wandb.finish()

    '''Evaluate'''
    # model = AutoModelForTokenClassification.from_pretrained(
    #         "./checkpoints/output_ner/checkpoint-1755", id2label=id_to_label, label2id=label_to_id
    #     )
    # tokenizer = AutoTokenizer.from_pretrained("./checkpoints/output_ner/checkpoint-1755")
    # tokenized_wnut = ds.map(lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer))
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    #
    # trainer = Trainer(
    #     model=model,
    #     args = TrainingArguments(report_to=[],
    #                              output_dir='./debug'),
    #     train_dataset=tokenized_wnut["train"],
    #     eval_dataset=tokenized_wnut["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    # metrics = trainer.evaluate()
    # pretty_print_metrics(metrics)

    # Overall Classification performance with 'O' class included
    # Precision: 0.529
    # Recall: 0.657
    # F1 Score: 0.586
    #
    # Class-wise Report:
    #               precision    recall  f1-score   support
    #
    #       action       0.29      0.42      0.35       194
    # organization       0.59      0.71      0.64       562
    #     relation       0.70      0.79      0.74       135

    '''Inference'''
    # nlp = spacy.blank("en")
    # plots = []
    # entity_colors = {
    #     "organization": "#7B68EE",  # Medium Slate Blue for predicted organizations
    #     "action": "#CD5C5C",  # Indian Red for predicted actions
    #     "relation": "#32CD32",  # Lime Green for predicted relations
    # }
    # all_preds = []
    # all_true_labels = []
    # classifier = pipeline("ner", model="./checkpoints/output_ner/checkpoint-1755")
    # vis_dir = "./datasets/ner_visualization"
    # os.makedirs(vis_dir, exist_ok=True)
    #
    # for it in tqdm(range(len(test_dataset))):
    #     text = ' '.join(test_dataset[it]['tokens'])
    #     output = classifier(text)
    #     # Prediction
    #     cleaned_outputs = ner_clean_predictions(output, text)
    #     pred_doc = ner_create_spacy_doc(cleaned_outputs, nlp)
    #
    #     # Ground-truth
    #     ground_truth = ner_clean_ground_truth(test_dataset[it]['tokens'], test_dataset[it]['ner_tags'], id_to_label)
    #     gt_doc = ner_create_spacy_doc(ground_truth, nlp)
    #
    #     html = visualize_predictions_and_ground_truth(pred_doc, gt_doc,
    #                                                   metadata=test_dataset[it]['metadata'],
    #                                                   options={"colors": entity_colors})
    #     with open(f'{vis_dir}/{it}.html', 'w', encoding='utf-8') as f:
    #         f.write(html)
