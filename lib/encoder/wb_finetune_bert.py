
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
from transformers.integrations import WandbCallback
import torch.nn.functional as F

def compute_metrics(p):
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def postprocess(model, tokenizer, model_outputs, input_ids, offset_mapping):

    entities = []
    for i in range(len(model_outputs.logits)):
        logits = model_outputs.logits[i]
        token_scores = logits.softmax(dim=-1)  # 计算每个 token 的 softmax 得分
        token_labels = token_scores.argmax(dim=-1)  # 选择得分最高的标签

        for token_index, token_label in enumerate(token_labels):
            # 跳过特殊 token（例如 [CLS], [SEP] 等）
            if input_ids[i][token_index] in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                continue

            # 获取原始文本中 token 的起始和结束位置
            start, end = offset_mapping[i][token_index].tolist()
            entity = {
                "word": tokenizer.convert_ids_to_tokens([input_ids[i][token_index]])[0],
                "entity": model.config.id2label[token_label.item()],
                "score": token_scores[token_index][token_label].item(),
                "start": start,
                "end": end
            }
            entities.append(entity)

    return entities

# Define a function to clean the output
def clean_output(output, text):
    entities = []
    current_entity = None
    current_label = None

    for item in output:
        if item['entity'].startswith('B-') or (item['entity'].startswith('I-') and current_entity is None):
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "label": current_label
                })
            current_entity = {"start": item['start'], "end": item['end']}
            current_label = item['entity'][2:]
        elif item['entity'].startswith('I-') and current_label == item['entity'][2:]:
            current_entity['end'] = item['end']
        else:
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "label": current_label
                })
            current_entity = None
            current_label = None

    if current_entity is not None:
        entities.append({
            "start": current_entity['start'],
                    "end": current_entity['end'],
            "label": current_label
        })

    return {"text": text, "ents": entities, "title": None}

def create_spacy_doc(cleaned_output, nlp):
    doc = nlp.make_doc(cleaned_output['text'])
    ents = []
    for ent in cleaned_output['ents']:
        span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
        if span is not None:
            ents.append(span)
    doc.ents = ents
    return doc

class VisualizationCallback(WandbCallback):

    def __init__(self, trainer, test_dataset, num_samples=30):
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        nlp = spacy.blank("en")
        plots = []
        entity_colors = {"organization": "#ADD8E6",  # Light Blue
                         "action": "#FFA07A",  # Light Salmon
                         "relation": "#98FB98"}  # Pale Green

        for example in tqdm(self.sample_dataset, leave=False):
            text = ' '.join(example['tokens'])
            tokens = self.tokenizer([text], return_tensors='pt',
                                    truncation=True, padding=True,
                                    is_split_into_words=False, return_offsets_mapping=True)
            inputs = {
                "input_ids": tokens["input_ids"].to(self.model.device),
                "attention_mask": tokens["attention_mask"].to(self.model.device)
            }
            with torch.inference_mode():
                outputs = self.model(**inputs)  # Assume model is on the appropriate device

            output = postprocess(self.model, self.tokenizer, outputs, tokens["input_ids"], tokens["offset_mapping"])
            cleaned_output = clean_output(output, text)
            doc = create_spacy_doc(cleaned_output, nlp)
            # Apply entity colors
            html = spacy.displacy.render(doc, style="ent", options={"colors": entity_colors})
            plots.append([wandb.Html(html)])

        table = wandb.Table(data=plots, columns=['predictions'])
        wandb.log({'NER predictions': table})

        # Restore the model to the original device
        return control


if __name__ == '__main__':
    # Define the path to the dataset directory
    model_id = "google-bert/bert-large-uncased"
    dataset = 'NER'
    os.environ["WANDB_PROJECT"] = f"{dataset}_bert"  # name your W&B project

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
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    # Load the dataset using the custom dataset class
    ds = datasets.load_dataset("json", data_files={"train": dataset_dir + "train.json", "test": dataset_dir + "test.json"})
    train_dataset = ds["train"]
    test_dataset = ds["test"]
    print(train_dataset)
    print(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized_wnut = ds.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    '''Train'''
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=7, id2label=id_to_label, label2id=label_to_id
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    wandb.init(project=os.getenv("WANDB_PROJECT"), job_type='train', name=model_id)
    wandb_callback = VisualizationCallback(trainer, test_dataset, num_samples=30)
    trainer.add_callback(wandb_callback)
    trainer.train()
    # Save the best model manually
    trainer.save_model(output_dir="./checkpoints/output_ner/best_model")
    # Upload the model checkpoint to WandB
    artifact = wandb.Artifact('model', type='model')
    artifact.add_dir('./checkpoints/output_ner/best_model')
    wandb.log_artifact(artifact)
    wandb.finish()

    '''Inference'''
    # wandb.init(project=os.getenv("WANDB_PROJECT"), job_type='test', name=model_id)
    # nlp = spacy.blank("en")
    # plots = []
    # entity_colors = {"organization": "#ADD8E6",  # Light Blue
    #                  "action": "#FFA07A",  # Light Salmon
    #                  "relation": "#98FB98"}  # Pale Green
    #
    # classifier = pipeline("ner", model="./checkpoints/output_ner/checkpoint-348")
    # for it in tqdm(range(len(test_dataset))):
    #     text = ' '.join(test_dataset[it]['tokens'])
    #     output = classifier(text)
    #     # Clean the output
    #     cleaned_output = clean_output(output, text)
    #     # Load SpaCy model
    #     nlp = spacy.blank("en")
    #     # Create SpaCy doc
    #     doc = create_spacy_doc(cleaned_output, nlp)
    #     # Render the visualization
    #     # Apply entity colors
    #     html = spacy.displacy.render(doc, style="ent", options={"colors": entity_colors})
    #     plots.append([wandb.Html(html)])
    #
    # table = wandb.Table(data=plots, columns=['predictions'])
    # wandb.log({'All Eval NER predictions': table})
    #
    # wandb.finish()



