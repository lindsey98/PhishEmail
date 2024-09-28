import os
import datasets
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, BertConfig, BertPreTrainedModel
import wandb
from .preprocessing import tokenize_and_align_labels
from .evaluation import compute_token_classification_metrics
from .trainer import BertTrainer_FocalLoss
import nltk
from collections import Counter
from ...reference_db.CharacterBert import CharacterIndexer
from ...reference_db import CharacterBertModel
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import torch
import torch.nn as nn

def tokenize_and_align_labels(examples, character_indexer):
    all_input_ids = []
    all_attention_masks = []
    all_label_ids = []

    for tokens, labels in zip(examples["tokens"], examples["ner_tags"]):
        # Convert tokens to character IDs
        input_ids = character_indexer.tokens_to_indices(tokens)  # List of [char_ids]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Copy labels (assumes labels are already numerical IDs)
        label_ids = labels.copy()

        # Handle padding for sequences shorter than max_seq_length
        max_seq_length = 128  # Adjust as needed
        seq_padding_length = max_seq_length - len(input_ids)
        if seq_padding_length > 0:
            # Pad input_ids
            padding_word = character_indexer._default_value_for_padding()
            input_ids += [padding_word] * seq_padding_length
            # Pad attention_mask
            attention_mask += [0] * seq_padding_length
            # Pad labels
            label_ids += [-100] * seq_padding_length  # Use -100 to ignore padding in loss computation
        else:
            # Truncate sequences longer than max_seq_length
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
            label_ids = label_ids[:max_seq_length]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_label_ids.append(label_ids)

    return {
        "input_ids": all_input_ids,            # Shape: (batch_size, seq_length, char_length)
        "attention_mask": all_attention_masks, # Shape: (batch_size, seq_length)
        "labels": all_label_ids                # Shape: (batch_size, seq_length)
    }


def custom_collate_fn(features):
    batch_input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
    batch_attention_masks = [torch.tensor(f['attention_mask'], dtype=torch.long) for f in features]
    batch_labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]

    # Stack the tensors
    batch_input_ids = torch.stack(batch_input_ids)
    batch_attention_masks = torch.stack(batch_attention_masks)
    batch_labels = torch.stack(batch_labels)

    return {
        'input_ids': batch_input_ids,  # Shape: (batch_size, seq_length, char_length)
        'attention_mask': batch_attention_masks,  # Shape: (batch_size, seq_length)
        'labels': batch_labels  # Shape: (batch_size, seq_length)
    }


class CharacterBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize the base CharacterBertModel
        self.character_bert = CharacterBertModel.from_pretrained(config)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights
        self.init_weights()

    def forward(
            self,
            input_ids=None,  # Shape: (batch_size, seq_length, char_length)
            attention_mask=None,  # Shape: (batch_size, seq_length)
            token_type_ids=None,  # Shape: (batch_size, seq_length)
            labels=None,  # Shape: (batch_size, seq_length)
            **kwargs
    ):
        # Pass inputs through CharacterBertModel
        outputs = self.character_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        # Get the sequence output
        sequence_output = outputs[0]  # Shape: (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # Compute logits
        logits = self.classifier(sequence_output)  # Shape: (batch_size, seq_length, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten the tensors for loss computation
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        output = (logits,) + outputs[2:]  # Include hidden states and attentions if they are present
        return ((loss,) + output) if loss is not None else output

if __name__ == '__main__':
    dataset = 'NER'

    # dataset_dir = f'./datasets/ner_training/'
    dataset_dir = f'./datasets/ner_training_augmented/'
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
    num_labels = len(label_list)

    # Load the dataset using the custom dataset class
    ds = datasets.load_dataset("json", data_files={"train": dataset_dir + "train.json",
                                                   "test": dataset_dir + "test.json"})
    train_dataset = ds["train"]
    test_dataset = ds["test"]

    character_indexer = CharacterIndexer()
    tokenized_wnut = ds.map(lambda examples: tokenize_and_align_labels(examples, character_indexer=character_indexer))

    '''Train'''
    config = BertConfig.from_pretrained('./checkpoints/characterbert-typos-st/')

    # Initialize the Model
    model = CharacterBertForTokenClassification(config)

    training_args = TrainingArguments(
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        # output_dir="./checkpoints/output_ner",
        output_dir="./checkpoints/output_ner_augmented",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=7,
        weight_decay=0.01,
        seed=42,
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
    # classifier = pipeline("ner", model="./checkpoints/output_ner/checkpoint-1344", aggregation_strategy="simple")
    #
    # vis_dir = "./datasets/ner_visualization"
    # if os.path.exists(vis_dir):
    #     shutil.rmtree(vis_dir)
    # os.makedirs(vis_dir)
    #
    # for it in tqdm(range(len(test_dataset))):
    #     text = ' '.join(test_dataset[it]['tokens'])
    #     entities = classifier(text)
    #
    #     # Prediction
    #     pred_doc = ner_create_spacy_doc(text, entities, nlp)
    #
    #     # Ground-truth
    #     ground_truth = ner_clean_ground_truth(test_dataset[it]['tokens'], test_dataset[it]['ner_tags'], id_to_label)
    #     gt_doc = ner_create_spacy_doc(ground_truth["text"], ground_truth["ents"], nlp)
    #
    #     html = visualize_predictions_and_ground_truth(pred_doc, gt_doc,
    #                                                   metadata=test_dataset[it]['metadata'],
    #                                                   options={"colors": entity_colors})
    #     with open(f'{vis_dir}/{it}.html', 'w', encoding='utf-8') as f:
    #         f.write(html)
