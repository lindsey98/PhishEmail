import os
import datasets
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
import wandb
from .preprocessing import tokenize_and_align_labels
from .evaluation import compute_token_classification_metrics
from .trainer import BertTrainer_FocalLoss
import nltk
from collections import Counter

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

# Create a mapping from labels to integers
def compute_metrics(p):
    return compute_token_classification_metrics(p, label_list)


if __name__ == '__main__':
    model_id = "google-bert/bert-large-uncased"
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
    os.environ["WANDB_PROJECT"] = f"{dataset}_bert"  # name your W&B project


    training_args = TrainingArguments(
        report_to="wandb",  # this tells the Trainer to log the metrics to W&B
        # output_dir="./checkpoints/output_ner",
        # output_dir="./checkpoints/output_ner_augmented",
        output_dir="./checkpoints/output_ner_augmented_corrected",
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
