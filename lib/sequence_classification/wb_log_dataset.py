import json
import wandb
import re
from transformers import AutoTokenizer
import random
import pandas as pd
import os
from datasets import load_from_disk  # for some reason load_dataset gives an error
import json
from lib.data.utils import load_jsonl, prepare_prompt, remove_urls
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


def map_label(label):
    label2id = {"A.": 0, "B.": 1, "Unclear.": 2}
    return label2id.get(label, 2)  # Return 2 if label not found

if __name__ == '__main__':

    data = []

    dataset_file_spam = "./datasets/spamarchieve-annot-2023-classificaiton.jsonl"

    with open(dataset_file_spam, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry)

    dataset_file_benign = "./datasets/enron-annot-classificaiton.jsonl"
    with open(dataset_file_benign, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry)

    random.seed(1234)
    random.shuffle(data)  # shuffle inplace
    train_dataset = data[:-int(len(data)*0.2)]
    eval_dataset = data[-int(len(data)*0.2):]
    train_df = pd.DataFrame(train_dataset)
    train_df = train_df.rename(columns={"input": "sentence", "output": "label"})
    train_df['label'] = train_df['label'].apply(map_label)
    train_df['label'] = train_df['label'].astype(int)
    train_df = train_df[['sentence', 'label']] # optional: only keep these 2 columns

    eval_df = pd.DataFrame(eval_dataset)
    eval_df = eval_df.rename(columns={"input": "sentence", "output": "label"})
    eval_df['label'] = eval_df['label'].apply(map_label)
    eval_df['label'] = eval_df['label'].astype(int) # must map label to int!!
    eval_df = eval_df[['sentence', 'label']] # only keep these 2 columns

    train_table = wandb.Table(dataframe=train_df)
    eval_table = wandb.Table(dataframe=eval_df)

    train_df.to_csv(f"./datasets/mixed_classification_train.csv", index=False)
    eval_df.to_csv(f"./datasets/mixed_classification_eval.csv",  index=False)


    '''log dataset'''
    with wandb.init(project=f"mixed_classification_ft"):
        at1 = wandb.Artifact(
            name=f"mixed_classification_gpt",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at1.add_file(dataset_file_spam)
        at1.add_file(dataset_file_benign)

        # log as a table
        table = wandb.Table(columns=list(data[0].keys()))
        for row in data:
            table.add_data(*row.values())
        wandb.log({f"mixed_classification_gpt_table": table})

        at2 = wandb.Artifact(
            name=f"mixed_classification_gpt_splitted",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at2.add_file(f"./datasets/mixed_classification_train.csv")
        at2.add_file(f"./datasets/mixed_classification_eval.csv")
        wandb.log_artifact(at2)
        wandb.log({"train_dataset": train_table,
                   "eval_dataset": eval_table})
