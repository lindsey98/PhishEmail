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


if __name__ == '__main__':

    dataset = "enron_relation_classification"
    data = []

    dataset_file_benign = "./datasets/enron-annot-classificaiton.jsonl"
    with open('./lib/prompt/relationship_inference_prompt.json', 'rb') as handle:
        relation_prompt = json.load(handle)
    relation_instruction = relation_prompt[0]["content"]
    with open(dataset_file_benign, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            entry['instruction'] = relation_instruction
            entry['output'] = re.sub(r'^\d+\.\s*', '', entry['output'])
            data.append(entry)

    random.seed(1234)
    random.shuffle(data)  # shuffle inplace
    train_dataset = data[:-int(len(data)*0.2)]
    eval_dataset = data[-int(len(data)*0.2):]
    train_df = pd.DataFrame(train_dataset)
    eval_df = pd.DataFrame(eval_dataset)

    train_table = wandb.Table(dataframe=train_df)
    eval_table = wandb.Table(dataframe=eval_df)

    train_df.to_json(f"./datasets/{dataset}_gpt_train.jsonl", orient='records', lines=True)
    eval_df.to_json(f"./datasets/{dataset}_gpt_eval.jsonl", orient='records', lines=True)

    train_dataset = load_jsonl(f"./datasets/{dataset}_gpt_train.jsonl")
    eval_dataset = load_jsonl(f"./datasets/{dataset}_gpt_eval.jsonl")

    '''log dataset'''
    with wandb.init(project=f"{dataset}_ft"):
        at1 = wandb.Artifact(
            name=f"{dataset}_gpt",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at1.add_file(dataset_file_benign)

        # log as a table
        table = wandb.Table(columns=list(data[0].keys()))
        for row in data:
            table.add_data(*row.values())
        wandb.log({f"{dataset}_gpt_table": table})

        at2 = wandb.Artifact(
            name=f"{dataset}_gpt_splitted",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at2.add_file(f"./datasets/{dataset}_gpt_train.jsonl")
        at2.add_file(f"./datasets/{dataset}_gpt_eval.jsonl")
        wandb.log_artifact(at2)
        wandb.log({"train_dataset": train_table,
                   "eval_dataset": eval_table})

