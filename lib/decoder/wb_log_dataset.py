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

# # remove answers
# def create_prompt_no_anwer(row, tokenizer, max_seq_len=4096):
#     row["output"] = ""
#     return {"text": prepare_prompt(remove_urls(row['input']), tokenizer, max_seq_len=max_seq_len)}
#
# def formatting_prompts_func_no_out(examples, max_seq_len=500):
#     output_text = []
#     for i in range(len(examples["instruction"])):
#         instruction = examples["instruction"][i]
#         input_text = examples["input"][i]
#
#         input_text = remove_urls(input_text)  # Assuming 'input' is a key in the row dictionary
#         words = input_text.split()  # Split the text into words
#         input_text = ' '.join(words[:max_seq_len])  # Join the first 500 words back into a string
#
#         text = f'''Write a response that appropriately completes the request.
#
#                 ### Instruction:
#                 {instruction}
#
#                 ### Input:
#                 {input_text}
#
#                 ### Response:
#                 '''
#
#         output_text.append(text)
#
#     return output_text
#
# def formatting_prompts_func(examples, max_seq_len=500):
#     output_text = []
#     for i in range(len(examples["instruction"])):
#         instruction = "Step 1: Identify the claimed capabilities of the sender. Step 2: Identify the claimed organization. Step 3: Infer the role of the sender inside this organization: department or title. For Step 1 and Step 2, give an explanation by quoting the most informative phrases from the original paragraph." # shared instruciton
#         input_text = examples["input"][i]
#         response = examples["output"][i]
#
#         input_text = remove_urls(input_text)  # Assuming 'input' is a key in the row dictionary
#         words = input_text.split()  # Split the text into words
#         input_text = ' '.join(words[:max_seq_len])  # Join the first 500 words back into a string
#         input_text = input_text.strip()
#
#         text = f'''Write a response that appropriately completes the request. ### Instruction: {instruction} \n ### Input: {input_text} \n ### Response: {response}'''
#
#         output_text.append(text)
#
#     return output_text


def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['output']}{EOS_TOKEN}" for row in ds]


def pack(dataset, tokenizer, max_seq_len=2048):
    all_token_ids = []
    all_labels = []

    # Tokenize and combine prompt and output, then add EOS token
    for item in dataset:
        tokenized_prompt = tokenizer(item["prompt"],
                                     add_special_tokens=False,
                                     truncation=True,
                                     max_length=max_seq_len)

        tokenized_output = tokenizer(item["output"],
                                     add_special_tokens=False,
                                     truncation=True,
                                     max_length=max_seq_len)

        combined_ids = tokenized_prompt['input_ids'] + tokenized_output['input_ids']
        all_token_ids.extend(combined_ids[:-1]) # align input_ids and labels

        # Create labels, masking the prompt part and including EOS token
        labels = [-100] * len(tokenized_prompt['input_ids']) + tokenized_output['input_ids'][1:]
        all_labels.extend(labels)
        assert len(combined_ids[:-1]) == len(labels)

    print(f"Total number of tokens: {len(all_token_ids)}")
    packed_ds = []

    # Chunking the combined data into sequences
    for i in range(0, len(all_token_ids), max_seq_len):
        input_ids = all_token_ids[i: i + max_seq_len]
        labels = all_labels[i: i + max_seq_len]

        # 检查长度并进行填充
        padding_length = max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids += [tokenizer.pad_token_id] * padding_length
            labels += [-100] * padding_length

        packed_ds.append({"input_ids": input_ids, "labels": labels})

    return packed_ds


def wrap_html(text, answer, result_dir, i):
    # Replace '<' with '&lt;' and '>' with '&gt;' in the answer string
    escaped_answer = answer.replace("<", "&lt;").replace(">", "&gt;")

    # Create an HTML document
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email and Answers</title>
        </head>
        <body>
            <h1>Email:</h1>
            <p>{text}</p> 

            <h1>Answer:</h1>
            <p>{escaped_answer}</p>
        </body>
        </html>
    """

    with open(f"{result_dir}/{i}.html", "w") as html_file:
        html_file.write(html_content)


if __name__ == '__main__':


    data = []
    dataset = 'spamarchieve' # or spamarchieve
    internal_ct = 0
    external_ct = 0
    unclear_ct = 0

    dataset_file_spam = "./datasets/spamarchieve-annot-2023-corrected.jsonl"

    with open(dataset_file_spam, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            entry['metadata'].pop('Id', None)
            data.append(entry)

            if "Step 1: Internal" in entry["output"]:
                internal_ct += 1
            elif "Step 1: Ambiguous" in entry["output"]:
                unclear_ct += 1
            else:
                external_ct += 1

    print(f"Length of Ambiguous = {unclear_ct}")
    print(f"Length of External = {external_ct}")
    print(f"Length of Internal = {internal_ct}")
    exit()

    dataset_file_benign = './datasets/enron-annot-corrected.jsonl'
    with open(dataset_file_benign, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            entry['metadata'].pop('Id', None)
            data.append(entry)

            if "Step 1: Internal" in entry["output"]:
                internal_ct += 1
            elif "Step 1: Ambiguous" in entry["output"]:
                unclear_ct += 1
            else:
                external_ct += 1

    # dataset_file_benign = './datasets/nus-annot.jsonl'
    # with open(dataset_file_benign, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         entry = json.loads(line)
    #         entry['metadata'].pop('Id', None)
    #         if "Step 1: Internal" in entry["output"]:
    #             internal_ct += 1
    #             data.append(entry)

    print(f"Length of Ambiguous = {unclear_ct}")
    print(f"Length of External = {external_ct}")
    print(f"Length of Internal = {internal_ct}")

    random.seed(1234)
    random.shuffle(data)  # shuffle inplace
    train_dataset = data[:-int(len(data)*0.2)]
    eval_dataset = data[-int(len(data)*0.2):]
    train_df = pd.DataFrame(train_dataset)
    eval_df = pd.DataFrame(eval_dataset)

    # train_table = wandb.Table(dataframe=train_df)
    # eval_table = wandb.Table(dataframe=eval_df)

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
        at1.add_file(dataset_file_spam)

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
        # wandb.log({"train_dataset": train_table,
        #            "eval_dataset": eval_table})

