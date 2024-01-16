import json
import wandb
import re
from transformers import AutoTokenizer
import random
import pandas as pd
import os
from datasets import load_from_disk  # for some reason load_dataset gives an error
import json

def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def remove_urls(text):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(pattern, '', text)

def prompt_input(row, max_seq_len=500):
    cleaned_input = remove_urls(row['input'])  # Assuming 'input' is a key in the row dictionary
    cleaned_input = cleaned_input[:max_seq_len]
    return f"### Instruction:\n{row['instruction']}\n\n### Input:\n{cleaned_input}\n\n### Response:\n"

# remove answers
def create_prompt_no_anwer(row):
    row["output"] = ""
    return {"text": prompt_input(row)}


def formatting_prompts_func_no_out(examples, max_seq_len=500):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]

        input_text = remove_urls(input_text)  # Assuming 'input' is a key in the row dictionary
        words = input_text.split()  # Split the text into words
        input_text = ' '.join(words[:max_seq_len])  # Join the first 500 words back into a string

        text = f'''Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                '''

        output_text.append(text)

    return output_text

def formatting_prompts_func(examples, max_seq_len=500):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        input_text = remove_urls(input_text)  # Assuming 'input' is a key in the row dictionary
        words = input_text.split()  # Split the text into words
        input_text = ' '.join(words[:max_seq_len])  # Join the first 500 words back into a string
        input_text = input_text.strip()

        text = f'''Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                {response}
                '''

        output_text.append(text)

    return output_text


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
    model_id = 'NousResearch/llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    data = []
    dataset_file = "./datasets/spamarchieve-annot-2023.jsonl"

    with open(dataset_file, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry)

    random.seed(1234)
    random.shuffle(data)  # shuffle inplace
    train_dataset = data[:-int(len(data)*0.2)]
    eval_dataset = data[-int(len(data)*0.2):]
    train_df = pd.DataFrame(train_dataset)
    eval_df = pd.DataFrame(eval_dataset)

    train_table = wandb.Table(dataframe=train_df)
    eval_table = wandb.Table(dataframe=eval_df)

    train_df.to_json("./datasets/spamarchieve_gpt_train.jsonl", orient='records', lines=True)
    eval_df.to_json("./datasets/spamarchieve_gpt_eval.jsonl", orient='records', lines=True)

    train_dataset = load_jsonl("./datasets/spamarchieve_gpt_train.jsonl")
    eval_dataset = load_jsonl("./datasets/spamarchieve_gpt_eval.jsonl")
    train_prompts = [prompt_input(row) for row in train_dataset]
    eval_prompts = [prompt_input(row) for row in eval_dataset]

    train_outputs = pad_eos(train_dataset)
    eval_outputs = pad_eos(eval_dataset)

    train_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
    eval_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

    # result_dir = './datasets/iwspa-cot/'
    # os.makedirs(result_dir, exist_ok=True)
    #
    # for it, data in enumerate(train_dataset):
    #     input = data['input']
    #     output = data['output']
    #     wrap_html(input, output, result_dir, it)
    #
    '''Pack the after tokenization'''
    # We will pack multiple short examples into a longer chunk, so we can train more efficiently!

    # log to wandb
    with wandb.init(project="spamarchieve_ft"):
        at1 = wandb.Artifact(
            name="spamarchieve_gpt",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at1.add_file(dataset_file)

        # log as a table
        table = wandb.Table(columns=list(data[0].keys()))
        for row in data:
            table.add_data(*row.values())
        wandb.log({"spamarchieve_gpt_table": table})

        at2 = wandb.Artifact(
            name="spamarchieve_gpt_splitted",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at2.add_file("./datasets/spamarchieve_gpt_train.jsonl")
        at2.add_file("./datasets/spamarchieve_gpt_eval.jsonl")
        wandb.log_artifact(at2)
        wandb.log({"train_dataset": train_table, "eval_dataset": eval_table})
