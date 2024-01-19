
import openai
import json
from lib.data.download_miller import parse_json
from lib.prompt.prompt import PromptClass, FunctionCall
from lib.deprecated.ner.auto_label import chat_completion, truncate_json_string
import os
from tqdm import tqdm
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890"  # proxy
    model_name = "gpt-3.5-turbo-16k"

    orig_json = './datasets/spamarchieve-data.json'
    with open(orig_json, 'r', encoding='utf-8') as file:
        data = file.read()
    parsed_json_objects = parse_json(data, delimiter='{"Id":')
    parsed_json_objects_2023 = [x for x in parsed_json_objects if x['path'].split('/')[-3] == '2023']  # this year data

    annot_json = './datasets/spamarchieve-instruction-annot-2023.jsonl'
    data = []
    existing_paths = set()

    # Load existing data if file exists
    if os.path.exists(annot_json):
        with open(annot_json, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                data.append(entry)
                existing_paths.add(entry["metadata"]["path"])

    # automatically identify named entity
    for i in tqdm(range(len(parsed_json_objects_2023))):  #

        id = parsed_json_objects_2023[i]["Id"]
        sender_addr = parsed_json_objects_2023[i]["From: (Address)"]
        to_name = parsed_json_objects_2023[i]["To: (Name)"]
        to_addr = parsed_json_objects_2023[i]["To: (Address)"]
        path = parsed_json_objects_2023[i]["path"]
        body = parsed_json_objects_2023[i]['text']

        if path in existing_paths:
            continue

        answer = chat_completion(model_name=model_name, filled_content=body,
                                prompt_template=PromptClass.find_instruction,
                                functions = [None],
                                function_name = None)

        new_entry =  {
            "input": f"{body}",
            "output": f"{answer}",
            "metadata": {
                "Id": f"{id}",
                "From: (Address)": f"{sender_addr}",
                "To: (Name)": f"{to_name}",
                "To: (Address)": f"{to_addr}",
                "path": f"{path}"
            }
        }
        # Append new entry to data
        data.append(new_entry)

        # Save data periodically
        if i % 50 == 0:
            with open(annot_json, 'w', encoding='utf-8') as file:
                for item in data:
                    file.write(json.dumps(item, ensure_ascii=False) + '\n')
