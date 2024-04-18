import openai
from lib.prompt.prompt import PromptClass
import os
from lib.prompt.prompt import chat_completion
from lib.data.utils import *

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890"  # proxy
    model_name = "gpt-3.5-turbo-16k"
    dataset = 'enron'

    if dataset == 'spamassassin':
        orig_json = './datasets/spamarchieve-data.json'
        with open(orig_json, 'r', encoding='utf-8') as file:
            data = file.read()
        parsed_json_objects = parse_json(data, delimiter='{"Id":')
        data_list = [x for x in parsed_json_objects if x['path'].split('/')[-3] == '2023'] # this year data
        annot_json = './datasets/spamarchieve-annot-2023-classificaiton.jsonl'

    elif dataset == 'enron':
        data_list = pd.read_csv('./datasets/enron_mail_2015/emails_processed_clean.csv')
        print(len(data_list))
        data_list = data_list.head(10000) # fixme
        data_list = data_list.to_dict(orient='records')
        annot_json = './datasets/enron-annot-classificaiton.jsonl'

    else:
        raise NotImplementedError

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
    for i in tqdm(range(len(data_list))): #

        if dataset == 'spamassassin':
            id = data_list[i]["Id"]
            sender_addr = data_list[i]["From: (Address)"]
            to_name = data_list[i]["To: (Name)"]
            to_addr = data_list[i]["To: (Address)"]
            path = data_list[i]["path"]
            body = data_list[i]['text']
        elif dataset == 'enron':
            id = i
            sender_name, sender_addr = parseaddr(data_list[i]['X-From'])
            to_name, to_addr = parseaddr(data_list[i]["X-To"])
            path = data_list[i]["file"]
            body = f"Subject: {data_list[i]['Subject']} \n From: {sender_name} \n Recipient address: {to_addr} \n Body: {data_list[i]['body']}"

        if path in existing_paths:
            continue

        answer = chat_completion(model_name=model_name,
                                 filled_content=body,
                                prompt_template=PromptClass.classify,
                                functions = [None],
                                function_name = None)

        new_entry =  {
            "instruction": "Given an email, classify whether the sender is claimed to from the internal organization (e.g. colleague, boss, admin staff, subordinate, teacher, student, etc.) or external organization as the recipient. Answer 'A' if internal, answer 'B' if external, answer 'Unclear' if unsure. Do not give any explanation.",
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

    ### TODO: check those A. internal organization
    # ct = 0
    # with open(annot_json, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         entry = json.loads(line)
    #         answer = entry['output']
    #         if 'Step 1: A.' in answer:
    #             ct += 1
    # print(ct)

    ## TODO correct instruction
    # data = []
    # annot_json = './datasets/spamarchieve-annot-2023-relation-corrected.jsonl'
    #
    # with open(annot_json, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         entry = json.loads(line)
    #         entry["instruction"] = "Step 1: Based on the inferred relationship between message sender and recipient, output 'A' if the message sender is from the internal organization as the message recipient (e.g. colleague, supervisor, or admin), output 'B' if from external organization (e.g. service provider from certain brand), or output 'Unclear' if relationship is not clear. Step 2: Identify the claimed capabilities of the sender, answer 'Unclear' if none. Step 3: Identify the claimed organization, answer 'Unclear' if none. For step 1, give inferred relationship as explanation. For step 2 and 3, quote the one most decisive phrase as explanation."
    #         data.append(entry)
    #
    # with open('./datasets/spamarchieve-annot-2023-relation-corrected.jsonl', 'w', encoding='utf-8') as file:
    #     for item in data:
    #         file.write(json.dumps(item, ensure_ascii=False) + '\n')
