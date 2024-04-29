import openai
from lib.prompt.prompt import PromptClass
import os
from lib.prompt.prompt import chat_completion
from lib.data.utils import *
from fuzzywuzzy import process, fuzz
import numpy as np
import concurrent.futures
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

def find_unique_subjects(sub_df, unique_subjects):
    local_unique = []
    for subject in sub_df['Subject']:
        if not process.extractOne(subject, unique_subjects + local_unique, scorer=fuzz.token_sort_ratio, score_cutoff=85):
            local_unique.append(subject)
    return local_unique

def drop_similar_subjects(df):
    num_splits = 10  # Number of splits can be adjusted based on your system's capabilities
    split_dfs = np.array_split(df, num_splits)
    unique_subjects = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(find_unique_subjects, split_dfs, [unique_subjects]*num_splits)

    # Combine results
    for result in results:
        unique_subjects.extend(result)

    # Filter DataFrame to keep only unique subjects
    final_df = df[df['Subject'].isin(unique_subjects)]
    return final_df

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
        data_list = data_list.drop_duplicates(subset=['Subject'])
        print(len(data_list))
        data_list = data_list.sample(frac=1, random_state=1234)
        data_list = data_list.head(5000) # fixme
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
                                 prompt_template=PromptClass.ask_relation,
                                 functions = [None],
                                 function_name = None)

        new_entry =  {
            "instruction": "Given an email, infer the relationship between the sender and recipient. Select ONE of the following: 1. Colleague. 2. Manager and Team Members. 3. Executive and Employees. 4. Admin (from IT, HR, other internal departments) and Staff. 5. Student and Teacher. 6. Mentor and Mentee. 7. Friend. 8. Family. 9. Client and Service Provider. 10. Customer and Business.  11. None of the above.",
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

