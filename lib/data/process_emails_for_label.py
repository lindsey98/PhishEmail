import os
from lib.data.dataloader import EmailDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
import pandas as pd
from lib.data.utils import *

# Custom tokenizer function
def custom_tokenize(text):
    # Use regex to split the text into tokens, considering words and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def filter_duplicates(subjects, email_bodies):
    vectorizer = TfidfVectorizer().fit(subjects + email_bodies)
    subject_vectors = vectorizer.transform(subjects)
    email_body_vectors = vectorizer.transform(email_bodies)

    subject_similarity_matrix = cosine_similarity(subject_vectors)
    email_body_similarity_matrix = cosine_similarity(email_body_vectors)

    threshold = 0.5
    duplicates = set()
    for i in tqdm(range(len(subjects))):
        if i in duplicates:
            continue
        for j in range(i + 1, len(subjects)):
            if subject_similarity_matrix[i, j] > threshold or email_body_similarity_matrix[i, j] > threshold:
                duplicates.add(j)

    # Create a new dataset without duplicates
    unique_indices = set(range(len(subjects))) - duplicates
    return unique_indices

if __name__ == '__main__':
    # dataset_name = "spam_archive_2023"
    # dataset_folder = f'./datasets/spam_archive/2023'
    # dataset = EmailDataset(dataset_folder)
    # print(len(dataset))
    # #
    # subjects = []
    # email_bodies = []
    # for it in tqdm(range(len(dataset))):
    #     email_file_path, (sender_name, sender_address), (to_names, to_addresses), subject, email_body_text, headers = dataset[it]
    #     subjects.append(subject)
    #     email_bodies.append(email_body_text)
    #
    # unique_indices = filter_duplicates(subjects, email_bodies)
    # filtered_dataset = [dataset[i] for i in unique_indices]
    #
    # os.makedirs(f'./datasets/{dataset_name}_unique_annotation', exist_ok=True)
    # with open(f'./datasets/{dataset_name}_unique_annotation/annotated_all.json', 'r') as json_file:
    #     existing_data = json.load(json_file)
    # existing_texts = [data['text'] for data in existing_data]
    # entries = []
    # for unique_ind, data in tqdm(zip(unique_indices, filtered_dataset)):
    #
    #     email_file_path, (sender_name, sender_address), (to_names, to_addresses), subject, email_body_text, headers = data
    #
    #     parsed_email = f'Subject: {subject} \n From: {sender_name} \n Recipient address: {to_addresses} \n Body: {email_body_text}'
    #     if parsed_email in existing_texts:
    #         continue
    #
    #     entries.append({
    #         "Id": unique_ind,
    #         "Path": email_file_path,
    #         "text": parsed_email}
    #     )
    #
    #     if len(entries) >= 1000:
    #         break
    #
    # with open(f'./datasets/{dataset_name}_unique_annotation/all_foryifei.json', 'w', encoding='utf-8') as f:
    #     json.dump(entries, f, ensure_ascii=False, indent=4)

    '''Data from Enron'''
    dataset_name = 'Enron_2015'
    df = pd.read_csv("./datasets/enron_mail_2015/emails_subsample_10k.csv")
    df = df.dropna(subset=['body', 'Subject'])
    subjects = df['Subject'].tolist()
    email_bodies = df['body'].tolist()
    unique_indices = filter_duplicates(subjects, email_bodies)
    filtered_dataset = df.iloc[list(unique_indices)]

    # Create output directory if not exists
    os.makedirs(f'./datasets/{dataset_name}_unique_annotation', exist_ok=True)

    # Prepare entries for the JSON file
    entries = []
    for unique_ind, row in tqdm(filtered_dataset.iterrows(), total=len(filtered_dataset)):
        email_file_path = row['file']
        subject = row['Subject']
        email_body_text = row['body']
        sender_name, sender_addr = parseaddr(row['X-From'])
        to_name, to_addr = parseaddr(row["X-To"])

        parsed_email = f'Subject: {subject} \n From: {sender_name} \n Recipient address: {to_addr} \n Body: {email_body_text}'

        entries.append({
            "Id": unique_ind,
            "Path": email_file_path,
            "text": parsed_email
        })

        if len(entries) >= 1000:
            break

    # Save the entries to a JSON file
    with open(f'./datasets/{dataset_name}_unique_annotation/all.json', 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    #

    '''Data from Paul'''
    # with open(f'./datasets/annotated_datasets_from_paul/annotated_all.json', 'r') as json_file:
    #     existing_data = json.load(json_file)
    # existing_texts = [data['text'] for data in existing_data]
    #
    # label_set_organizations = ["words_sender_organization", "signature_org"]
    # label_set_actions = ["sentence_intent_click", "sentence_intent_phonecall", "sentence_passwd",
    #                      "sentence_tone_urgent"]
    #
    # id = existing_data[-1]['Id'] + 1
    # for json_file in os.listdir('./datasets/annotated_datasets_from_paul'):
    #     if json_file == 'annotated_all.json':
    #         continue
    #     data_list = load_jsonl(os.path.join('./datasets/annotated_datasets_from_paul', json_file))
    #
    #     for data in data_list:
    #         text = data['text']
    #         tokens = custom_tokenize(text)
    #         if text in existing_texts:
    #             continue
    #         if data.get('spans'):
    #             annotations = data['spans']
    #             reframed_annotations = []
    #
    #             for annot in annotations:
    #                 entity_text = text[annot['start']:annot['end']]
    #                 entity_class = annot['label']
    #
    #                 if entity_class in label_set_actions:
    #                     label = 'action'
    #                     reframed_annotations.append({"text": entity_text,
    #                                                  "labels": [label]})
    #                 elif entity_class in label_set_organizations:
    #                     label = 'organization'
    #                     reframed_annotations.append({"text": entity_text,
    #                                                  "labels": [label]})
    #
    #             existing_data.append({
    #                 "Id": id,
    #                 "Path": json_file,
    #                 "text": text,
    #                 "annotations": reframed_annotations
    #             })
    #             id += 1
    #
    # with open(f'./datasets/annotated_datasets_from_paul/annotated_all_v2.json', 'w', encoding='utf-8') as f:
    #     json.dump(existing_data, f, ensure_ascii=False, indent=4)