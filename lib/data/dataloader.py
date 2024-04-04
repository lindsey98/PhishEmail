from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
from lib.data.utils import *
import os, sys, email, re
import json

class NerDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {"B-person": 0,
                   "I-person": 1,
                   "B-corporation": 2,
                   "I-corporation": 3,
                   "O": 4}

    INDEX2LABEL = {0: "B-person",
                   1: "I-person",
                   2: "B-corporation",
                   3: "I-corporation",
                   4: "O"}
    NUM_LABELS = 5

    def load_dataset(self, path):
        # Read file
        data = open(path, 'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in tqdm(data, desc="Loading NER dataset"):
            if len(line.strip()) > 0:
                token, label = line.strip().split('\t')
                sentence.append(token)
                if label in self.LABEL2INDEX:
                    seq_label.append(self.LABEL2INDEX[label])
                else:
                    seq_label.append(self.LABEL2INDEX['O'])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset

    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']

        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1]  # For CLS

        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list

        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]

        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']

    def __len__(self):
        return len(self.data)

class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len

    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))

        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            # Truncate or pad subwords and labels to max_seq_len
            len_subwords = min(len(subwords), max_seq_len)
            len_labels = min(len(seq_label), max_seq_len)

            subword_batch[i, :len_subwords] = subwords[:len_subwords]
            mask_batch[i, :len_subwords] = 1  # Mask for real tokens
            subword_to_word_indices_batch[i, :len_subwords] = subword_to_word_indices[:len_subwords]
            seq_label_batch[i, :len_labels] = seq_label[:len_labels]

            # Padding labels with -100 for the ignored index in loss calculation
            seq_label_batch[i, len_labels:] = -100

            seq_list.append(raw_seq)

        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list



class Nazario(Dataset):
    def __init__(self, root_path):
        file_list = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith('eml'):
                    continue
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)

        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = open(email_file_path, encoding="ISO-8859-1").read()

        # Parse the email
        email_message = message_from_string(email_content)

        # sender IP
        sender_ip = extract_sender_ip(email_content)
        # Assuming email_message is your email object
        sender_name, sender_address = parseaddr(email_message.get('From', ''))
        to_names, to_addresses = parseaddr(email_message.get('To', ''))

        # Get 'Subject' header and decode if needed
        subject = email_message['Subject']

        content_collection = []

        # Walk through each part of the email to find text or HTML body
        for part in email_message.walk():
            content_collection = process_email_parts(part, content_collection)

        # remove duplicates
        content_collection = sorted(set(content_collection), key=content_collection.index)
        email_body_text, email_images = parse_email_content(content_collection)

        # replace multiple newline characters with a single period
        email_body_text = re.sub(r'\n+', '. ', email_body_text)
        # replace multiple consecutive periods with a single period
        email_body_text = re.sub(r'\.{2,}', '', email_body_text)
        # replace multiple spaces with a single space
        email_body_text = re.sub(r'\s+', ' ', email_body_text)
        email_body_text = sanitize_text(email_body_text)  # Sanitize the text

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                subject, \
                email_body_text, email_images, \
               sender_ip

class SpamArchieve(Nazario):
    def __init__(self, root_path):
        super().__init__(root_path)

        file_list = []

        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(".txt"):
                    file_list.append(os.path.join(root, file))

        self.file_list = file_list

class IWSPA(Nazario):
    def __init__(self, root_path):
        super().__init__(root_path)
        file_list = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith('txt'):
                    continue
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)

        self.file_list = file_list



if __name__ == '__main__':

    # Parse the email
    # dataset = Nazario(root_path = './datasets/Nazario_2005')
    # columns = ['Id', 'Parsed email', 'From: (Address)', 'To: (Name)', 'To: (Address)']
    #
    # # Create an empty DataFrame with these columns
    # data = []
    #
    # for it, item in tqdm(enumerate(dataset)):
    #     email_file_path, (sender_name, sender_address), (to_names, to_addresses), subject, email_body_text, email_images = item
    #     parsed_email = f'Subject: {subject} \n From: (Name) {sender_name} \n Body: {email_body_text}'
    #
    #     if parsed_email not in [x['text'] for x in data]:
    #         entry = {'Id': it, 'text': parsed_email, 'From: (Address)': sender_address,
    #                  'To: (Name)': to_names, 'To: (Address)': to_addresses}
    #
    #         data.append(entry)
    #
    #         # Save to a new JSON file
    #         with open('./datasets/nazario-data.json', 'w', encoding='utf-8') as f:
    #             for a in data:
    #                 json.dump(a, f)
    #                 f.write('\n')
    #
    #
    # email_content = open('./datasets/IWSPA-AP/Dataset_Full_Header_Training/Dataset_Submit_Legit/1167.txt', encoding="ISO-8859-1").read()
    # email_message = message_from_string(email_content)
    # sender_ip = extract_sender_ip(email_content)

    dataset = IWSPA(root_path = './datasets/IWSPA-AP/Dataset_Full_Header_Training/Dataset_Submit_Legit')
    print(len(dataset.file_list)) # 503 -> after removing duplicates 447

    # Create an empty DataFrame with these columns
    data = []

    for it, item in tqdm(enumerate(dataset)):
        email_file_path, (sender_name, sender_address), (to_names, to_addresses), subject, email_body_text, email_images, sender_ip = item
        parsed_email = f'Subject: {subject} \n From: (Name) {sender_name} \n Body: {email_body_text}'

        if parsed_email not in [x['text'] for x in data]:
            entry = {'id': it, 'text': parsed_email, 'From: (Address)': sender_address,
                     'To: (Name)': to_names, 'To: (Address)': to_addresses,
                     'Path': email_file_path, 'Sender IP': sender_ip}

            data.append(entry)

            # Save to a new JSON file
            # with open('./datasets/iwspa-data.json', 'w', encoding='utf-8') as f:
            #     for a in data:
            #         json.dump(a, f)
            #         f.write('\n')
            with open('./datasets/iwspa-data-benign.json', 'w', encoding='utf-8') as f:
                for a in data:
                    json.dump(a, f)
                    f.write('\n')
            # with open('./datasets/iwspa-data-unlabelled.json', 'w', encoding='utf-8') as f:
            #     for a in data:
            #         json.dump(a, f)
            #         f.write('\n')

    # df = pd.read_csv('./datasets/enron_mail_2015/emails_processed.csv')
    # df['From'] = extract_data("From", df['message'])
    # df['To'] = extract_data("To", df['message'])
    # # drop X-From and X-To columns
    # df = df.drop(['X-From', 'X-To'], axis=1)
    # # dropna
    # df.dropna(subset=['From', 'To'], how='any', inplace=True)
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.to_csv('./datasets/enron_mail_2015/emails_processed_clean.csv', index=False)
    # process_enron_dataset('./datasets/enron_mail_2015/emails_processed_clean.csv')

    # dataset = SpamArchieve(root_path='./datasets/spam_archieve')
    # print(len(dataset.file_list))
    # columns = ['Id', 'Parsed email', 'From: (Address)', 'To: (Name)', 'To: (Address)', 'path']
    #
    # # Create an empty DataFrame with these columns
    # data = set()
    # entries = []
    #
    # # Assuming 'dataset' is an iterable with your email data
    # for it, item in tqdm(enumerate(dataset)):
    #     email_file_path, (sender_name, sender_address), (
    #     to_names, to_addresses), subject, email_body_text, email_images = item
    #     parsed_email = f'Subject: {subject} \n From: (Name) {sender_name} \n Body: {email_body_text}'
    #
    #     # Use a set for faster duplicate check
    #     if parsed_email not in data:
    #         data.add(parsed_email)
    #         entry = {
    #             'Id': it,
    #             'text': parsed_email,
    #             'From: (Address)': sender_address,
    #             'To: (Name)': to_names,
    #             'To: (Address)': to_addresses,
    #             'path': email_file_path
    #         }
    #         entries.append(entry)
    #
    # # Write all data to a JSON file after collecting it
    # with open('./datasets/spamarchieve-data.json', 'w', encoding='utf-8') as f:
    #     for entry in entries:
    #         json.dump(entry, f)
    #         f.write('\n')

    # Load json1 data
    # with open('./datasets/iwspa-data.json', 'r', encoding='utf-8') as file:
    #     data = file.read()
    # json1_data = parse_json(data, delimiter='{"Id":')
    #
    # # Load json2 data
    # with open('./datasets/iwspa-annot.jsonl', 'r', encoding='utf-8') as file:
    #     data = file.read()
    # json2_data = parse_json(data, delimiter='{"id":')
    #
    # # Adjust id in json1 to match json2
    # for json1_record, json2_record in zip(json1_data, json2_data):
    #     json2_record['Path'] = json1_record['Path']
    #     json2_record['Sender IP'] = json1_record['Sender IP']
    #
    # with open('./datasets/iwspa-annot-aug.jsonl', 'w', encoding='utf-8') as f:
    #     for a in json2_data:
    #         json.dump(a, f)
    #         f.write('\n')
    # print()

