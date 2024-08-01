from langdetect import detect_langs
import os
import langdetect
from tqdm import tqdm
import csv
from lib.data.dataloader import EmailDataset
import pandas as pd
import numpy as np
import json
from tldextract import tldextract

if __name__ == '__main__':
    desc_folder = './datasets/nazario-recent'
    # desc_folder = './datasets/CSDMC2010/Ham'
    dataset = EmailDataset(desc_folder)
    ct = 0

    dataset_name = "GPT"
    # dataset_name = "nazario"
    # dataset_name = "CSDMC2010_benign"
    csv_file_path = f'./datasets/{dataset_name}_results_augmented.csv'
    df = pd.read_csv(csv_file_path)
    noisy_list = []
    brand_domain_map_path = './datasets/company_database.json'
    with open(brand_domain_map_path, 'r') as file:
        brand_domain_map = json.load(file)
    well_known_domains = [x for y in list(brand_domain_map.values()) for x in y]
    well_known_domains = [tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix for x in well_known_domains]

    for idx in tqdm(df.index):
        # Extract data from the DataFrame
        email_file_path = df.loc[idx, 'email_file_path']
        sender_name, sender_address = df.loc[idx, 'sender_name'], df.loc[idx, 'sender_address']
        to_names, to_addresses = df.loc[idx, 'to_names'], df.loc[idx, 'to_addresses']
        subject = df.loc[idx, 'subject']
        email_body_text = df.loc[idx, 'email_body_text']
        identity = df.loc[idx, 'sender_identity']

        if isinstance(email_body_text, float):
            ct += 1
            noisy_list.append(email_file_path)
            continue

        try:
            detected_langs = detect_langs(email_body_text)
        except langdetect.lang_detect_exception.LangDetectException:
            ct += 1
            noisy_list.append(email_file_path)
            continue
        for lang in detected_langs:
            if lang.lang != 'en':
                print(lang.lang, lang.prob)
                ct += 1
                noisy_list.append(email_file_path)
                break

        if not isinstance(sender_address, float):
            sender_domain = sender_address.split('@')[-1]
            sender_domain = tldextract.extract(sender_domain).domain + '.' + tldextract.extract(sender_domain).suffix
            if sender_domain in ['gmail.com', 'yahoo.com', 'outlook.com',
                                 'protonmail.com', 'zoho.com', 'icloud.com',
                                 'aol.com', 'yandex.com', 'gmx.com', 'tutanota.com',
                                 '163.com', 'qq.com', '126.com']:
                continue
            if sender_domain in well_known_domains: # sender address spoofing
                print(sender_domain)
                ct += 1
                noisy_list.append(email_file_path)
                continue

    with open(f'./datasets/{dataset_name}_noisy_list.txt', 'w') as f:
        for file in noisy_list:
            f.write(file + '\n')

    print(ct, len(df))