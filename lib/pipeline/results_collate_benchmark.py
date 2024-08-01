import pandas as pd
import numpy as np
import difflib
import re
import os
import time
import openai
from openai import OpenAI
from lib.llm_utils.gpt_utils import assistant_completion
import json
from tqdm import tqdm
import re
import difflib
import shutil
from transformers import pipeline
import spacy
from lib.model_utils.postprocessing import visualize_predictions, ner_create_spacy_doc, CustomNERPipeline, visualize_token_predictions
from transformers import TokenClassificationPipeline, AutoTokenizer, AutoModelForTokenClassification
import random
from spacy import displacy
from tldextract import tldextract

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

def filter_duplicates(strings, threshold=0.8):
    filtered = []
    for string in strings:
        # Remove "from:" or any case variation with optional spaces
        string = re.sub(r'(?i)from\s*:?', '', string)
        # Remove leading "##"
        string = re.sub(r'^\s*#+', '', string)
        string = string.strip()
        if len(string) == 0:
            continue
        # Check similarity with already filtered strings
        similar_found = False
        for i, f in enumerate(filtered):
            if len(f) == 0:
                continue
            if difflib.SequenceMatcher(None, string, f).ratio() > threshold:
                similar_found = True
                # Keep the longer string
                if len(string) > len(f):
                    filtered[i] = string
                break
        if not similar_found:
            filtered.append(string)
    return filtered


def find_closest_match(query, options):
    # Convert query and options to lowercase for case-insensitive matching
    query_lower = query.lower()
    options_lower = [option.lower() for option in options]

    # Define the cutoff based on the query length and word count
    if len(query_lower) <= 6:
        cutoff = 1.0  # Exact match required
    elif len(query_lower.split()) == 1:
        cutoff = 0.85
    else:
        cutoff = 0.8

    # Use difflib for matching
    best_match = difflib.get_close_matches(query_lower, options_lower, n=1, cutoff=cutoff)

    # If a match is found, return the original case version from options
    if best_match:
        return options[options_lower.index(best_match[0])]
    return None

def contains_domain(text):
    # Regex pattern to match domain names, including those split by spaces or special characters
    pattern = r'^\b(?:[a-zA-Z0-9-]+\s?\.\s?)+[a-zA-Z]{2,}\b'

    # Search for the pattern in the given text
    matches = re.findall(pattern, text)

    # Clean up the matches by removing any spaces and checking if the TLD is valid
    cleaned_matches = []
    for match in matches:
        cleaned_match = match.replace(' ', '')
        # Extract the TLD and check if it's valid
        tld = tldextract.extract(cleaned_match).suffix
        if tld:
            cleaned_matches.append(cleaned_match)

    # Return True if a domain name is found, otherwise False
    return bool(cleaned_matches), cleaned_matches if cleaned_matches else None


# Build a regex pattern to match the internal relations with optional words and special characters in between
def build_pattern(phrases):
    parts = []
    for phrase in phrases:
        words = phrase.split()
        parts.append(r'\b' + r'(?:\W*\w+)*\W*'.join(map(re.escape, words)) + r'(?:\W*\w+)*\b')
    return re.compile(r'|'.join(parts), re.IGNORECASE)


def results_calculation(df_cleaned,
                        check_action=True,
                        knowledge_base_expansion=True,
                        ignore_entries=[]):

    reported_ct = 0
    no_pred_ct = 0
    reported = []
    failed_list = set()

    for it, row in tqdm(df_cleaned.iterrows()):

        # if it <= 308:
        #     continue

        # if row['email_file_path'] != './datasets/GPT_Dataset/Giancarlo Pellegrino_Web_Zero_GPT.eml':
        #     continue

        email_file_path = row['email_file_path']
        sender_address = row['sender_address']
        if isinstance(sender_address, float):
            reported_ct += 1
            reported.append(email_file_path)
            continue

        sender_domain = sender_address.split('@')[-1] # sender email domain
        sender_domain = tldextract.extract(sender_domain).domain + '.' + tldextract.extract(sender_domain).suffix
        # to_address_domains = [x.split('@')[-1] for x in row['to_addresses'].split(',')] # recipient email domain
        to_address_domain = "monkey.org" # recipient email domain

        sender_organization = eval(row['sender_identity'])
        sender_organization = set(filter_duplicates(sender_organization))
        sender_relation = eval(row['sender_relation'])
        required_action = eval(row['required_action'])
        required_action = (len(required_action) > 0) if check_action else True

        if (email_file_path in ignore_entries):
            continue

        # First, try to look for the sender's organization, if any
        sender_organization_and_relation = sender_organization.union(sender_relation)

        # No prediction on sender organization or relation
        if len(sender_organization_and_relation) == 0:
            no_pred_ct += 1
            failed_list.add(email_file_path)
            continue

        failed_list.add(email_file_path)

        official_emails = None
        for potential_organization in sender_organization:
            if potential_organization.startswith("##"):
                continue
            closest_match_in_map = find_closest_match(potential_organization, list(brand_domain_map.keys()))
            contains_a_domain = contains_domain(potential_organization)
            official_emails = None
            if closest_match_in_map: # if the brand name is in the reference list
                official_emails = brand_domain_map[closest_match_in_map]
                official_emails = [tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix for x in official_emails]
            elif contains_a_domain[0]: # if the prediction contains the domain directly
                official_emails = contains_a_domain[1]
            elif knowledge_base_expansion: # try to search the official email and update brand-domain map
                search_email = assistant_completion(client=client, query=potential_organization, assistant_id=assistant.id)
                print(f'Searching {potential_organization} in GPT, Return {search_email}')
                email_regex = re.compile(r'^\[?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(,\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})*\]?$')

                if email_regex.fullmatch(search_email): # if the return are valid email addresses
                    emails = search_email.strip('[]').split(',')
                    official_emails = list(set([email.split('@')[1].strip() for email in emails]))
                    brand_domain_map[potential_organization] = official_emails

                    with open(brand_domain_map_path, 'w') as file: # update the reference list
                        json.dump(brand_domain_map, file, indent=4)

            if official_emails:
                if (sender_domain not in official_emails) or ('@' not in sender_address):
                    if required_action:  # has action
                        reported_ct += 1
                        reported.append(email_file_path)
                        failed_list.remove(email_file_path)
                break

        if official_emails:
            continue

        pattern = build_pattern(internal_relation_list)

        # If there is no valid organization predicted, try to see whether the relation is internal
        for relation in sender_organization_and_relation:
            if pattern.search(relation): # internal
                if (sender_domain != to_address_domain) or ('@' not in sender_address):
                    if required_action:  # has action
                        reported_ct += 1
                        reported.append(email_file_path)
                        failed_list.remove(email_file_path)
                        break

    print(f'Total = {len(df_cleaned)}')
    print(f'Reported = {reported_ct}, % = {reported_ct/len(df_cleaned)}')
    print(f'Failed due to no prediction = {no_pred_ct}, % = {no_pred_ct/len(df_cleaned)}')

    return reported, failed_list

if __name__ == '__main__':

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.proxy = "http://127.0.0.1:7890"  # proxy
    # client = OpenAI()
    #
    # assistant = client.beta.assistants.create(
    #     name="Brand-Email mapping",
    #     instructions="Given a brand, output its official email address. If the input is not a brand/organization name, output ''. Directly give the email address with no additional explanation. If the you are not sure about the official email, do not respond. If there are multiple possible emails, output them all as a list [email_1, email_2, ...].",
    #     model="gpt-4-turbo",
    # )

    # dataset_name = "nazario"
    dataset_name = "CSDMC2010_benign"
    # dataset_name = 'GPT'
    csv_file_path = f'./datasets/{dataset_name}_results_augmented.csv'
    df = pd.read_csv(csv_file_path)
    df = df.drop_duplicates(subset='email_file_path')
    print(f'Original # Emails = {len(df)}') # 2584
    print(f"Median runtime = {np.median(df['pred_time'])}")

    if os.path.exists(f'./datasets/{dataset_name}_noisy_list.txt'):
        noisy_email_files = [x.strip() for x in open(f'./datasets/{dataset_name}_noisy_list.txt').readlines()]
    else:
        noisy_email_files = []
    df_cleaned = df[~df['email_file_path'].isin(noisy_email_files)]

    ### some fake organizations
    df_cleaned = df_cleaned[~df_cleaned['sender_name'].isin(["Cyberdefense College",
                                                             "Cyberdefense Consortium",
                                                             "Cybersecurity Innovation Center",
                                                             "DDoS Clear"])]

    df_cleaned = df_cleaned.reset_index(drop=True)
    print(f'Clean # Emails = {len(df_cleaned)}') #

    ## Observe the classification results
    brand_domain_map_path = './datasets/company_database.json'
    with open(brand_domain_map_path, 'r') as file:
        brand_domain_map = json.load(file)

    with open('./datasets/internal_relations.txt', 'r') as file:
        internal_relation_list = file.read().splitlines()

    reported, failed_list = results_calculation(df_cleaned,
                                               check_action=True,
                                               knowledge_base_expansion=False)
    exit()

    '''Debug FPs'''
    # if os.path.exists(f'./datasets/fps_{dataset_name}'):
    #     shutil.rmtree(f'./datasets/fps_{dataset_name}')
    # os.makedirs(f'./datasets/fps_{dataset_name}', exist_ok=True)
    #
    # model_path = "./checkpoints/output_ner_augmented/checkpoint-1351"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForTokenClassification.from_pretrained(model_path)
    # classifier = pipeline("ner", model=model_path, aggregation_strategy="simple")
    # token_pipeline = CustomNERPipeline(model=model, tokenizer=tokenizer)
    # #
    # random.seed(1234)
    # selected_100 = random.sample(range(len(reported)), len(reported))
    # for it in tqdm(selected_100):
    #     email_file = list(reported)[it]
    #     row = df.loc[df['email_file_path'] == email_file].iloc[0, :].to_dict()
    #     email_file_path = row['email_file_path']
    #     sender_address = row['sender_address']
    #     parsed_email = f"Subject: {row['subject']}. From: {row['sender_name']}. Body: {row['email_body_text']}"
    #
    #     # Tokenize the parsed email
    #     tokens = tokenizer(parsed_email, return_offsets_mapping=True, truncation=True)
    #     tokenized_email = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
    #     tokenized_email_str = tokenizer.convert_tokens_to_string(tokenized_email)
    #
    #     entities = classifier(tokenized_email_str)
    #     nlp = spacy.blank("en")
    #     pred_doc = ner_create_spacy_doc(tokenized_email_str, entities, nlp)
    #     html_template = visualize_predictions(pred_doc,
    #                                           metadata=email_file_path+'\n'+f"sender_address: {sender_address}",
    #                                           options={"colors": {"organization": "#ADD8E6", "action": "#FFA07A", "relation": "#98FB98"}})
    #
    #     debug_output = token_pipeline(parsed_email)
    #     html_template += visualize_token_predictions(debug_output)
    #
    #     html_file_path = f'./datasets/fps_{dataset_name}/{it}.html'
    #     with open(html_file_path, "w", encoding="utf-8") as file:
    #         file.write(html_template)
    #
    '''Debug FNS'''
    if os.path.exists(f'./datasets/fns_{dataset_name}'):
        shutil.rmtree(f'./datasets/fns_{dataset_name}')
    os.makedirs(f'./datasets/fns_{dataset_name}', exist_ok=True)

    model_path = "./checkpoints/output_ner_augmented/checkpoint-1351"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    classifier = pipeline("ner", model=model_path, aggregation_strategy="simple")
    token_pipeline = CustomNERPipeline(model=model, tokenizer=tokenizer)
    #
    random.seed(1234)
    selected_100 = random.sample(range(len(failed_list)), 30)
    # selected_100 = random.sample(range(len(failed_list)), len(failed_list))
    for it in tqdm(selected_100):
        email_file = list(failed_list)[it]
        row = df.loc[df['email_file_path'] == email_file].iloc[0, :].to_dict()
        email_file_path = row['email_file_path']
        sender_address = row['sender_address']
        parsed_email = f"Subject: {row['subject']}. From: {row['sender_name']}. Body: {row['email_body_text']}"

        # Tokenize the parsed email
        tokens = tokenizer(parsed_email, return_offsets_mapping=True, truncation=True)
        tokenized_email = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        tokenized_email_str = tokenizer.convert_tokens_to_string(tokenized_email)

        entities = classifier(tokenized_email_str)
        nlp = spacy.blank("en")
        pred_doc = ner_create_spacy_doc(tokenized_email_str, entities, nlp)
        html_template = visualize_predictions(pred_doc,
                                              metadata=email_file_path+'\n'+f"sender_address: {sender_address}",
                                              options={"colors": {"organization": "#ADD8E6", "action": "#FFA07A", "relation": "#98FB98"}})

        debug_output = token_pipeline(tokenized_email_str)
        html_template += visualize_token_predictions(debug_output)

        html_file_path = f'./datasets/fns_{dataset_name}/{it}.html'
        with open(html_file_path, "w", encoding="utf-8") as file:
            file.write(html_template)
    #
