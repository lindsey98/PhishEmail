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
from lib.model_utils.postprocessing import visualize_predictions, ner_clean_predictions, ner_create_spacy_doc
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

def filter_duplicates(strings, threshold=0.6):
    filtered = []
    for string in strings:
        string = string.split(". ")[0] # a minor fix
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


def find_closest_match(query, options, cutoff=0.5):
    # Convert query and options to lowercase for case-insensitive matching
    query_lower = query.lower()
    options_lower = [option.lower() for option in options]

    # Find the best match for the query from the list of options
    best_match = difflib.get_close_matches(query_lower, options_lower, n=1, cutoff=cutoff)

    # If a match is found, return the original case version from options
    if best_match:
        return options[options_lower.index(best_match[0])]
    return None

def contains_domain(text):
    # Regex pattern to match domain names
    pattern = r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,7}\b'

    # Search for the pattern in the given text
    matches = re.findall(pattern, text)

    # Return True if a domain name is found, otherwise False
    return bool(matches), matches if matches else None

def results_calculation(df_cleaned,
                        check_action=True,
                        knowledge_base_expansion=True,
                        ignore_entries=[]):

    reported_ct = 0
    no_pred_ct = 0
    reported = []
    no_prediction_list = []

    for it, row in tqdm(df_cleaned.iterrows()):

        # if it <= 2019:
        #     continue

        email_file_path = row['email_file_path']
        sender_address = row['sender_address']
        sender_domain = sender_address.split('@')[-1] # sender email domain
        to_address_domains = [x.split('@')[-1] for x in row['to_addresses'].split(',')] # recipient email domain

        sender_organization = eval(row['sender_organization'])
        sender_organization = set(filter_duplicates(sender_organization))
        sender_relation = eval(row['sender_relation'])
        required_action = eval(row['required_action'])
        required_action = (len(required_action) > 0) if check_action else True

        if (email_file_path in ignore_entries):
            continue

        # First, try to look for the sender's organization, if any
        sender_organization_and_relation = sender_organization.union(sender_relation)
        sender_organization_and_relation = filter_duplicates(sender_organization_and_relation)

        # No prediction on sender organization or relation
        if len(sender_organization_and_relation) == 0:
            no_pred_ct += 1
            no_prediction_list.append(row)
            continue

        for potential_organization in sender_organization:
            closest_match_in_map = find_closest_match(potential_organization, list(brand_domain_map.keys()), cutoff=0.8)
            contains_a_domain = contains_domain(potential_organization)
            official_emails = None
            if closest_match_in_map: # if the brand name is in the reference list
                official_emails = brand_domain_map[closest_match_in_map]
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
                if (sender_domain not in official_emails):
                    if required_action:  # has action
                        reported_ct += 1
                        reported.append(row)
                break

        if official_emails:
            continue

        # If there is no valid organization predicted, try to see whether the relation is internal
        for relation in sender_organization_and_relation:
            if any([internal_relation.lower() in relation.lower() for internal_relation in internal_relation_list]): # internal
                if sender_domain not in to_address_domains:
                    if required_action:  # has action
                        reported_ct += 1
                        reported.append(row)
                        break

    print(f'Total = {len(df_cleaned)}')
    print(f'Reported = {reported_ct}, % = {reported_ct/len(df_cleaned)}')
    print(f'Failed due to no prediction = {no_pred_ct}, % = {no_pred_ct/len(df_cleaned)}')

    return reported, no_prediction_list

if __name__ == '__main__':

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890"  # proxy
    client = OpenAI()

    assistant = client.beta.assistants.create(
        name="Brand-Email mapping",
        instructions="Given a brand, output its official email address. If the input is not a brand/organization name, output ''. Directly give the email address with no additional explanation. If the you are not sure about the official email, do not respond. If there are multiple possible emails, output them all as a list [email_1, email_2, ...].",
        model="gpt-4-turbo",
    )

    dataset_name = "nazario"
    # dataset_name = "CSDMC2010_benign"
    csv_file_path = f'./datasets/{dataset_name}_results.csv'
    df = pd.read_csv(csv_file_path)
    df = df.drop_duplicates(subset='email_file_path')
    print(f'Original # Emails = {len(df)}') # 2584

    df_cleaned = df.dropna(subset=['sender_address', 'to_addresses', 'email_body_text'], how='any')
    print(f'After dropping nan = {len(df_cleaned)}') # 2387

    print(f"Median runtime = {np.median(df_cleaned['pred_time'])}")

    ## Observe the classification results
    brand_domain_map_path = './datasets/company_database.json'
    with open(brand_domain_map_path, 'r') as file:
        brand_domain_map = json.load(file)

    internal_relation_list = [x.strip() for x in open('./datasets/internal_relations.txt').readlines()]

    reported, no_prediction_list = results_calculation(df_cleaned, check_action=True, knowledge_base_expansion=False)

    # if os.path.exists(f'./datasets/fns_{dataset_name}'):
    #     shutil.rmtree(f'./datasets/fns_{dataset_name}')
    # os.makedirs(f'./datasets/fns_{dataset_name}', exist_ok=True)
    #
    # classifier = pipeline("ner", model="./checkpoints/output_ner/checkpoint-1755")
    # for it, row in enumerate(no_prediction_list):
    #     email_file_path = row['email_file_path']
    #     parsed_email = f"Subject: {row['subject']}. From: {row['sender_name']}. Body: {row['email_body_text']}"
    #
    #     output = classifier(parsed_email)
    #     # Clean the output
    #     cleaned_output = ner_clean_predictions(output, parsed_email)
    #     # Load SpaCy model
    #     nlp = spacy.blank("en")
    #     # Create SpaCy doc
    #     pred_doc = ner_create_spacy_doc(cleaned_output, nlp)
    #     # Render the visualization
    #     html = visualize_predictions(pred_doc, metadata=email_file_path, options={"colors": {"organization": "#ADD8E6", "action": "#FFA07A", "relation": "#98FB98"}  })
    #
    #     html_file_path = f'./datasets/fns_{dataset_name}/{it}.html'
    #     with open(html_file_path, "w", encoding="utf-8") as file:
    #         file.write(html)

    # reported_df = pd.DataFrame(no_prediction_list)
    # reported_df.to_csv('no_pred_emails.csv', index=False)

    # Identity prediction only
    ## on Nazario dataset
    # Reported = 1942, % = 0.7871909201459262

    ## on Benign dataset
    # Reported = 798, % = 0.3674033149171271

    # Identity prediction + instruction
    ## on Nazario dataset
    # Reported = 1744, % = 0.7069314957438184

    ## on Benign dataset
    # Reported = 165, % = 0.07596685082872928

    ### FN reasons
    # R1: ambigous