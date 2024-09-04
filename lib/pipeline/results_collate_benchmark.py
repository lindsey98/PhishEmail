import pandas as pd
import numpy as np
import os

import json
from tqdm import tqdm
from lib.reference_db.BrandMatcher import CharacterBERT, BrandMatcher, BaseFaissIPRetriever
from tldextract import tldextract
import matplotlib.pyplot as plt
from typing import List
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()


def results_calculation(df_cleaned,
                        brand_index_db: BaseFaissIPRetriever,
                        internal_relation_index_db: BaseFaissIPRetriever,
                        embed_model: CharacterBERT,
                        brand_domain_map_path: str,
                        knowledge_base_expansion: bool = False,
                        gpt_client = None,
                        gpt_assistant = None,
                        check_action: bool = True,
                        sim_threshold: float = 0.78,
                        ignore_entries: List=[]):


    matcher_cls = BrandMatcher(brand_index_db=brand_index_db,
                internal_relation_index_db=internal_relation_index_db,
                embed_model=embed_model,
                brand_domain_map_path=brand_domain_map_path,
                knowledge_base_expansion=knowledge_base_expansion,
                gpt_client=gpt_client, gpt_assistant=gpt_assistant,
                check_action=check_action,
                threshold=sim_threshold)

    reported_ct = 0
    no_pred_ct = 0
    reported = []
    failed_list = set()
    no_pred_list = set()

    for it, row in tqdm(df_cleaned.iterrows()):
        email_file_path = row['email_file_path']
        sender_address = row['sender_address']
        sender_identities = eval(row['sender_identity'])
        sender_relation = eval(row['sender_relation'])
        required_actions = eval(row['required_action'])

        # Skip entries that are in the ignore list
        if email_file_path in ignore_entries:
            continue

        if isinstance(sender_address, float):
            sender_domain = None
        else:
            invalid_address = '@' not in sender_address
            if invalid_address:
                sender_domain = None
            else:
                sender_domain = sender_address.split('@')[-1]
                sender_domain = tldextract.extract(sender_domain).domain + '.' + tldextract.extract(sender_domain).suffix

        recipient_domains = ["monkey.org"]  # hardcode for this Nazario dataset

        is_inconsistent, _ = matcher_cls(identities=sender_identities,
                                    actions=required_actions,
                                    relations=sender_relation,
                                    sender_domain=sender_domain,
                                    recipient_domains=recipient_domains)

        if is_inconsistent:
            reported_ct += 1
            reported.append(email_file_path)
        elif is_inconsistent is None:
            no_pred_list.add(email_file_path)
        else:
            failed_list.add(email_file_path)

    print(f'Total = {len(df_cleaned)}')
    print(f'Reported = {reported_ct}, % = {reported_ct / len(df_cleaned)}')
    print(f'Failed due to no prediction = {no_pred_ct}, % = {no_pred_ct / len(df_cleaned)}')

    return reported, failed_list, reported_ct/len(df_cleaned)





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

    dataset_name = "nazario"
    # dataset_name = "CSDMC2010_benign"
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

    ### CharacterBert model
    ref_embed_list = np.load('./datasets/company_database_reps.npy')
    ref_tag_list = np.load('./datasets/company_database_names.npy').tolist()
    brand_index_db = BaseFaissIPRetriever(init_reps=ref_embed_list,
                                          tags=ref_tag_list)

    relation_embed_list = np.load('./datasets/internal_relation_reps.npy')
    relation_tag_list = np.load('./datasets/internal_relation_names.npy').tolist()
    internal_relation_index_db = BaseFaissIPRetriever(init_reps=relation_embed_list,
                                                      tags=relation_tag_list)

    CharacterBert_MODEL = CharacterBERT()

    benchmarking_txt = f'./datasets/{dataset_name}_benchmarking.txt'
    for thre in np.linspace(0.9, 0.99, 30):
        reported, failed_list, recall = results_calculation(df_cleaned,
                                                            brand_index_db=brand_index_db,
                                                            internal_relation_index_db=internal_relation_index_db,
                                                            embed_model=CharacterBert_MODEL,
                                                            brand_domain_map_path=brand_domain_map_path,
                                                            knowledge_base_expansion = False,
                                                            gpt_client = None,
                                                            gpt_assistant = None,
                                                            check_action = True,
                                                            sim_threshold = thre)

        if os.path.exists(benchmarking_txt) and str(thre)+'\t' in open(benchmarking_txt).read():
            continue
        with open(benchmarking_txt, 'a+') as f:
            f.write(str(thre) + '\t' + str(recall) + '\n')

    exit()
    #############
    # Example data
    tpr = [eval(x.strip().split('\t')[1]) for x in open('./datasets/nazario_benchmarking.txt').readlines()]
    fpr = [eval(x.strip().split('\t')[1]) for x in open('./datasets/CSDMC2010_benign_benchmarking.txt').readlines()]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='x', linestyle='-', color='blue')

    # Finalizing the plot
    plt.xscale('log')
    plt.xlim(1e-3, 0.5)
    plt.xlabel('FPR (on CSDMC dataset)', fontsize=12)
    plt.ylabel('TPR (on Nazario dataset)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Show the plot
    plt.savefig('./debug.png')
    print()

    ############

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
    # selected_100 = random.sample(range(len(reported)), 30)
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
    # if os.path.exists(f'./datasets/fns_{dataset_name}'):
    #     shutil.rmtree(f'./datasets/fns_{dataset_name}')
    # os.makedirs(f'./datasets/fns_{dataset_name}', exist_ok=True)
    #
    # model_path = "./checkpoints/output_ner_augmented/checkpoint-1351"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForTokenClassification.from_pretrained(model_path)
    # classifier = pipeline("ner", model=model_path, aggregation_strategy="simple")
    # token_pipeline = CustomNERPipeline(model=model, tokenizer=tokenizer)
    # #
    # random.seed(1234)
    # selected_100 = random.sample(range(len(failed_list)), 50)
    # # selected_100 = random.sample(range(len(failed_list)), len(failed_list))
    # for it in tqdm(selected_100):
    #     email_file = list(failed_list)[it]
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
    #     debug_output = token_pipeline(tokenized_email_str)
    #     html_template += visualize_token_predictions(debug_output)
    #
    #     html_file_path = f'./datasets/fns_{dataset_name}/{it}.html'
    #     with open(html_file_path, "w", encoding="utf-8") as file:
    #         file.write(html_template)
    #
