
import os
from tqdm import tqdm
import csv
from lib.encoder.IdentityBert import IdentityBert
from lib.reference_db.IdentityMatcher import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
from lib.encoder.visualizer import Visualizer
from lib.utilities.logger import Logger
import numpy as np
from lib.data.Dataset import EmailDataset
import argparse
from datetime import datetime
from pathlib import Path
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

class Config:
    IDENTITY_MODEL_CHECKPOINT = "./checkpoints/output_ner_augmented/checkpoint-1351"
    MATCHING_MODEL_CHECKPOINT = "./checkpoints/characterbert-typos-st"

    REF_IDENTITY_REPS = "./checkpoints/company_database_reps.npy"
    REF_IDENTITY_NAMES = "./checkpoints/company_database_names.npy"
    REF_IDENTITY_MAP = "./checkpoints/company_database_knowphish.json"

    REF_RELATION_REPS = "./checkpoints/internal_relation_reps.npy"
    REF_RELATION_NAMES = "./checkpoints/internal_relation_names.npy"

    identity_model = IdentityBert(IDENTITY_MODEL_CHECKPOINT)
    visualizer_model = Visualizer(IDENTITY_MODEL_CHECKPOINT)

    matching_model = CharacterBERT(MATCHING_MODEL_CHECKPOINT)
    Logger.spit('Loaded the identity recognition model and identity matching model into memory', caller_prefix="Main", debug=True)

    ref_embed_list = np.load(REF_IDENTITY_REPS) if REF_IDENTITY_REPS else None
    ref_tag_list = np.load(REF_IDENTITY_NAMES).tolist()
    brand_index_db = BaseFaissIPRetriever(init_reps=ref_embed_list,
                                          tags=ref_tag_list,
                                          embed_model=matching_model)
    brand_domain_map_path = REF_IDENTITY_MAP
    Logger.spit('Loaded the identity knowledge base into memory', caller_prefix="Main", debug=True)

    relation_embed_list = np.load(REF_RELATION_REPS) if REF_RELATION_REPS else None
    relation_tag_list = np.load(REF_RELATION_NAMES).tolist()
    internal_relation_index_db = BaseFaissIPRetriever(init_reps=relation_embed_list,
                                                      tags=relation_tag_list,
                                                      embed_model=matching_model)

    thre = 0.95

if __name__ == '__main__':

    Logger.set_debug_on()
    today = datetime.today()
    today_date = today.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--email_dir", help="Dir containing all the .eml files", required=True, type=str)
    parser.add_argument("--save_vis", help="Save the visualized results or not", action='store_true')
    parser.add_argument("--vis_dir", help="Where to save the visualized result", default='./datasets/vis', type=str)
    parser.add_argument("--output_csv", default=f'{today_date}_results.csv', help="Output txt path")
    args = parser.parse_args()

    matcher_cls = IdentityMatcher(brand_index_db=Config.brand_index_db,
                                  internal_relation_index_db=Config.internal_relation_index_db,
                                  embed_model=Config.matching_model,
                                  brand_domain_map_path=Config.brand_domain_map_path,
                                  knowledge_base_expansion=False,
                                  gpt_client=None, gpt_assistant=None,
                                  check_action=True,
                                  threshold=Config.thre)

    dataset = EmailDataset(args.email_dir)
    csv_file_path = args.output_csv
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
    Logger.spit('Loaded the testing dataset into memory', caller_prefix="Main", debug=True)
    # Check if we're writing to a new file, and write the header if so
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['email_file_path',
                             'sender_name',
                             'sender_address',
                             'to_names',
                             'to_addresses',
                             'subject',
                             'email_body_text',
                             'sender_identities',
                             'sender_relations',
                             'required_actions',
                             'next_step_of_engagement',
                             'is_inconsistent',
                             'matched_identity',
                             'identity_recog_runtime',
                             'identity_matching_runtime'
                             'pred_time'])

    for it in tqdm(range(len(dataset))):

        if os.path.exists(csv_file_path) and dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), reply_to_address, \
            subject, email_body_text, header = dataset[it]

        # Identity recognition
        parsed_email = f'Subject: {subject}. From: {sender_name}. Body: {email_body_text}'
        identities, actions, relations, urls_after_actions, identity_recog_runtime = Config.identity_model(parsed_email)
        if args.save_vis:
            html = Config.visualizer_model(parsed_email, metadata=email_file_path)
            # Extract the second-level directory name and the base file name from the file path
            email_file_path_obj = Path(email_file_path)
            email_file_path_2nd_level_basename = email_file_path_obj.parts[-2]  # Assuming the file path has at least two directory levels
            email_file_path_basename = email_file_path_obj.stem  # The file name without the extension

            html_file_path = Path(args.vis_dir) / f"{email_file_path_2nd_level_basename}_{email_file_path_basename}.html"
            with open(html_file_path, 'w') as f:
                f.write(html)

        if len(urls_after_actions):
            next_step_of_engagement = urls_after_actions
        else:
            next_step_of_engagement = reply_to_address
        next_step_of_engagement_domains = dataset.domain_parsing(next_step_of_engagement)

        # Brand matching
        sender_domains = dataset.domain_parsing(sender_address) # a set
        recipient_domains = dataset.domain_parsing(to_addresses)
        sender_domains = sender_domains.union(next_step_of_engagement_domains)


        is_inconsistent, matched_identity, identity_matching_runtime = matcher_cls(identities=identities,
                                                                                   actions=actions,
                                                                                   relations=relations,
                                                                                   sender_domains=sender_domains,
                                                                                   recipient_domains=recipient_domains)


        # Append the new row to the CSV file
        with open(csv_file_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
            writer = csv.writer(file)
            writer.writerow([email_file_path,
                             sender_name, sender_address,
                             to_names, to_addresses,
                             subject, email_body_text,
                             identities,
                             relations,
                             actions,
                             next_step_of_engagement,
                             is_inconsistent,
                             matched_identity,
                             identity_recog_runtime,
                             identity_matching_runtime
                             ])



