
import os
from tqdm import tqdm
import csv
from lib.data import EmailDataset
from lib.encoder import IdentityBert
from inference import main
from lib.baselines import dfence, helphed, rspamd
from lib.utilities.logger import Logger
from lib.reference_db import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
import numpy as np
import json

# Function to read existing email_file_paths from the CSV
def get_existing_paths(csv_file_path):
    existing_paths = set()
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header row
            for row in reader:
                existing_paths.add(row[0])  # Assuming email_file_path is the first column
    except FileNotFoundError:
        pass  # If the file does not exist, we have no existing paths
    return existing_paths


class Config:
    IDENTITY_MODEL_CHECKPOINT = os.getenv("IDENTITY_MODEL_CHECKPOINT", "./checkpoints/identity-model")
    MATCHING_MODEL_CHECKPOINT = os.getenv("MATCHING_MODEL_CHECKPOINT", "./checkpoints/characterbert-typos-st")

    REF_IDENTITY_REPS = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_reps_v2.npy")
    REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_names_v2.npy")
    REF_IDENTITY_MAP = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_knowphish_v2.json") # todo: this is an extended version of knowledge base (unclean) with >6k brands, only for the LLM experiment

    REF_RELATION_REPS = os.getenv("REF_RELATION_REPS", "./checkpoints/internal_relation_reps.npy")
    REF_RELATION_NAMES = os.getenv("REF_RELATION_NAMES", "./checkpoints/internal_relation_names.npy")

    identity_model = IdentityBert(IDENTITY_MODEL_CHECKPOINT)
    matching_model = CharacterBERT(MATCHING_MODEL_CHECKPOINT)

    Logger.spit('Loaded the identity recognition model and identity matching model into memory', caller_prefix="Main", debug=True)

    with open(REF_IDENTITY_MAP, 'r') as file:
        brand_domain_map = json.load(file)
    ref_embed_list = np.load(REF_IDENTITY_REPS) if os.path.exists(REF_IDENTITY_REPS) else None

    if ref_embed_list is None or len(ref_embed_list) != len(list(brand_domain_map.keys())):
        Logger.spit('Cache the knowledge base embeddings...', caller_prefix="Main", debug=True)
        index_reps = np.empty((0, 768))
        batch_size = 128
        tags = []
        brand_name_list = list(brand_domain_map.keys())
        for i in tqdm(range(0, len(brand_name_list), batch_size)):
            batch = brand_name_list[i:min(i + batch_size, len(brand_name_list))]  # Get the next batch of brand names
            batch_embeddings, _ = matching_model(batch)
            batch_embeddings = batch_embeddings.cpu().numpy()  # Predict embeddings for the batch
            index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
            tags.extend(batch)  # Collect tags
        np.save(REF_IDENTITY_NAMES, np.asarray(tags))
        np.save(REF_IDENTITY_REPS, index_reps)

    ref_embed_list = np.load(REF_IDENTITY_REPS) if REF_IDENTITY_REPS else None
    ref_tag_list = np.load(REF_IDENTITY_NAMES).tolist()
    brand_index_db = BaseFaissIPRetriever(init_reps=ref_embed_list, tags=ref_tag_list, embed_model=matching_model)
    brand_domain_map_path = REF_IDENTITY_MAP
    Logger.spit('Loaded the identity knowledge base into memory', caller_prefix="Main", debug=True)

    relation_embed_list = np.load(REF_RELATION_REPS) if REF_RELATION_REPS else None
    relation_tag_list = np.load(REF_RELATION_NAMES).tolist()
    internal_relation_index_db = BaseFaissIPRetriever(init_reps=relation_embed_list, tags=relation_tag_list, embed_model=matching_model)

    thre = 0.95

if __name__ == '__main__':

    '''Load identity detection model'''
    IdentityBert_MODEL = Config.identity_model
    matcher_cls = IdentityMatcher(brand_index_db=Config.brand_index_db,
                                  internal_relation_index_db=Config.internal_relation_index_db,
                                  embed_model=Config.matching_model,
                                  brand_domain_map_path=Config.brand_domain_map_path,
                                  knowledge_base_expansion=False,
                                  gpt_client=None, gpt_assistant=None,
                                  check_action=True,
                                  threshold=Config.thre)
    Logger.set_debug_on()

    ''''''
    # desc_folder = './datasets/nazario-recent'
    # desc_folder = './datasets/CSDMC2010/Ham'
    desc_folder = './datasets/GPT_V6/v6'
    dataset = EmailDataset(desc_folder)
    # csv_file_path = './datasets/nazario_results_augmented.csv'
    # csv_file_path = './datasets/CSDMC2010_benign_results.csv'
    # csv_file_path = './datasets/CSDMC2010_benign_results_augmented.csv'
    csv_file_path = './datasets/GPT_results_augmented.csv'

    # dataset = EmailBoxDataset("./datasets/All mail Including Spam and Trash.mbox")
    # csv_file_path = './datasets/sjtu_phish_results.csv'
    #
    # Check if we're writing to a new file, and write the header if so
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['email_file_path',
                             'sender_name', 'sender_address',
                             'to_names', 'to_addresses',
                             'subject',
                             'sender_identities',
                             'sender_relation',
                             'required_actions',
                             'matched_identity',
                             'our_pred',
                             'our_runtime',
                             'dfence_pred',
                             'dfence_runtime',
                             'helphed_stacking_pred',
                             'helphed_stacking_runtime',
                             'helphed_voting_pred',
                             'helphed_voting_runtime',
                             'rspamd_pred',
                             'rspamd_score',
                             'rspamd_metadata',
                             'rspamd_runtime'])

    for it in tqdm(range(len(dataset))):
        if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), reply_to_address, \
            subject, email_body_text, header = dataset[it]

        # if email_file_path != './datasets/GPT_Dataset/Giancarlo Pellegrino_Web_Zero_Gemini.eml':
        #     continue

        _, dfence_pred, dfence_runtime = dfence.inference.test(email_file_path)
        dfence_pred = dfence_pred[0]
        Logger.spit(f"D-Fence prediction = {dfence_pred} with runtime = {dfence_runtime}", debug=True, caller_prefix='D-Fence')

        helphed_stacking_pred, helphed_voting_pred, helphed_stacking_runtime, helphed_voting_runtime = helphed.inference.test(email_file_path)
        helphed_stacking_pred = helphed_stacking_pred[0]
        helphed_voting_pred = helphed_voting_pred[0]
        Logger.spit(f"HelpHed stacking prediction = {helphed_stacking_pred} with runtime = {helphed_stacking_runtime} \t"
                    f"HelpHed voting prediction = {helphed_voting_pred} with runtime = {helphed_voting_runtime}", debug=True, caller_prefix='HelpHed')

        rspamd_pred, rspamd_score, rspamd_metadata, rspamd_runtime = rspamd.inference.test(email_file_path)
        rspamd_pred = rspamd_pred[0]
        rspamd_score = rspamd_score[0]
        rspamd_metadata = rspamd_metadata[0]
        rspamd_runtime = rspamd_runtime[0]
        Logger.spit(f"Rspamd prediction = {rspamd_pred}, score = {rspamd_score} with runtime = {rspamd_runtime}", debug=True, caller_prefix='Rspamd')

        parsed_email = f'Subject: {subject} \n From: {sender_name} \n Body: {email_body_text}'
        identities, actions, relations, urls_after_actions, identity_recog_runtime = IdentityBert_MODEL(parsed_email)
        if len(urls_after_actions):
            next_step_of_engagement = urls_after_actions
        else:
            next_step_of_engagement = reply_to_address
        next_step_of_engagement_domains = dataset.domain_parsing(next_step_of_engagement)

        # Brand matching
        sender_domains = dataset.domain_parsing(sender_address)  # a set
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
                             subject,
                             identities,
                             relations,
                             actions,
                             matched_identity,
                             is_inconsistent,
                             identity_recog_runtime + identity_matching_runtime,
                             dfence_pred,
                             dfence_runtime,
                             helphed_stacking_pred,
                             helphed_stacking_runtime,
                             helphed_voting_pred,
                             helphed_voting_runtime,
                             rspamd_pred,
                             rspamd_score,
                             rspamd_metadata,
                             rspamd_runtime
                             ])



