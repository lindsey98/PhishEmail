
import os
import time

from tqdm import tqdm
import csv
from lib.encoder import IdentityBert, Visualizer
from lib.reference_db import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
from lib.utilities import Logger
from lib.data import EmailDataset, EmailBoxDataset
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import click
from lib.baselines import dfence, helphed, rspamd
import json

class Config:
    IDENTITY_MODEL_CHECKPOINT = os.getenv("IDENTITY_MODEL_CHECKPOINT", "./checkpoints/identity-model")
    MATCHING_MODEL_CHECKPOINT = os.getenv("MATCHING_MODEL_CHECKPOINT", "./checkpoints/characterbert-typos-st")

    REF_IDENTITY_REPS = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_reps.npy")
    REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_names.npy")
    REF_IDENTITY_MAP = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_knowphish.json") # todo: this is the compact version of knowledge base (manually cleaned) with <1k brands

    REF_RELATION_REPS = os.getenv("REF_RELATION_REPS", "./checkpoints/internal_relation_reps.npy")
    REF_RELATION_NAMES = os.getenv("REF_RELATION_NAMES", "./checkpoints/internal_relation_names.npy")

    identity_model = IdentityBert(IDENTITY_MODEL_CHECKPOINT)
    visualizer_model = Visualizer(IDENTITY_MODEL_CHECKPOINT)
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

    
today = datetime.today()
today_date = today.strftime("%Y-%m-%d")
@click.command()
@click.option("--email_dir", help="Dir containing all the .eml files", required=True, type=str)
@click.option("--save_vis", help="Save the visualized results or not", is_flag=True, show_default=True, default=False)
@click.option("--vis_dir", help="Where to save the visualized result", default='./datasets/vis', type=str)
@click.option("--output_csv", default=f'{today_date}_results.csv', help="Output txt path")
@click.option("--run_dfence", is_flag=True)
@click.option("--run_helphed", is_flag=True)
@click.option("--run_rspamd", is_flag=True)
def main(email_dir, save_vis, vis_dir, output_csv, run_dfence, run_helphed, run_rspamd):
    matcher_cls = IdentityMatcher(brand_index_db=Config.brand_index_db,
                                  internal_relation_index_db=Config.internal_relation_index_db,
                                  embed_model=Config.matching_model,
                                  brand_domain_map_path=Config.brand_domain_map_path,
                                  knowledge_base_expansion=False,
                                  gpt_client=None, gpt_assistant=None,
                                  check_action=True,
                                  threshold=Config.thre)

    if email_dir.endswith('.mbox'):
        dataset = EmailBoxDataset(email_dir)
    else:
        dataset = EmailDataset(email_dir)
    csv_file_path = output_csv
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    Logger.set_debug_on()
    Logger.spit('Loaded the testing dataset into memory', caller_prefix="Main", debug=True)
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
                             'next_step_of_engagement',
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
                             'rspamd_runtime'
                             ])

    for it in tqdm(range(len(dataset))):

        # if it <= 9173:
        #     continue
        if os.path.exists(csv_file_path) and dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue
        # if dataset.file_list[it] != 'datasets/sjtu_phish/email_154.eml':
        #     continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), reply_to_address, \
            subject, email_body_text, header = dataset[it] ## fixme: the GoogleTranslator takes time

        '''Baseline: D-Fence, HelpHed, Rspamd'''
        if run_dfence:
            _, dfence_pred, dfence_runtime = dfence.inference.test(email_file_path)
            dfence_pred = dfence_pred[0]
            Logger.spit(f"D-Fence prediction = {dfence_pred} with runtime = {dfence_runtime}", debug=True,
                        caller_prefix='D-Fence')

        else:
            dfence_pred, dfence_runtime = None, None

        if run_helphed:
            helphed_stacking_pred, helphed_voting_pred, helphed_stacking_runtime, helphed_voting_runtime = helphed.inference.test(email_file_path)
            helphed_stacking_pred = helphed_stacking_pred[0]
            helphed_voting_pred = helphed_voting_pred[0]
            Logger.spit(
                f"HelpHed stacking prediction = {helphed_stacking_pred} with runtime = {helphed_stacking_runtime} \t"
                f"HelpHed voting prediction = {helphed_voting_pred} with runtime = {helphed_voting_runtime}",
                debug=True, caller_prefix='HelpHed')
        else:
            helphed_stacking_pred, helphed_voting_pred, helphed_stacking_runtime, helphed_voting_runtime = None, None, None, None

        if run_rspamd:
            rspamd_pred, rspamd_score, rspamd_metadata, rspamd_runtime = rspamd.inference.test(email_file_path)
            rspamd_pred = rspamd_pred[0]
            rspamd_score = rspamd_score[0]
            rspamd_metadata = rspamd_metadata[0]
            rspamd_runtime = rspamd_runtime[0]
            Logger.spit(f"Rspamd prediction = {rspamd_pred}, score = {rspamd_score} with runtime = {rspamd_runtime}",
                        debug=True, caller_prefix='Rspamd')

        else:
            rspamd_pred, rspamd_score, rspamd_metadata, rspamd_runtime = None, None, None, None

        '''Our method'''
        # Identity recognition
        parsed_email = f'Subject: {subject} \n  From: {sender_name} \n Body: {email_body_text}'
        identities, actions, relations, urls_after_actions, identity_recog_runtime = Config.identity_model(parsed_email)
        if save_vis:
            html = Config.visualizer_model(parsed_email, metadata=email_file_path)
            # Extract the second-level directory name and the base file name from the file path
            email_file_path_obj = Path(email_file_path)
            email_file_path_2nd_level_basename = email_file_path_obj.parts[-2]  # Assuming the file path has at least two directory levels
            email_file_path_basename = email_file_path_obj.stem  # The file name without the extension

            html_file_path = Path(vis_dir) / f"{email_file_path_2nd_level_basename}_{email_file_path_basename}.html"
            with open(html_file_path, 'w') as f:
                f.write(html)

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
            try:
                writer.writerow([email_file_path,
                                 sender_name, sender_address,
                                 to_names, to_addresses,
                                 subject,
                                 identities,
                                 relations,
                                 actions,
                                 next_step_of_engagement,
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
            except:
                continue

        time.sleep(0.001)
        # fixme: scanning the sequential data too fast will cause faiss.IndexFlatIP fails to identity a match (even if the match is there)??


if __name__ == '__main__':

    Logger.set_debug_on()
    main()

    # "google-bert/bert-large-uncased" =>  340 million parameters.
    # character-bert => 105 million parameters
    # 5000 MiB


