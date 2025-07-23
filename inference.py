
import os
import time
from tqdm import tqdm
import csv
from lib.encoder import IdentityBert, Visualizer
from lib.reference_db import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
from lib.utilities import Logger, pst_to_eml, mbox_to_eml
from lib.data import RenderDataset, OCR
import numpy as np
from datetime import datetime
from pathlib import Path
import click
# from lib.baselines import dfence, helphed
import json
from lib.utilities.data_utils import DomainUtils


class Config:

    Internal_Relations = [
        'admin',
        'mail admin',
        'webmail mail service team',
        'admin portal',
        'administrator',
        'administration',
        'administrative',
        'mail server administrator',
        'mail delivery system',
        'employee',
        'staff',
        'colleague',
        'colleagues',
        'student',
        'human resource team',
        'HR team',
        'HR',
        'human resources department',
        'Finance team'
        'IT department',
        'IT support',
        'IT service support',
        'Payroll',
        'IT maintenance services',
        'IT report',
        'IT Desk',
        'internal company',
        # the following internal roles are observed in field study
        'billing support',
        'helpdesk',
        'IT helpdesk',
        'Postmaster',
        'IMAP',
        'Internal company',
        'administration department',
        'mail inc.',
        'email security center',
        'financial management department',
        'general affairs department',
        'finance department',
        'human resources department',
        'email system network record',
        'ministry of information',
        'email system management notification',
        'trusted server',
        'oa data migration and upgrade',
        'information management center',
        'the mail system',
        'support IT',
        'mailbox administrator',
        'enterprise mailbox',
        'lending department',
        'system administrator',
        'accounts payable team',
        'management department',
        'service information center',
        'e-mail mailbox system',
        "security administrator",
        "account administrator",
        "maintenance department",
        "IT teams",
        "campus email",
        "network and data center",
        "post office system administrator",
        "mygov",
        "security center",
        "tomHRM",
        "home depot department",
        "mail inc."
    ]

    # IDENTITY_MODEL_CHECKPOINT = os.getenv("IDENTITY_MODEL_CHECKPOINT", "./checkpoints/identity_adversarial_training")
    IDENTITY_MODEL_CHECKPOINT = os.getenv("IDENTITY_MODEL_CHECKPOINT", "./checkpoints/identity-model")

    MATCHING_MODEL_CHECKPOINT = os.getenv("MATCHING_MODEL_CHECKPOINT", "./checkpoints/characterbert-typos-st-adv")

    REF_IDENTITY_REPS = os.getenv("REF_IDENTITY_REPS", "./checkpoints/company_database_reps_field_study.npy")
    REF_IDENTITY_NAMES = os.getenv("REF_IDENTITY_NAMES", "./checkpoints/company_database_names_field_study.npy")
    REF_IDENTITY_MAP = os.getenv("REF_IDENTITY_MAP", "./checkpoints/company_database_field_study.json")

    REF_RELATION_REPS = os.getenv("REF_RELATION_REPS", "./checkpoints/internal_relation_reps.npy")
    REF_RELATION_NAMES = os.getenv("REF_RELATION_NAMES", "./checkpoints/internal_relation_names.npy")

    identity_model = IdentityBert(IDENTITY_MODEL_CHECKPOINT)
    visualizer_model = Visualizer(IDENTITY_MODEL_CHECKPOINT)
    matching_model = CharacterBERT(MATCHING_MODEL_CHECKPOINT)

    Logger.spit('Loaded the identity recognition model and identity matching model into memory', caller_prefix="Main", debug=True)

    with open(REF_IDENTITY_MAP, 'r') as file:
        brand_domain_map = json.load(file)
    ref_embed_list = np.load(REF_IDENTITY_REPS) if os.path.exists(REF_IDENTITY_REPS) else None

    index_reps = np.empty((0, 768))
    batch_size = 128
    tags = []
    brand_name_list = list(brand_domain_map.keys())
    for i in tqdm(range(0, len(brand_name_list), batch_size), desc="Caching the knowledge base embeddings"):
        batch = brand_name_list[i:min(i + batch_size, len(brand_name_list))]  # Get the next batch of brand names
        batch_embeddings, _ = matching_model(batch)
        batch_embeddings = batch_embeddings.cpu().numpy()  # Predict embeddings for the batch
        index_reps = np.concatenate((index_reps, batch_embeddings), axis=0)  # Append new embeddings
        tags.extend(batch)  # Collect tags
    np.save(REF_IDENTITY_NAMES, np.asarray(tags))
    np.save(REF_IDENTITY_REPS, index_reps)

    tags = [x.lower() for x in Internal_Relations]
    embed, _ = matching_model(tags)
    embed = embed.cpu().numpy()
    np.save(REF_RELATION_NAMES, np.asarray(tags))
    np.save(REF_RELATION_REPS, embed)

    Logger.spit('Loaded the identity knowledge base into memory', caller_prefix="Main", debug=True)
    ref_embed_list = np.load(REF_IDENTITY_REPS) if REF_IDENTITY_REPS else None
    ref_tag_list = np.load(REF_IDENTITY_NAMES).tolist()
    brand_index_db = BaseFaissIPRetriever(init_reps=ref_embed_list, tags=ref_tag_list, embed_model=matching_model)
    brand_domain_map_path = REF_IDENTITY_MAP

    relation_embed_list = np.load(REF_RELATION_REPS) if REF_RELATION_REPS else None
    relation_tag_list = np.load(REF_RELATION_NAMES).tolist()
    internal_relation_index_db = BaseFaissIPRetriever(init_reps=relation_embed_list, tags=relation_tag_list, embed_model=matching_model)

    thre = 0.95

    whitelist_senders = set([x.strip() for x in open("whitelist_senders.txt").readlines()])  # ignore the paper submission, online course, and the internal senders
    forbidden_subject_prefix = ["re:", "fwd:", "fw:", "回复:", "转发:", "reply:"]


TODAY = datetime.today()
TODAY_DATE = TODAY.strftime("%Y-%m-%d")


@click.command()
@click.option("--email_dir", help="Dir containing all the .eml files", required=True, type=str)
@click.option("--save_vis", help="Save the visualized results or not", is_flag=True, show_default=True, default=False)
@click.option("--vis_dir", help="Where to save the visualized result", default='./datasets/vis', type=str)
@click.option("--output_csv", default=f'{TODAY_DATE}_results.csv', help="Output txt path")
@click.option("--run_dfence", is_flag=True, default=False)
@click.option("--run_helphed", is_flag=True, default=False)
@click.option("--auto_translate", is_flag=True, default=False, help='Whether to translate the email (including subject)')
def main(email_dir, save_vis, vis_dir, output_csv, run_dfence, run_helphed, auto_translate):
    '''
    PiMRef main inference function
    :param email_dir: a directory containing all eml files, also support .mbox and .pst format
    :param save_vis: whether to save the visualized results or not
    :param vis_dir: where to save the visualized result
    :param output_csv: where to save the output
    :param run_dfence: whether to run Dfence as well
    :param run_helphed: whether to run helphed as well
    :param auto_translate: whether to translate the email
    :return:
    '''
    matcher_cls = IdentityMatcher(brand_index_db=Config.brand_index_db,
                                  internal_relation_index_db=Config.internal_relation_index_db,
                                  embed_model=Config.matching_model,
                                  brand_domain_map_path=Config.brand_domain_map_path,
                                  knowledge_base_expansion=False,
                                  gpt_client=None,
                                  gpt_assistant=None,
                                  check_action=True,
                                  threshold=Config.thre,
                                  relax_match=True)  # todo: relax_match!

    if '.mbox' in email_dir:
        mbox_to_eml(email_dir, email_dir.replace('.mbox', ''))
        desc_folder = email_dir.replace('.mbox', '')
    elif email_dir.endswith('.pst'):
        pst_to_eml(email_dir, email_dir.replace('.pst', ''))
        desc_folder = email_dir.replace('.pst', '')
    else:
        desc_folder = email_dir

    ocr_model = OCR()
    dataset = RenderDataset(desc_folder, ocr_model=ocr_model, dumpDir=desc_folder + '_imgs', save_imgs=False, translate_on=auto_translate)

    csv_file_path = output_csv
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    Logger.set_debug_on()
    Logger.spit(f'Loaded the testing dataset = {len(dataset)} emails', caller_prefix="Main", debug=True)

    seen_subjects = set()
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
                             'helphed_voting_runtime'
                             ])
    else:
        # If it exists, load existing subjects to avoid duplicates
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)   # Skip header
            for row in reader:
                if row:
                    seen_subjects.add(row[1] + row[5])   # Assuming 'subject' is the 6th column (index 5)

    for it in tqdm(range(len(dataset))):

        if os.path.exists(csv_file_path) and dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue

        # if dataset.file_list[it] != 'datasets/field/silas/inbox/mbox/2099076.eml':
        #     continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), reply_to_address, \
            subject, email_body_text, header = dataset[it]  # fixme: the GoogleTranslator may take some time
        sender_domains = dataset.domain_parsing(sender_address)  # a set

        # Check if the subject has been seen before
        # if str(sender_name) + str(subject) in seen_subjects:
        #     continue   # Skip
        seen_subjects.add(str(sender_name) + str(subject))

        '''Baseline: D-Fence, HelpHed, Rspamd'''
        # if run_dfence:
        #     _, dfence_pred, dfence_runtime = dfence.inference.test(email_file_path)
        #     dfence_pred = dfence_pred[0]
        #     Logger.spit(f"D-Fence prediction = {dfence_pred} with runtime = {dfence_runtime}", debug=True,
        #                 caller_prefix='D-Fence')
        #
        # else:
        dfence_pred, dfence_runtime = None, None

        # if run_helphed:
        #     helphed_stacking_pred, helphed_voting_pred, helphed_stacking_runtime, helphed_voting_runtime = helphed.inference.test(email_file_path)
        #     helphed_stacking_pred = helphed_stacking_pred[0]
        #     helphed_voting_pred = helphed_voting_pred[0]
        #     Logger.spit(
        #         f"HelpHed stacking prediction = {helphed_stacking_pred} with runtime = {helphed_stacking_runtime} \t"
        #         f"HelpHed voting prediction = {helphed_voting_pred} with runtime = {helphed_voting_runtime}",
        #         debug=True, caller_prefix='HelpHed')
        # else:
        helphed_stacking_pred, helphed_voting_pred, helphed_stacking_runtime, helphed_voting_runtime = None, None, None, None

        '''Our method'''
        if subject and any([x in subject.lower() for x in Config.forbidden_subject_prefix]):
            identities = []
            relations = []
            actions = set()
            next_step_of_engagement = set()
            matched_identity = "Non-original email (Reply/Forward)"
            is_inconsistent = False
            identity_recog_runtime = 0
            identity_matching_runtime = 0
            dfence_pred = 0
            dfence_runtime = 0
            helphed_stacking_pred = 0
            helphed_stacking_runtime = 0
            helphed_voting_pred = 0
            helphed_voting_runtime = 0
            Logger.spit("Non-original email (Reply/Forward)", debug=True, caller_prefix='Main')
        elif DomainUtils.domain_set_overlap(sender_domains, Config.whitelist_senders):
            identities = []
            relations = []
            actions = set()
            next_step_of_engagement = set()
            matched_identity = "Sent from an whitelisted address"
            is_inconsistent = False
            identity_recog_runtime = 0
            identity_matching_runtime = 0
            dfence_pred = 0
            dfence_runtime = 0
            helphed_stacking_pred = 0
            helphed_stacking_runtime = 0
            helphed_voting_pred = 0
            helphed_voting_runtime = 0
            Logger.spit("Sent from an whitelisted address", debug=True, caller_prefix='Main')
        else:
            # Identity recognition
            parsed_email = f'Subject: {subject} \n From: {sender_name} \n Body: {email_body_text}'
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
            recipient_domains = dataset.domain_parsing(to_addresses)
            sender_domains = sender_domains.union(next_step_of_engagement_domains)

            is_inconsistent, matched_identity, identity_matching_runtime = matcher_cls(identities=identities,
                                                                                       actions=actions,
                                                                                       relations=relations,
                                                                                       sender_domains=sender_domains,
                                                                                       recipient_domains=recipient_domains)

            # exit()

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
                                 ])
            except Exception as e:
                Logger.spit(f"Encounter error {e} when inferencing on {email_file_path}", warning=True, caller_prefix='Main')
                continue

        time.sleep(0.001)
        # fixme: scanning the sequential data too fast will cause faiss.IndexFlatIP fails to identity a match (even if the match is there)??


if __name__ == '__main__':

    Logger.set_debug_on()
    main()