import os
import time
from tqdm import tqdm
import csv
from lib.reference_db import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
from lib.utilities import Logger, pst_to_eml, mbox_to_eml
from lib.data import RenderDataset, OCR
from datetime import datetime
from pathlib import Path
import click
from lib.baselines import dfence, helphed
from lib.utilities.data_utils import DomainUtils
from config import Config

TODAY = datetime.today()
TODAY_DATE = TODAY.strftime("%Y-%m-%d")
TODAY_STR = TODAY.strftime("%Y-%m-%d_%H%M%S")  # avoid collision when rerunning same day


CSV_HEADER = [
    'email_file_path',
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
]


def _load_existing_rows(csv_path: str):
    """Return (seen_paths, seen_subjects) from a previous run's CSV.

    Uses csv.reader so that quoted fields containing commas (e.g. identities
    list, multi-recipient to_addresses) are parsed correctly. Skips the
    header row.
    """
    seen_paths: set = set()
    seen_subjects: set = set()
    if not os.path.exists(csv_path):
        return seen_paths, seen_subjects
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header
        except StopIteration:
            return seen_paths, seen_subjects
        for row in reader:
            if not row:
                continue
            seen_paths.add(row[0])
            if len(row) >= 6:
                seen_subjects.add(str(row[1]) + str(row[5]))
    return seen_paths, seen_subjects


@click.command()
@click.option("--email_dir", help="Dir containing all the .eml files", required=True, type=str)
@click.option("--save_vis", help="Save the visualized results or not", is_flag=True, show_default=True, default=False)
@click.option("--vis_dir", help="Where to save the visualized result", default='./datasets/vis', type=str)
@click.option("--output_csv", default=f'{TODAY_STR}_results.csv', help="Output txt path")
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
                                  knowledge_base_expansion=Config.knowledge_expansion_on,
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

    # ------------------------------------------------------------------ #
    seen_paths, seen_subjects = _load_existing_rows(csv_file_path)

    # Write header if the file is new or empty.
    file_is_new = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0
    if file_is_new:
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)
    else:
        Logger.spit(
            f"Resuming from {csv_file_path}: {len(seen_paths)} rows already present",
            caller_prefix="Main", debug=True,
        )

    for it in tqdm(range(len(dataset))):
        email_file_path_iter = dataset.file_list[it]

        # Fast in-memory skip — no more per-iteration file reads.
        if email_file_path_iter in seen_paths:
            continue

        # if email_file_path_iter != 'datasets/field/silas/inbox/mbox/2099076.eml':
        #     continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), reply_to_address, \
            subject, email_body_text, header = dataset[it]  # fixme: the GoogleTranslator may take some time
        sender_domains = dataset.domain_parsing(sender_address)  # a set

        # Subject-level de-dup (re-enable if desired).
        subj_key = str(sender_name) + str(subject)
        # if subj_key in seen_subjects:
        #     continue
        seen_subjects.add(subj_key)

        '''Baseline: D-Fence, HelpHed, Rspamd'''
        if run_dfence:
            _, dfence_pred, dfence_runtime = dfence.inference.test(email_file_path)
            dfence_pred = dfence_pred[0]
            Logger.spit(f"D-Fence prediction = {dfence_pred} with runtime = {dfence_runtime}",
                        debug=True,
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
                debug=True,
                caller_prefix='HelpHed')
        else:
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
                email_file_path_obj = Path(email_file_path)
                email_file_path_2nd_level_basename = email_file_path_obj.parts[-2]
                email_file_path_basename = email_file_path_obj.stem
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

        # Mark as written — so a crash/resume within the same run doesn't redo it.
        seen_paths.add(email_file_path)

        time.sleep(0.001)


if __name__ == '__main__':
    Logger.set_debug_on()
    main()