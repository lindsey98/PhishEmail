
import os
from tqdm import tqdm
import csv
from lib.encoder.IdentityBert import IdentityBert
from lib.reference_db.BrandMatcher import CharacterBERT, BrandMatcher, BaseFaissIPRetriever
import numpy as np
from lib.data.dataloader import EmailDataset
import argparse
from datetime import datetime
from tldextract import tldextract

if __name__ == '__main__':

    today = datetime.now().strftime('%Y%m%d')

    parser = argparse.ArgumentParser()
    parser.add_argument("--identity_model", help="Path to the identity prediction model", default="./checkpoints/output_ner_augmented/checkpoint-1351", type=str)
    parser.add_argument("--matching_model", help="Path to the brand matching model", default="./checkpoints/characterbert-typos-st", type=str)
    parser.add_argument("--brand_domain_map_path", help="Path to the brand-email knowledge base", default="./datasets/company_database_knowphish.json", type=str)

    parser.add_argument("--ref_embed_list", help="Path to the pre-saved embeddings for the referenced brand names",
                        default="./datasets/company_database_reps.npy", type=str)
    parser.add_argument("--ref_tag_list", help="List of the referenced brand names",
                        default="./datasets/company_database_names.npy", type=str)

    parser.add_argument("--relation_embed_list", help="Path to the pre-saved embeddings for the internal relations",
                        default="./datasets/internal_relation_reps.npy", type=str)
    parser.add_argument("--relation_tag_list", help="List of the internal relations",
                        default="./datasets/internal_relation_names.npy", type=str)


    parser.add_argument("--thre", "-t", help="Brand matching threshold", default=0.95, type=float)

    parser.add_argument("--email_dir", help="Dir containing all the .eml files", required=True, type=str)
    parser.add_argument("--output_csv", default=f'{today}_results.csv', help="Output txt path")
    args = parser.parse_args()

    '''Load models'''
    identity_model = IdentityBert(args.identity_model)
    matching_model = CharacterBERT(args.matching_model)
    ref_embed_list = np.load(args.ref_embed_list)
    ref_tag_list = np.load(args.ref_tag_list).tolist()
    brand_index_db = BaseFaissIPRetriever(init_reps=ref_embed_list, tags=ref_tag_list)

    relation_embed_list = np.load(args.relation_embed_list)
    relation_tag_list = np.load(args.relation_tag_list).tolist()
    internal_relation_index_db = BaseFaissIPRetriever(init_reps=relation_embed_list, tags=relation_tag_list)

    matcher_cls = BrandMatcher(brand_index_db=brand_index_db,
                                internal_relation_index_db=internal_relation_index_db,
                                embed_model=matching_model,
                                brand_domain_map_path=args.brand_domain_map_path,
                                knowledge_base_expansion=False,
                                gpt_client=None, gpt_assistant=None,
                                check_action=True,
                                threshold=args.thre)

    dataset = EmailDataset(args.email_dir)
    csv_file_path = args.output_csv

    for it in tqdm(range(len(dataset))):

        if os.path.exists(csv_file_path) and dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), \
            subject, email_body_text, header = dataset[it]

        # Identity recognition
        parsed_email = f'Subject: {subject}. From: {sender_name}. Body: {email_body_text}'
        identities, actions, relations, runtime = identity_model(parsed_email)

        # Brand matching
        if isinstance(sender_address, float):
            sender_domain = None
        else:
            invalid_address = '@' not in sender_address
            if invalid_address:
                sender_domain = None
            else:
                sender_domain = sender_address.split('@')[-1]
                sender_domain = tldextract.extract(sender_domain).domain + '.' + tldextract.extract(sender_domain).suffix

        if isinstance(to_addresses, float):
            recipient_domains = None
        else:
            recipient_domains = [x.split('@')[-1] for x in to_addresses.split(',')]
            recipient_domains = [tldextract.extract(x).domain+'.'+tldextract.extract(x).suffix for x in recipient_domains]

        is_inconsistent, matched_brand = matcher_cls(identities=identities,
                                    actions=actions,
                                    relations=relations,
                                    sender_domain=sender_domain,
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
                             is_inconsistent,
                             matched_brand,
                             runtime
                             ])



