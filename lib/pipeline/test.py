
import os
from tqdm import tqdm
import csv
from lib.data.dataloader import EmailDataset
from lib.model_utils.postprocessing import ner_clean_predictions, CustomNERPipeline
from transformers import pipeline
import time

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


if __name__ == '__main__':

    '''Load identity detection model'''
    identity_checkpoint_path = "./checkpoints/output_ner/checkpoint-1776"
    classifier = pipeline("ner", model=identity_checkpoint_path, device=0, aggregation_strategy="simple")

    ''''''
    # desc_folder = './datasets/nazario-recent'
    desc_folder = './datasets/CSDMC2010/Ham'
    dataset = EmailDataset(desc_folder)
    # csv_file_path = './datasets/nazario_results.csv'
    csv_file_path = './datasets/CSDMC2010_benign_results.csv'

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
                             'sender_identity',
                             'sender_relation',
                             'required_action',
                             'pred_time'])

    for it in tqdm(range(len(dataset))):
        # if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
        #     continue

        email_file_path, (sender_name, sender_address), \
        (to_names, to_addresses), \
        subject, email_body_text, header = dataset[it]

        parsed_email = f'Subject: {subject}. From: {sender_name}. Body: {email_body_text}'

        start_time = time.time()
        entities = classifier(parsed_email)
        runtime = time.time() - start_time

        # cleaned_output = ner_clean_predictions(output, parsed_email)
        # entities = cleaned_output['ents']

        identities = set()
        relations = set()
        actions = set()

        for ent in entities:
            ent_label = ent['entity_group']
            ent_text = ent['word']
            if ent_label == 'identity':
                identities.add(ent_text)
            elif ent_label == 'relation':
                relations.add(ent_text)
            else:
                actions.add(ent_text)

        # Append the new row to the CSV file
        # with open(csv_file_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([email_file_path,
        #                      sender_name, sender_address,
        #                      to_names, to_addresses,
        #                      subject, email_body_text,
        #                      identities,
        #                      relations,
        #                      actions,
        #                      runtime
        #                      ])



