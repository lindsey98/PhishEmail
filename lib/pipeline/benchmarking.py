
import os
from tqdm import tqdm
import csv
from lib.data import EmailDataset, EmailBoxDataset
from lib.encoder import IdentityBert
from inference import main

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
    IdentityBert_MODEL = IdentityBert("checkpoints/identity-model")

    ''''''
    # desc_folder = './datasets/nazario-recent'
    # desc_folder = './datasets/CSDMC2010/Ham'
    desc_folder = './datasets/GPT_V6/v6'
    # csv_file_path = './datasets/nazario_results_augmented.csv'
    # csv_file_path = './datasets/CSDMC2010_benign_results.csv'
    # csv_file_path = './datasets/CSDMC2010_benign_results_augmented.csv'
    csv_file_path = './datasets/GPT_results_augmented.csv'

    # dataset = EmailBoxDataset("./datasets/All mail Including Spam and Trash.mbox")
    # csv_file_path = './datasets/sjtu_phish_results.csv'
    #
    # Check if we're writing to a new file, and write the header if so
    main(desc_folder, save_vis=False, vis_dir=None, output_csv=csv_file_path, dfence=True, helphed=True, rspamd=True)
    # if not os.path.exists(csv_file_path):
    #     with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['email_file_path',
    #                          'sender_name',
    #                          'sender_address',
    #                          'to_names',
    #                          'to_addresses',
    #                          'subject',
    #                          'email_body_text',
    #                          'sender_identity',
    #                          'sender_relation',
    #                          'required_action',
    #                          'pred_time'])
    #
    # for it in tqdm(range(len(dataset))):
    #     if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
    #         continue
    #
    #     email_file_path, (sender_name, sender_address), \
    #         (to_names, to_addresses), reply_to_address, \
    #         subject, email_body_text, header = dataset[it]
    #
    #     # if email_file_path != './datasets/GPT_Dataset/Giancarlo Pellegrino_Web_Zero_Gemini.eml':
    #     #     continue
    #
    #     parsed_email = f'Subject: {subject}. From: {sender_name}. Body: {email_body_text}'
    #     identities, actions, relations, urls_after_actions, runtime = IdentityBert_MODEL(parsed_email)
    #
    #     # Append the new row to the CSV file
    #     with open(csv_file_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([email_file_path,
    #                          sender_name, sender_address,
    #                          to_names, to_addresses,
    #                          subject, email_body_text,
    #                          identities,
    #                          relations,
    #                          actions,
    #                          runtime
    #                          ])



