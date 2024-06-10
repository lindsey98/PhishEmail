

from lib.sequence_generation.wb_evaluate import _generate_identity
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
import chardet
import json
from lib.data.utils import remove_extra_spaces, remove_urls
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
import html
from tqdm import tqdm
from time import perf_counter
import csv
import pandas as pd
from lib.data.dataloader import EmailDataset
import torch

def decode_body(body_bytes):
    detected_encoding = chardet.detect(body_bytes)['encoding']

    # Define a list of common encodings to try if detection fails or results in common but incorrect choices
    common_encodings = ['utf-8', 'iso-8859-1', 'latin-1', 'ascii', 'gbk', 'shift_jis']

    # Try the detected encoding first
    if detected_encoding:
        try:
            decoded_str = body_bytes.decode(detected_encoding)
            print(f"Successfully decoded using detected encoding: {detected_encoding}")
            return decoded_str
        except UnicodeDecodeError:
            print(f"Failed to decode using detected encoding: {detected_encoding}")

    # Fallback to common encodings if the detected encoding fails or wasn't useful
    for encoding in common_encodings:
        try:
            decoded_str = body_bytes.decode(encoding)
            print(f"Successfully decoded using fallback encoding: {encoding}")
            return decoded_str
        except UnicodeDecodeError:
            continue

    # As a last resort, decode with 'utf-8' using 'replace' to handle undecodable bytes
    print("Decoding with utf-8 and replacing undecodable bytes.")
    return body_bytes.decode('utf-8', errors='replace')


def message_to_eml(message):
    if message.html_body:
        body = decode_body(message.html_body)
    elif message.plain_text_body:
        body = decode_body(message.plain_text_body)
    else:
        body = ''

    if message.transport_headers:
        return message.transport_headers + body
    else:
        return body

def pst_to_eml(source_folder, desc_folder):
    os.makedirs(desc_folder, exist_ok=True)

    for sub_folder in source_folder.sub_folders:
        pst_to_eml(sub_folder, desc_folder)

    for message in source_folder.sub_messages:
        filename = f"{desc_folder}/{message.identifier}.eml"
        if message.identifier != 2106692:
            continue
        message_eml = message_to_eml(message)

        # Write the JSON string to a file
        if len(message_eml) == 0:
            continue
        with open(filename, 'w', encoding='utf-8') as f: # must set this encoding
            f.write(message_eml)


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

def prepare_prompt_no_output_unstucture(raw_input,
                                        instruction,
                                        tokenizer,
                                        max_seq_len=4096):
    input = remove_extra_spaces(remove_urls(raw_input))
    length_predefined_prompt = len(tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

    # Tokenize the input and truncate to the max_seq_len
    tokens = tokenizer.tokenize(input)[:max_seq_len-length_predefined_prompt-50]  # Reserving space for an EOS token, if necessary
    cleaned_input = tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

    return f"### Instruction:\n{instruction}\n\n### Input:\n{cleaned_input}\n\n### Response:\n"

if __name__ == '__main__':

    # # '''Read the pst file => Convert to individual eml'''
    # pst_file = pypff.open("./datasets/backup.pst")
    # root = pst_file.get_root_folder()
    # Process all folders and subfolders
    # pst_to_eml(root, desc_folder='./datasets/sjtu_phish')
    # pst_file.close()

    '''Load identity detection model'''
    # identity_checkpoint_path = './checkpoints/output_identity/checkpoint-llama3'
    identity_checkpoint_path = './checkpoints/output_identity/checkpoint-llama3-spamenron'
    identity_tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)
    identity_tokenizer.pad_token = identity_tokenizer.eos_token
    identity_model = AutoModelForCausalLM.from_pretrained(identity_checkpoint_path,
                                                         use_cache=False,
                                                         device_map="auto",
                                                         torch_dtype=torch.bfloat16,
                                                        )

    identity_model.eval()

    with open('./lib/llm_utils/identity_recognition_prompt.json', 'rb') as handle:
        identity_prompt = json.load(handle)
    identity_instruction = identity_prompt[0]["content"]


    model_id = 'meta-llama/Meta-Llama-3-8B'
    gen_config = GenerationConfig.from_pretrained(
                                                  model_id,
                                                  temperature=0.001,
                                                  max_new_tokens=75,
                                                  return_full_text=False,
                                                  )

    ''''''
    # desc_folder = './datasets/sjtu_phish'
    desc_folder = './datasets/nazario-recent'
    # desc_folder = './datasets/CSDMC2010/Ham'
    dataset = EmailDataset(desc_folder)
    # csv_file_path = './datasets/sjtu_phish_results.csv'
    # csv_file_path = './datasets/nazario_results.csv'
    # csv_file_path = './datasets/CSDMC2010_benign_results.csv'
    # csv_file_path = './datasets/sjtu_phish_results_orig.csv'
    csv_file_path = './datasets/nazario_results_spamenron.csv'
    # csv_file_path = './datasets/CSDMC2010_benign_results_spamenron.csv'

    # Check if we're writing to a new file, and write the header if so
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['email_file_path', 'sender_name', 'sender_address',
                             'to_names', 'to_addresses',
                             'subject', 'email_body_text',
                             'header',
                             'identity_pred',
                             'action_pred',
                             'identity_pred_time'])

    for it in tqdm(range(len(dataset))):
        if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue

        # if dataset.file_list[it] not in ['./datasets/nazario-recent/2021/66.eml']:
        #     continue

        email_file_path, (sender_name, sender_address), \
        (to_names, to_addresses), \
        subject, email_body_text, header = dataset[it]

        parsed_email = f'Subject: {subject} \n From: {sender_name} \n Recipient address: {to_addresses} \n Body: {email_body_text}'

        ## infer the sender claimed organization
        query_identity = prepare_prompt_no_output_unstucture(parsed_email,
                                                             tokenizer=identity_tokenizer,
                                                             instruction=identity_instruction)
        identity_results = _generate_identity(prompt=query_identity,
                                           model=identity_model,
                                           tokenizer=identity_tokenizer,
                                           gen_config=gen_config)

        step_1_match = re.search(r"((Step 1|S1):.*?)(?=(Step 2|S2):|$)", identity_results['generation'], re.DOTALL | re.IGNORECASE)
        step_1_text = step_1_match.group(1).strip() if step_1_match else None

        # Match either "Step 2" or "S2"
        step_2_match = re.search(r"(Step 2|S2):.*", identity_results['generation'], re.DOTALL | re.IGNORECASE)
        step_2_text = step_2_match.group(0).strip() if step_2_match else None

        identity_pred_time = identity_results['total_time']
        subject = subject.replace('\n', ' ') if subject else None
        email_body_text = email_body_text.replace('\n', '.')  # Preserving visual indication of newlines
        step_1_text = step_1_text.replace('\n', ' ') if step_1_text else None
        step_2_text = step_2_text.replace('\n', ' ') if step_2_text else None

        # Append the new row to the CSV file
        with open(csv_file_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
            writer = csv.writer(file)
            writer.writerow([email_file_path, sender_name, sender_address,
                             to_names, to_addresses,
                             subject, email_body_text, header,
                             step_1_text,
                             step_2_text,
                             identity_pred_time
                             ])


    # df = pd.read_csv(csv_file_path)
    # print(df)

    '''Add new column for rspamd prediction'''
    # df = pd.read_csv(csv_file_path)
    # rspamd_prediction_list = []
    # for it, row in df.iterrows():
    #     email_content = message_from_string(open(row['email_file_path']).read())
    #     rspamd_prediction = email_content.get('X-Spam-Status', 'No').split(',')[0].strip()
    #     rspamd_prediction_list.append(rspamd_prediction)
    #
    # df['rspamd_prediction'] = rspamd_prediction_list
    # df.to_csv(csv_file_path, index=False)


