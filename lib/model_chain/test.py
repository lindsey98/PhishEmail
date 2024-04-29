

from lib.sequence_generation.wb_evaluate import _generate_identity
from lib.sequence_classification.wb_evaluate import _generate_relation
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModelForSequenceClassification, pipeline
from email import generator
from pathlib import Path
import os
import chardet
import json
import email
from email import message_from_string
from email.utils import parseaddr
from lib.data.dataloader import Nazario, extract_sender_ip
from lib.data.utils import remove_extra_spaces, remove_urls
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
import html
from tqdm import tqdm
from time import perf_counter
import csv
import pandas as pd
from email.header import decode_header
from translate import Translator
from email.utils import parseaddr, getaddresses
import pypff
import quopri
import base64

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

class Feed(Nazario):
    def __init__(self, root_path):
        super().__init__(root_path)
        file_list = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.eml'):
                    full_path = os.path.join(root, filename)  # Join root with filename
                    file_list.append(full_path)

        self.file_list = file_list

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = message_from_string(open(email_file_path).read())

        # sender IP
        # sender_ip = extract_sender_ip(email_content._headers)
        headers = str(email_content._headers)
        # Assuming email_message is your email object
        sender_name, sender_address = parseaddr(email_content.get('From', ''))
        if len(sender_name):
            sender_name = decode_header(sender_name)
            sender_name = ''.join(part if isinstance(part, str) else part.decode(charset or 'utf-8') for part, charset in sender_name)

        # Extracting all recipients
        to_addresses = email_content.get('To', '')
        cc_addresses = email_content.get('Cc', '')  # Handling Cc if needed
        bcc_addresses = email_content.get('Bcc', '')  # Handling Bcc if needed, though Bcc should not be visible

        # Parsing multiple addresses
        all_recipients = getaddresses([to_addresses, cc_addresses, bcc_addresses])

        # Extract just the email addresses, and filter out any empty entries
        to_names = [name for name, addr in all_recipients if addr]
        to_addresses = [addr for name, addr in all_recipients if addr]

        # Joining all recipient email addresses with a comma
        to_names = ', '.join(to_names)
        to_addresses = ', '.join(to_addresses)

        subject = email_content.get('subject', '')
        if len(subject):
            subject = decode_header(subject)
            if not isinstance(subject, str):
                try:
                    subject = ''.join(part if isinstance(part, str) else part.decode(charset or 'utf-8') for part, charset in subject)
                except UnicodeDecodeError: # unicode error
                    subject = email_content.get('subject', '')

        # Check if the email message is multipart
        if email_content.is_multipart():
            text_content = ""
            # Iterate over each part of the email
            for part in email_content.get_payload():
                content_type = part.get_content_type()
                content_transfer_encoding = part.get('Content-Transfer-Encoding')

                # If part is text/plain or text/html and not an attachment, process it
                if content_type in ('text/plain', 'text/html'):
                    raw_email_content = part.get_payload()

                    if content_transfer_encoding == 'base64':
                        decoded_content = base64.b64decode(raw_email_content).decode('utf-8', 'ignore')
                    else:
                        decoded_content = quopri.decodestring(raw_email_content.encode()).decode('utf-8', 'ignore')

                    soup = BeautifulSoup(decoded_content, 'html.parser')
                    text_part = ' '.join(soup.stripped_strings)
                    text_content += text_part
        else:
            raw_email_content = email_content.get_payload()
            content_transfer_encoding = email_content.get('Content-Transfer-Encoding')

            if content_transfer_encoding == 'base64':
                decoded_content = base64.b64decode(raw_email_content).decode('utf-8', 'ignore')
            else:
                decoded_content = quopri.decodestring(raw_email_content.encode()).decode('utf-8', 'ignore')

            soup = BeautifulSoup(decoded_content, 'html.parser')
            text_part = ' '.join(soup.stripped_strings)
            text_content = text_part

        # remove text surrounded by <>, since they are likely be comments that are invisible
        text_content = re.sub(r'<[^>]*>', '', text_content)
        # replace multiple newline characters with a single newline
        text_content = re.sub(r'\n{2,}', '\n', text_content)
        # replace multiple consecutive periods with a single period
        text_content = re.sub(r'\.{2,}', '', text_content)
        # replace multiple spaces with a single space
        text_content = re.sub(r'\s+', ' ', text_content)
        # Deal with &nbsp
        text_content = re.sub(r'\xa0', ' ', text_content)
        text_content = re.sub(r'&nbsp;', ' ', text_content)
        # Deal with invisible -
        text_content = re.sub(r'\xad', '', text_content)
        text_content = re.sub(r'&shy;', '', text_content)

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                subject, \
                text_content, \
                headers


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

    '''Load relationship inference model'''
    relation_checkpoint_path = "./checkpoints/output_relation/checkpoint-llama3"
    relation_tokenizer = AutoTokenizer.from_pretrained(relation_checkpoint_path)
    relation_tokenizer.pad_token = relation_tokenizer.eos_token
    relation_model = AutoModelForCausalLM.from_pretrained(relation_checkpoint_path,
                                                          use_cache=False,
                                                          device_map="auto")
    relation_model.eval()

    with open('./lib/prompt/relationship_inference_prompt.json', 'rb') as handle:
        relation_prompt = json.load(handle)
    relation_instruction = relation_prompt[0]["content"]

    '''Load identity detection model'''
    identity_checkpoint_path = './checkpoints/output_identity/checkpoint-llama3'
    identity_tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)
    identity_tokenizer.pad_token = identity_tokenizer.eos_token
    identity_model = AutoModelForCausalLM.from_pretrained(identity_checkpoint_path,
                                                     use_cache=False,
                                                     device_map="auto")
    identity_model.eval()

    with open('./lib/prompt/identity_recognition_prompt.json', 'rb') as handle:
        identity_prompt = json.load(handle)
    identity_instruction = identity_prompt[0]["content"]


    model_id = 'meta-llama/Meta-Llama-3-8B'
    gen_config = GenerationConfig.from_pretrained(
                                                  model_id,
                                                  temperature=0.001,
                                                  max_new_tokens=100,
                                                  return_full_text=False,
                                                  )

    ''''''
    # desc_folder = './datasets/sjtu_phish'
    desc_folder = './datasets/nazario-recent'
    dataset = Feed(desc_folder)
    # csv_file_path = './datasets/sjtu_phish_results.csv'
    csv_file_path = './datasets/nazario_results.csv'

    # Check if we're writing to a new file, and write the header if so
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['email_file_path', 'sender_name', 'sender_address',
                             'to_names', 'to_addresses',
                             'subject', 'email_body_text', 'header',
                             'relation_pred',
                             'capability_pred',
                             'identity_pred',
                             'extra_response',
                             'relation_pred_time',
                             'identity_pred_time'])

    for it in tqdm(range(len(dataset))):
        if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue
        #
        # if dataset.file_list[it] not in ['./datasets/sjtu_phish/2108292.eml']:
        #     continue

        email_file_path, (sender_name, sender_address), \
        (to_names, to_addresses), \
        subject, email_body_text, header = dataset[it]

        parsed_email = f'Subject: {subject} \n From: {sender_name} \n Recipient address: {to_addresses} \n Body: {email_body_text}'

        ## infer the relationship between sender and recipient
        query_relation = prepare_prompt_no_output_unstucture(parsed_email,
                                                             tokenizer=relation_tokenizer,
                                                             instruction=relation_instruction)
        relation_results = _generate_relation(prompt=query_relation,
                                              model=relation_model,
                                              tokenizer=relation_tokenizer,
                                              gen_config=gen_config)
        match = re.search(r'^(.*?)\n', relation_results['generation'], re.DOTALL)
        relation_pred = match.group(1).strip() if match else None
        relation_pred_time = relation_results['total_time'] # todo: you can stop if this step predicts the sender is a marketers

        ## infer the sender claimed organization
        query_identity = prepare_prompt_no_output_unstucture(parsed_email,
                                                             tokenizer=identity_tokenizer,
                                                             instruction=identity_instruction)
        identity_results = _generate_identity(prompt=query_identity,
                                           model=identity_model,
                                           tokenizer=identity_tokenizer,
                                           gen_config=gen_config)

        step_1_match = re.search(r"(Step 1:.*?)(?=Step 2)", identity_results['generation'], re.DOTALL)
        step_1_text = step_1_match.group(1).strip() if step_1_match else None

        step_2_match = re.search(r"(Step 2:.*?\.)(?=\s|$)", identity_results['generation'], re.DOTALL)
        step_2_text = step_2_match.group(1).strip() if step_2_match else None

        extra_match = re.search(r"Step 2:.*?\n(.*)", identity_results['generation'], re.DOTALL)
        extra_response_text = extra_match.group(1).strip() if extra_match else None

        identity_pred_time = identity_results['total_time']

        #
        subject = subject.replace('\n', ' ') if subject else None
        email_body_text = email_body_text.replace('\n', '.')  # Preserving visual indication of newlines
        step_1_text = step_1_text.replace('\n', ' ') if step_1_text else None
        step_2_text = step_2_text.replace('\n', ' ') if step_2_text else None
        extra_response_text = extra_response_text.replace('\n', ' ') if extra_response_text else None

        # Append the new row to the CSV file
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([email_file_path, sender_name, sender_address,
                             to_names, to_addresses,
                             subject, email_body_text, header,
                             relation_pred,
                             step_1_text,
                             step_2_text,
                             extra_response_text,
                             relation_pred_time,
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


