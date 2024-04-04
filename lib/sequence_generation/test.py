

from lib.sequence_generation.wb_evaluate import _generate
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
from email import generator
from pathlib import Path
import pypff
import os
import chardet
import json
import email
from email import message_from_string
from email.utils import parseaddr
from lib.data.dataloader import Nazario, extract_sender_ip
from lib.data.utils import remove_specific_special_chars, process_email_parts, parse_email_content
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
import html
from tqdm import tqdm
from time import perf_counter
import csv
import pandas as pd


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
        if message.identifier != 2102084:
            continue
        message_eml = message_to_eml(message)

        # Write the JSON string to a file
        if len(message_eml) == 0:
            continue
        with open(filename, 'w', encoding='utf-8') as f: # must set this encoding
            f.write(message_eml)

def prepare_prompt(raw_input, max_seq_len=500):
    instruction = "Step 1: Identify the claimed capabilities of the sender. Step 2: Identify the claimed organization. Step 3: Infer the role of the sender inside this organization: department or title. For Step 1 and Step 2, give an explanation by quoting the most informative phrases from the original paragraph."  # shared instruciton
    length_predefined_prompt = len(tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

    # Tokenize the input and truncate to the max_seq_len
    tokens = tokenizer.tokenize(raw_input)[:max_seq_len-length_predefined_prompt]  # Reserving space for an EOS token, if necessary
    cleaned_input = tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

    return f"### Instruction:\n{instruction}\n\n### Input:\n{cleaned_input}\n\n### Response:\n"

class Feed(Nazario):
    def __init__(self, root_path):
        super().__init__(root_path)
        file_list = []

        for filename in os.listdir(root_path):
            if filename.endswith('eml'):
                full_path = os.path.join(desc_folder, filename)
                file_list.append(full_path)

        self.file_list = file_list

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        with open(email_file_path, 'rb') as f:
            email_content = BytesParser(policy=policy.default).parse(f)

        # sender IP
        sender_ip = extract_sender_ip(email_content._headers)
        # Assuming email_message is your email object
        sender_name, sender_address = parseaddr(email_content.get('From', ''))
        to_names, to_addresses = parseaddr(email_content.get('To', ''))

        subject = email_content['subject']

        # Check if the email message is multipart
        if email_content.is_multipart():
            # Iterate over each part of the email
            for part in email_content.walk():
                content_type = part.get_content_type()
                content_disposition = part.get('Content-Disposition')

                # If part is text/plain or text/html and not an attachment, process it
                if content_type in ('text/plain', 'text/html') and 'attachment' not in content_disposition:
                    soup = BeautifulSoup(decode_body(part.get_payload(decode=True)), 'html.parser')
                    for a_tag in soup.find_all('a'):
                        href = a_tag.get('href', '')
                        if len(a_tag.text) > 0:
                            a_tag.replace_with(a_tag.text + ' (' + href + ').')
                    text_content = ' '.join(soup.stripped_strings)
                    text_content = html.unescape(text_content)
        else:
            # For non-multipart emails, just process the payload
            soup = BeautifulSoup(decode_body(email_content.get_payload(decode=True)), 'html.parser')
            for a_tag in soup.find_all('a'):
                href = a_tag.get('href', '')
                if len(a_tag.text) > 0:
                    a_tag.replace_with(a_tag.text + ' (' + href + ').')
            text_content = ' '.join(soup.stripped_strings)
            text_content = html.unescape(text_content)

        # replace multiple newline characters with a single period
        text_content = re.sub(r'\n+', '. ', text_content)
        # replace multiple consecutive periods with a single period
        text_content = re.sub(r'\.{2,}', '', text_content)
        # replace multiple spaces with a single space
        text_content = re.sub(r'\s+', ' ', text_content)

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                subject, \
                text_content, \
               sender_ip


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

    # '''Read the pst file => Convert to individual eml'''
    # pst_file = pypff.open("./datasets/backup.pst")
    # root = pst_file.get_root_folder()
    # Process all folders and subfolders
    # pst_to_eml(root, desc_folder='./datasets/sjtu_phish')
    # pst_file.close()

    '''Load identity detection model'''
    checkpoint_path = './checkpoints/output_llama2_lora/checkpoint-19900'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path,
                                                     use_cache=False,
                                                     device_map="auto")
    model.eval()

    gen_config = GenerationConfig.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                  temperature=0.001,
                                                  max_new_tokens=250)

    ''''''
    desc_folder = './datasets/sjtu_phish'
    dataset = Feed(desc_folder)
    csv_file_path = './datasets/sjtu_phish_results.csv'

    # Check if we're writing to a new file, and write the header if so
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['email_file_path', 'sender_name', 'sender_address', 'to_names', 'to_addresses',
                             'subject', 'email_body_text', 'sender_ip', 'step_1_text', 'step_2_text',
                             'step_3_text', 'extra_response', 'generation_time'])

    for it in tqdm(range(len(dataset))):
        if dataset.file_list[it] in [x.split(',')[0] for x in open(csv_file_path).readlines()]:
            continue

        email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), \
            subject, email_body_text, sender_ip = dataset[it]

        parsed_email = f'Subject: {subject} \n From: (Name) {sender_name} \n Body: {email_body_text}'
        prompt = prepare_prompt(parsed_email, max_seq_len=gen_config.max_length)

        results = _generate(prompt, model, tokenizer, gen_config)
        total_time = results['total_time']
        results = results['generation']
        print(results.replace('\n', ' '))

        step_1_match = re.search(r"(Step 1:.*?)(?=Step 2)", results, re.DOTALL)
        step_1_text = step_1_match.group(1).strip() if step_1_match else None

        step_2_match = re.search(r"(Step 2:.*?)(?=Step 3|$)", results, re.DOTALL)
        step_2_text = step_2_match.group(1).strip() if step_2_match else None

        step_3_match = re.search(r"(Step 3:.*?)(?=\n)", results, re.DOTALL)
        step_3_text = step_3_match.group(1).strip() if step_3_match else None

        extra_match = re.search(r"Step 3:.*?\n(.*)", results, re.DOTALL)
        extra_response_text = extra_match.group(1).strip() if extra_match else None

        subject = subject.replace('\n', ' ') if subject else None
        email_body_text = email_body_text.replace('\n', '.')  # Preserving visual indication of newlines
        step_1_text = step_1_text.replace('\n', ' ') if step_1_text else None
        step_2_text = step_2_text.replace('\n', ' ') if step_2_text else None
        step_3_text = step_3_text.replace('\n', ' ') if step_3_text else None
        extra_response_text = extra_response_text.replace('\n', ' ') if extra_response_text else None

        # Append the new row to the CSV file
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([email_file_path, sender_name, sender_address,
                             to_names, to_addresses,
                             subject, email_body_text, sender_ip,
                             step_1_text, step_2_text, step_3_text, extra_response_text,
                             total_time])

    # df = pd.read_csv(csv_file_path)
    # print(df)