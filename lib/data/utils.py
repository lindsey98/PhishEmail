import nltk
from nltk.corpus import stopwords
import email
from email import message_from_string
from email.utils import parseaddr
import re
import requests
from io import BytesIO, StringIO
import urllib.parse
from requests.exceptions import Timeout, ConnectionError
import base64
# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from tqdm import tqdm
import json
import socket
import re
import mailbox
import email
import os


def mbox_to_eml(mbox_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the MBOX file
    mbox = mailbox.mbox(mbox_path)

    # Loop through all messages in the MBOX file
    for i, message in tqdm(enumerate(mbox), desc="Exporting mbox to eml files"):
        # Extract the message as a string
        try:
            msg_str = message.as_string()
        except UnicodeEncodeError:
            continue

        # If no Message-ID, fall back to a fallback identifier (e.g., email index)
        safe_message_id = f"email_{i + 1}"

        # Create a filename for the .eml file using the Message-ID or fallback
        eml_filename = os.path.join(output_dir, f"{safe_message_id}.eml")

        # Save the message to an .eml file
        with open(eml_filename, 'w', encoding="utf-8") as eml_file:
            eml_file.write(msg_str)


def pst_to_eml(pff_file_path, output_dir):
    import pypff

    def process_email(email, output_dir):
        # Create .eml filename based on subject or a unique identifier
        eml_filename = os.path.join(output_dir, f"{email.identifier}.eml")

        with open(eml_filename, 'w', encoding='utf-8') as eml_file:
            # Write headers
            eml_file.write(f"{email.transport_headers}\n")
            eml_file.write("\n")  # Blank line to separate headers from body

            # Write email body
            try:
                email_body = email.html_body.decode('utf-8')
            except:
                email_body = str(email.html_body)
            eml_file.write(email_body)

    def extract_emails_and_folders(folder, output_dir):
        # Iterate through items in the folder
        for item in folder.sub_items:
            if isinstance(item, pypff.folder):
                # Recursively process subfolder
                subfolder_output_dir = os.path.join(output_dir, item.name)
                os.makedirs(subfolder_output_dir, exist_ok=True)
                extract_emails_and_folders(item, subfolder_output_dir)
            elif isinstance(item, pypff.message):
                # Process email
                process_email(item, output_dir)

    # Open the pst file
    pff_file = pypff.file()
    pff_file.open(pff_file_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process root folder
    root_folder = pff_file.get_root_folder()
    extract_emails_and_folders(root_folder, output_dir)

    pff_file.close()


def remove_extra_spaces(text):
    # remove text surrounded by <>, since they are likely be comments that are invisible
    text_content = re.sub(r'<[^>]*>', '', text)
    # replace multiple newline characters with a single \n
    text_content = re.sub(r'\n+', '\n', text_content)
    # replace multiple consecutive periods with a single period
    text_content = re.sub(r'\.{2,}', '', text_content)
    # replace multiple spaces with a single space
    text_content = re.sub(r'\s+', ' ', text_content)
    return text_content

def remove_specific_special_chars(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)

    # Find all URLs in the text
    urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)

    # Replace URLs with placeholders
    for i, url in enumerate(urls):
        placeholder = f"URLPLACEHOLDER{i}"
        text = text.replace(url, placeholder)

    # Remove specific special characters
    text = re.sub('[$%^*+#=\[\]\\\\{}~]', '', text)

    # Replace placeholders with original URLs
    for i, url in enumerate(urls):
        placeholder = f"URLPLACEHOLDER{i}"
        text = text.replace(placeholder, url)

    return text


def process_email_parts(email_part, email_content_collection):
    html_tags = ['<table', '<tr', '<td', '<font', '<a', '<html', '<body', '<meta']

    content_type = email_part.get_content_type()

    # If it's plain text, check for HTML tags
    if content_type == "text/plain":
        payload = email_part.get_payload(decode=True).decode('ISO-8859-1')

        if any(tag in payload.lower() for tag in html_tags): # html inside the plain text
            content_type = "text/html"
            soup = BeautifulSoup(payload, 'html.parser')
            # Process <a> tags separately
            for a_tag in soup.find_all('a'):
                href = a_tag.get('href', '')
                if len(a_tag.text) > 0:
                    a_tag.replace_with(a_tag.text + ' (' + href + ').')

            # Extract text for the rest of the content
            text_content = ' '.join(soup.stripped_strings)
        else:
            text_content = payload

        text_content = remove_specific_special_chars(text_content)
        return email_content_collection + [(content_type, payload, text_content)]


    # If it's HTML, extract text content
    elif content_type.startswith("text/html"):
        payload = email_part.get_payload(decode=True).decode('ISO-8859-1')

        soup = BeautifulSoup(payload, 'html.parser')
        # Process <a> tags separately
        for a_tag in soup.find_all('a'):
            href = a_tag.get('href', '')
            if len(a_tag.text) > 0:
                a_tag.replace_with(a_tag.text + ' (' + href + ').')

        # Extract text for the rest of the content
        text_content = ' '.join(soup.stripped_strings)
        text_content = remove_specific_special_chars(text_content)

        return email_content_collection + [(content_type, payload, text_content)]

    elif content_type.startswith('image'):
        payload = email_part.get_payload(decode=True)
        return email_content_collection + [(content_type, payload, payload)]

    # Nested content
    elif content_type.startswith('multipart') or content_type.startswith('message/rfc'):
        try:
            subpart = email_part.get_payload(0)
        except TypeError: # sometimes the email_part._payload is a str, not a list, just return the payload directly
            text_content = email_part._payload
            text_content = remove_specific_special_chars(text_content)
            return email_content_collection + [(content_type, text_content, text_content)]

        return email_content_collection + process_email_parts(subpart, email_content_collection)

    return email_content_collection

def parse_email_content(email_content_collection):
    email_body_text = ''
    email_images = []
    for content in email_content_collection:
        content_type, original, text = content
        if 'image' not in content_type:
            email_body_text += text
        else:
            attached_images = BytesIO(original)
            email_images.append(attached_images)

    return email_body_text, email_images

def remove_wildcards(text):
    # Implement any specific sanitization rules here
    sanitized_text = re.sub(r'[^\x20-\x7E]', ' ', text)  # Removes non-printable characters
    # Replace homoglyphs
    homoglyphs_map = {
        'а': 'a',  # Cyrillic 'a' to Latin
        'е': 'e',  # Cyrillic 'e' to Latin
        'о': 'o',  # Cyrillic 'o' to Latin
        'р': 'p',  # Cyrillic 'p' to Latin
        'с': 'c',  # Cyrillic 'c' to Latin
        'у': 'y',  # Cyrillic 'y' to Latin
        'і': 'i',  # Cyrillic 'i' to Latin
        'ѕ': 's',  # Cyrillic 's' to Latin
        'і': 'i',  # Cyrillic 'i' to Latin
        'ј': 'j',  # Cyrillic 'j' to Latin
        'һ': 'h',  # Cyrillic 'h' to Latin
        'ӏ': 'l',  # Cyrillic 'l' to Latin
        'ԁ': 'd',  # Cyrillic 'd' to Latin
        'Ԍ': 'g',  # Cyrillic 'g' to Latin
        'ԏ': 'f',  # Cyrillic 'f' to Latin
        'Ԛ': 'q',  # Cyrillic 'q' to Latin
        'ԝ': 'w',  # Cyrillic 'w' to Latin
        'ԡ': 'b',  # Cyrillic 'b' to Latin
        'Խ': 'x',  # Cyrillic 'x' to Latin
        'Կ': 'k',  # Cyrillic 'k' to Latin
        'ᴏ': 'o',  # Small caps 'o' to Latin
        'ᴘ': 'p',  # Small caps 'p' to Latin
        'ᴜ': 'u',  # Small caps 'u' to Latin
        'ᴠ': 'v',  # Small caps 'v' to Latin
        'ᴡ': 'w',  # Small caps 'w' to Latin
        'ᴍ': 'm',  # Small caps 'm' to Latin
        'ᴎ': 'n',  # Small caps 'n' to Latin
        'ᴙ': 'r',  # Small caps 'r' to Latin
        'ᴚ': 'r',  # Small caps 'r' to Latin
        'ᴛ': 't',  # Small caps 't' to Latin
        'ᴀ': 'a',  # Small caps 'a' to Latin
        'ᴄ': 'c',  # Small caps 'c' to Latin
        'ᴅ': 'd',  # Small caps 'd' to Latin
        'ᴇ': 'e',  # Small caps 'e' to Latin
        'ᴊ': 'j',  # Small caps 'j' to Latin
        'ᴋ': 'k',  # Small caps 'k' to Latin
        'ᴧ': 'l',  # Small caps 'l' to Latin
        'ᴦ': 'g',  # Small caps 'g' to Latin
        'ᴨ': 'n',  # Small caps 'n' to Latin
        'ᴩ': 'p',  # Small caps 'p' to Latin
        'ᴪ': 'q',  # Small caps 'q' to Latin
        'ᴫ': 'r',  # Small caps 'r' to Latin
        'ᴬ': 'a',  # Small caps 'a' to Latin
        'ᴭ': 'ae',  # Small caps 'ae' to Latin
        'ᴮ': 'b',  # Small caps 'b' to Latin
        'ᴯ': 'o',  # Small caps 'o' to Latin
        'ᴰ': 'd',  # Small caps 'd' to Latin
        'ᴱ': 'e',  # Small caps 'e' to Latin
        'ᴲ': 'e',  # Small caps 'e' to Latin
        'ᴳ': 'g',  # Small caps 'g' to Latin
        'ᴴ': 'h',  # Small caps 'h' to Latin
        'ᴵ': 'i',  # Small caps 'i' to Latin
        'ᴶ': 'j',  # Small caps 'j' to Latin
        'ᴷ': 'k',  # Small caps 'k' to Latin
        'ᴸ': 'l',  # Small caps 'l' to Latin
        'ᴹ': 'm',  # Small caps 'm' to
    }
    sanitized_text = ''.join(homoglyphs_map.get(ch, ch) for ch in sanitized_text)

    return sanitized_text


def process_enron_dataset(csv_path):
    df = pd.read_csv(csv_path)
    message_list = list(df['message'])

    # Process each email and store the results in a list
    email_data = []

    for idx, email_content in tqdm(enumerate(message_list)):
        email_message = message_from_string(email_content)

        # Assuming email_message is your email object
        sender_name, sender_address = parseaddr(email_message.get('From', ''))
        to_names, to_addresses = parseaddr(email_message.get('To', ''))
        subject = email_message.get('Subject', '')

        content_collection = []

        # Walk through each part of the email to find text or HTML body
        for part in email_message.walk():
            content_collection = process_email_parts(part, content_collection)

        # remove duplicates
        content_collection = sorted(set(content_collection), key=content_collection.index)
        email_body_text, email_images = parse_email_content(content_collection)

        # Process the email body text
        email_body_text = re.sub(r'\n+', '. ', email_body_text)
        email_body_text = re.sub(r'\.{2,}', '', email_body_text)
        email_body_text = re.sub(r'\s+', ' ', email_body_text)
        email_body_text = remove_wildcards(email_body_text)  # Assuming sanitize_text is defined

        parsed_email = f'Subject: {subject} \n From: (Name) {sender_name} \n Body: {email_body_text}'

        email_dict = {
            'Id': idx,
            'text': parsed_email,
            'From: (Address)': sender_address,
            'To: (Name)': to_names,
            'To: (Address)': to_addresses
        }
        email_data.append(email_dict)

        # Add the processed text as a new column to the DataFrame
        email_df = pd.DataFrame(email_data)
        email_df.to_csv('./datasets/enron_mail_2015/emails_processed_clean_processed.csv', index=False)


def parse_json(text, delimiter='{"text":'):
    """
    Finds and extracts JSON objects from a string based on a given delimiter.
    """
    json_objects = []
    start = 0
    while True:
        start = text.find(delimiter, start)
        if start == -1:
            break
        end = text.find(delimiter, start + 1)
        json_str = text[start:end].strip()
        if end == -1:
            json_str = text[start:]
        try:
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        start += len(delimiter)
    return json_objects


def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data








