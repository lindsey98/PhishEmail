import json

from torch.utils.data import Dataset
import os
from email import message_from_string
from email.utils import parseaddr
from collections import Counter
from scripts.utils import *
import pandas as pd
from tqdm import tqdm


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
                a_tag.replace_with(href + ' ' + a_tag.text)

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
            a_tag.replace_with(href + ' ' + a_tag.text)

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


class Nazario(Dataset):
    def __init__(self, root_path):
        file_list = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith('eml'):
                    continue
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)

        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = open(email_file_path, encoding="ISO-8859-1").read()

        # Parse the email
        email_message = message_from_string(email_content)

        # Assuming email_message is your email object
        sender_name, sender_address = parseaddr(email_message.get('From', ''))
        to_names, to_addresses = parseaddr(email_message.get('To', ''))

        # Get 'Subject' header and decode if needed
        subject = email_message['Subject']

        content_collection = []

        # Walk through each part of the email to find text or HTML body
        for part in email_message.walk():
            content_collection = process_email_parts(part, content_collection)

        # remove duplicates
        content_collection = sorted(set(content_collection), key=content_collection.index)
        email_body_text, email_images = parse_email_content(content_collection)

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                subject, \
                email_body_text, email_images

#
# class Enron(Dataset):
#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path)
#         self.file_list = list(df['file'])
#         self.message_list = list(df['message'])
#
#     def __len__(self):
#         return len(self.message_list)
#
#     def __getitem__(self, idx):
#         email_file_path = self.file_list[idx]
#         email_content = self.message_list[idx]
#         parsed_email = message_from_string(email_content)
#
#         # Extract header information
#         from_email_address = parsed_email.get("From")
#         subject = parsed_email.get("Subject")
#
#         # Extract body content
#         text_content = parsed_email.get_payload()
#         return email_file_path, (from_email_address, subject, text_content)

if __name__ == '__main__':
    dataset = Nazario(root_path = './datasets/Nazario_2005')
    print(len(dataset))
    print(dataset[2500])

    columns = ['Id', 'Parsed email', 'From: (Address)', 'To: (Name)', 'To: (Address)', 'Images']

    # Create an empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)

    for it, item in tqdm(enumerate(dataset)):
        email_file_path, (sender_name, sender_address), (to_names, to_addresses), subject, email_body_text, email_images = item
        parsed_email = f'Subject: {subject} \n From: (Name) {sender_name} \n Body: {email_body_text}'

        entry = {'Id': it, 'Parsed email': parsed_email, 'From: (Address)': sender_address,
                 'To: (Name)': to_names, 'To: (Address)': to_addresses,
                 'Images': email_images}

        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv('./datasets/nazario-annot.csv', index=False)

    df = pd.read_csv('./datasets/nazario-annot.csv', header=0)

    # Select only the 'Parsed email' column
    texts = df['Parsed email'].tolist()

    # Convert to the required JSON format
    formatted_data = [{'text': text} for text in texts]

    # Save to a new JSON file
    with open('./datasets/nazario_data.json', 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            json.dump(entry, f)
            f.write('\n')

    # dataset = Enron(csv_path = './datasets/enron_mail_2015/emails.csv')
    # print(len(dataset))
    # print(dataset[500])