from torch.utils.data import Dataset, DataLoader
from lib.data.utils import *
import os, sys, email, re
from email.utils import parseaddr, getaddresses
import quopri
import base64
from email.header import decode_header
from email import message_from_string
from email.utils import parseaddr

class EmailDataset(Dataset):
    def __init__(self, root_path):
        file_list = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.eml'):
                    full_path = os.path.join(root, filename)  # Join root with filename
                    file_list.append(full_path)

        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        with open(email_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            email_content = file.read()
        email_content = message_from_string(email_content)

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

        # If 'To' addresses are missing, use 'Delivered-To' as a fallback
        if len(to_addresses) == 0:
            to_names = ', '.join(to_names)
            to_addresses = email_content.get('Delivered-To', '')
        else:
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
        # Deal with invisible hyphen
        text_content = re.sub(r'\xad', '', text_content)
        text_content = re.sub(r'&shy;', '', text_content)
        # Deal with ZWSP, ESC etc
        text_content = re.sub(r'\u200B', '', text_content)
        text_content = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text_content)
        text_content = re.sub(re.compile(r'[\u0090\u200C\u009F\u008F\uFEFF]'), '', text_content)
        # Removes non-printable characters
        text_content = re.sub(r'[^\x20-\x7E]', ' ', text_content)

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                subject, \
                text_content, \
                headers


