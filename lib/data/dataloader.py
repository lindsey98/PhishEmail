from torch.utils.data import Dataset, DataLoader
from lib.data.utils import *
import os, sys, email, re
from email.utils import parseaddr, getaddresses
import quopri
import base64
from email.header import decode_header
from email import message_from_string
from email.utils import parseaddr
from email.message import Message
import binascii

class EmailDataset(Dataset):
    def __init__(self, root_path):
        file_list = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.eml'):
                    full_path = os.path.join(root, filename)  # Join root with filename
                    file_list.append(full_path)
                elif filename.endswith('.txt'):
                    full_path = os.path.join(root, filename)  # Join root with filename
                    file_list.append(full_path)

        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def extract_text_content(self, part: Message) -> str:
        """
        Recursively extracts text content from an email part,
        including handling nested multiparts.
        """
        text_content = ""

        # Check if the part is a multipart
        if part.is_multipart():
            # Recursively extract text from each sub-part
            for subpart in part.get_payload():
                text_content += self.extract_text_content(subpart)
        else:
            # Process single part (leaf node)
            content_type = part.get_content_type()
            content_transfer_encoding = part.get('Content-Transfer-Encoding', '').lower()
            raw_email_content = part.get_payload(decode=False)  # directly decode the payload

            # Decode content based on transfer encoding
            try:
                if content_transfer_encoding == 'base64':
                    # Add padding if necessary
                    missing_padding = len(raw_email_content) % 4
                    if missing_padding:
                        raw_email_content += '=' * (4 - missing_padding)
                    decoded_content = base64.b64decode(raw_email_content).decode('utf-8', 'ignore')
                elif content_transfer_encoding == 'quoted-printable':
                    decoded_content = quopri.decodestring(raw_email_content).decode('utf-8', 'ignore')
                else:
                    decoded_content = raw_email_content
            except (binascii.Error, ValueError) as e:
                print(f"Error decoding content: {e}")
                decoded_content = raw_email_content

            # Convert HTML content to plain text if necessary
            if 'html' in content_type:
                soup = BeautifulSoup(decoded_content, 'html.parser')
                text_content = ' '.join(soup.stripped_strings)
            elif 'text' in content_type:
                text_content = decoded_content.strip()

        return text_content

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
        text_content = self.extract_text_content(email_content)

        reply_pattern = re.compile(r"On.*wrote:")
        forward_pattern = re.compile(r"^-{2,}\s*Forwarded message\s*-+$", re.MULTILINE)

        # First, attempt to split by the forwarding pattern
        forward_parts = forward_pattern.split(text_content, 1)
        if len(forward_parts) > 1:
            # If there is text before the forwarding header, return it
            pre_forward_content = forward_parts[0].strip()
            if pre_forward_content:
                text_content = pre_forward_content

        # If no forwarding header or no content before it, check for reply pattern
        reply_parts = reply_pattern.split(text_content, 1)
        if len(reply_parts) > 1:
            # Return text before the reply pattern
            text_content = reply_parts[0].strip()

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


