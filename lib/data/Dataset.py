from torch.utils.data import Dataset, DataLoader
import os, re
from email.utils import parseaddr, getaddresses
from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from email.header import decode_header
from email import message_from_string
from email.utils import parseaddr
from email.message import Message
from typing import Union, Optional, Tuple, List, Set
from tldextract import tldextract
import mailbox
from deep_translator import GoogleTranslator
import functools
import langdetect
from langdetect import detect_langs
import time

class EmailDataset(Dataset):
    _CallerPrefix = ".eml/.txt email files Loader"

    def __init__(self, root_path):
        file_list = []

        for root, dirs, files in os.walk(root_path):
            for filename in files:
                if filename.endswith('.eml') or filename.endswith('.txt'):
                    full_path = os.path.join(root, filename)  # Join root with filename
                    file_list.append(full_path)

        self.file_list = file_list

        proxy_set = os.getenv("http_proxy", None)
        if proxy_set is not None:
            self.translator = GoogleTranslator(source="auto", target="en",
                                               proxies={
                                                   "https": proxy_set,
                                                   "http": proxy_set
                                               })
        else:
            self.translator = GoogleTranslator(source="auto", target="en")


    def __len__(self):
        return len(self.file_list)

    def domain_parsing(self, address: Union[None, float, str, Set[str], List[str]]) -> Set[str]:

        url_pattern = re.compile(r'(https?://[^)]+)')
        parsed_domains = set()

        if isinstance(address, float) or address is None:
            return parsed_domains

        elif isinstance(address, str):
            match = url_pattern.search(address)
            if match: # is a URL
                domain = tldextract.extract(address).domain + '.' + tldextract.extract(address).suffix
                parsed_domains.add(domain)
            else: # is not a URL but an email address
                invalid_address = '@' not in address
                if not invalid_address:
                    domain = address.split('@')[-1]
                    domain = tldextract.extract(domain).domain + '.' + tldextract.extract(domain).suffix
                    parsed_domains.add(domain)

            return parsed_domains

        else:
            for add in address:
                ind_domain = self.domain_parsing(add)
                parsed_domains = parsed_domains.union(ind_domain)

        return parsed_domains

    @staticmethod
    def load_email_content(email_file_path: str) -> Message:
        with open(email_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            email_content = file.read()
        email_content = message_from_string(email_content)
        return email_content

    @staticmethod
    def decode_header(header_value) -> str:
        header_parts = decode_header(header_value)
        return ''.join(part if isinstance(part, str) else part.decode(charset or 'utf-8', 'replace') for part, charset in header_parts)

    @staticmethod
    def clean_text_content(text_content: str) -> str:
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
        return text_content

    @staticmethod
    def remove_prev_messages(text_content: str) -> str:
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

        return text_content

    @functools.lru_cache(maxsize=1000)
    def auto_translate(self, text):
        is_in_english = True
        try:
            detected_langs = detect_langs(text)
            for lang in detected_langs:
                if lang.lang != 'en':
                    is_in_english = False
                    break
        except langdetect.lang_detect_exception.LangDetectException:
            is_in_english = False

        max_retries = 3  # Number of times to retry the translation
        retry_delay = 2  # Seconds to wait before retrying

        if not is_in_english:
            for attempt in range(1, max_retries + 1):
                try:
                    # Deeptranslator has a character limit of 5000
                    return self.translator.translate(text[:min(5000, len(text))],
                                                     source='auto',
                                                     target='english')
                except Exception as e:
                    print(f"Attempt {attempt} - Error translating: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("Max retries reached. Returning original text.")
                        return text  # Return the original text if translation fails after retries
        return text

    def extract_subject(self, email_content: Message) -> str:
        subject = email_content.get('subject', '')
        if len(subject):
            try:
                subject = self.decode_header(subject)
            except UnicodeDecodeError: # unicode error
                subject = email_content.get('subject', '')

        return subject

    def extract_sender(self, email_content: Message) -> Tuple[Optional[str], Optional[str]]:
        sender_name, sender_address = parseaddr(email_content.get('From', ''))
        if sender_name:
            sender_name = self.decode_header(sender_name)
        return sender_name, sender_address

    def extract_recipients(self, email_content: Message) -> Tuple[List[str], List[str]]:
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
            if email_content.get('Delivered-To'):
                to_addresses = [email_content.get('Delivered-To')]

        return to_names, to_addresses

    def extract_reply_to_address(self, email_content: Message) -> Optional[str]:
        reply_to = email_content.get('Reply-To', '').strip()
        if reply_to:
            reply_to_addresses = getaddresses([reply_to])  # Handles potential list of addresses
            return reply_to_addresses[0][1]  # Return the first parsed email address
        return None

    def extract_text_content(self, part: Message) -> str:
        """
        Recursively extracts text content from an email part,
        including handling nested multiparts.
        """
        text_content = ""
        unwanted_extensions = {'.css', '.js', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.doc', '.docx', '.xls',
                               '.xlsx'}
        # Check if the part is a multipart
        if part.is_multipart():
            for subpart in part.get_payload():
                text_content += self.extract_text_content(subpart)
        else:
            # Process single part (leaf node)
            content_type = part.get_content_type()
            charset = part.get_content_charset('utf-8')
            raw_email_content = part.get_payload(decode=True)  # decoding is handled here

            # Handle decoded content directly if available
            if raw_email_content:
                try:
                    decoded_content = raw_email_content.decode(charset, 'replace')
                except LookupError:  # in case charset is not recognized
                    decoded_content = raw_email_content.decode('utf-8', 'replace')
            else:
                decoded_content = raw_email_content  # handle cases where payload is not encoded

            if 'html' in content_type:
                # soup = BeautifulSoup(decoded_content, 'html.parser')
                # text_content = ' '.join(soup.stripped_strings)
                soup = BeautifulSoup(decoded_content, 'html.parser')
                # Remove script and style elements
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()

                # Remove DOCTYPE and comments
                for element in soup.contents:
                    if isinstance(element, Comment) or isinstance(element, Doctype):
                        element.extract()

                text_parts = []
                for element in soup.descendants:
                    if isinstance(element, NavigableString):
                        # Add text directly, but strip to avoid excessive whitespace
                        stripped_text = element.strip()
                        if stripped_text:
                            text_parts.append(stripped_text)
                    elif element.name == 'a':
                        # Process <a> tags with href attributes
                        href = element.get('href', '')
                        # Check if the link ends with an unwanted extension
                        if not any(href.lower().endswith(ext) for ext in unwanted_extensions):
                            # Check for additional noisy patterns
                            if not any(noisy in href.lower() for noisy in ['mailto:', 'tel:', '#', 'javascript']):
                                link_text = f"{element.get_text()} ({href})"
                                text_parts.append(link_text)
                text_content = ' '.join(text_parts)  # Join all parts into a single string

            elif 'text' in content_type:
                text_content = decoded_content.strip()

        return str(text_content)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.load_email_content(email_file_path)

        headers = str(email_content._headers)

        sender_name, sender_address = self.extract_sender(email_content)
        sender_name = self.auto_translate(sender_name)

        to_names, to_addresses = self.extract_recipients(email_content)
        reply_to_address = self.extract_reply_to_address(email_content)
        if reply_to_address is None:
            reply_to_address = sender_address

        subject = self.extract_subject(email_content)
        subject = self.auto_translate(subject)

        text_content = self.extract_text_content(email_content)
        text_content = self.remove_prev_messages(text_content)
        text_content = self.clean_text_content(text_content)
        text_content = self.auto_translate(text_content)

        return email_file_path, \
               (sender_name, sender_address), \
                (to_names, to_addresses), \
                reply_to_address, \
                subject, \
                text_content, \
                headers


class EmailBoxDataset(EmailDataset):
    _CallerPrefix = ".mbox emailbox Loader"

    def __init__(self, root_path):

        messages = []
        file_list = []
        mbox = mailbox.mbox(root_path) # root_path must ends with .mbox
        for it, message in enumerate(mbox):
            file_list.append(root_path + '_' + str(it))
            messages.append(message)

        self.message_list = messages
        self.file_list = file_list

        proxy_set = os.getenv("http_proxy", None)
        if proxy_set is not None:
            self.translator = GoogleTranslator(source="auto", target="en",
                                               proxies={
                                                   "https": proxy_set,
                                                   "http": proxy_set
                                               })
        else:
            self.translator = GoogleTranslator(source="auto", target="en")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.message_list[idx]

        headers = str(email_content._headers)

        sender_name, sender_address = self.extract_sender(email_content)
        sender_name = self.auto_translate(sender_name)

        to_names, to_addresses = self.extract_recipients(email_content)
        reply_to_address = self.extract_reply_to_address(email_content)
        if reply_to_address is None:
            reply_to_address = sender_address

        subject = self.extract_subject(email_content)
        subject = self.auto_translate(subject)

        text_content = self.extract_text_content(email_content)
        text_content = self.remove_prev_messages(text_content)
        text_content = self.clean_text_content(text_content)
        text_content = self.auto_translate(text_content)

        return email_file_path, \
               (sender_name, sender_address), \
               (to_names, to_addresses), \
               reply_to_address, \
               subject, \
               text_content, \
               headers


if __name__ == '__main__':
    mailbox_dataset = EmailBoxDataset('./datasets/All mail Including Spam and Trash.mbox')
    item = mailbox_dataset[0]
    print()