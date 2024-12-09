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
import quopri
import bs4
from lib.utilities import Logger

class EmailDataset(Dataset):
    _CallerPrefix = "Dataset Loader"

    def __init__(self, root_path, translate_on=False):
        self.file_list = []

        if os.path.isdir(root_path):  # Check if root_path is a directory
            for root, dirs, files in os.walk(root_path):
                for filename in files:
                    if filename.endswith('.eml') or filename.endswith('.txt'):
                        full_path = os.path.join(root, filename)  # Join root with filename
                        self.file_list.append(full_path)
        elif os.path.isfile(root_path):  # Check if root_path is a file
            if root_path.endswith('.eml') or root_path.endswith('.txt'):
                self.file_list.append(root_path)  # Add the file directly if it ends with .eml or .txt

        proxy_set = os.getenv("http_proxy", None)
        if proxy_set is not None:
            self.translator = GoogleTranslator(source="auto", target="en",
                                               proxies={
                                                   "https": proxy_set,
                                                   "http": proxy_set
                                               })
        else:
            self.translator = GoogleTranslator(source="auto", target="en")

        self.translate_on = translate_on


    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def domain_parsing(address: Union[None, float, str, Set[str], List[str]]) -> Set[str]:
        '''
        parse domain/url/email address into domain.tld format
        :param address:
        :return:
        '''
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
                ind_domain = EmailDataset.domain_parsing(add)
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
        try:
            header_parts_decoded = ''.join(part if isinstance(part, str) else part.decode(charset or 'utf-8', 'replace') for part, charset in header_parts)
        except LookupError:
            header_parts_decoded = ''.join(str(part) for part, charset in header_parts)

        return header_parts_decoded

    @staticmethod
    def clean_text_content(text_content: str) -> str:
        '''
        remove wildcards
        :param text_content:
        :return:
        '''
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

        zero_width_pattern = r'[\u200B\u200C\u200D\u2060\u00AD\uFEFF\u2061\u2062\u2063\u2064\u034F\u17B4\u17B5]'
        text_content = re.sub(zero_width_pattern, '', text_content)
        return text_content

    @staticmethod
    def decode_quoted_printable(encoded_str: bytes, charset: str = 'utf-8') -> str:
        '''
        Decode a quoted-printable encoded string to its original form using the specified charset.
        :param encoded_str:
        :param charset:
        :return:
        '''
        try:
            # Decode the quoted-printable encoded string
            decoded_bytes = quopri.decodestring(encoded_str)
            # Decode the bytes to string using the specified charset
            decoded_str = decoded_bytes.decode(charset, 'replace')
        except LookupError:
            # Fallback to utf-8 if charset is not recognized
            decoded_str = decoded_bytes.decode('utf-8', 'replace')
        except Exception as e:
            Logger.spit(f"Failed to decode the email with exception {e}", debug=True,
                        caller_prefix=EmailDataset._CallerPrefix)
            decoded_str = encoded_str.decode(charset, 'replace')  # Fallback to raw decoding
        return decoded_str

    @staticmethod
    def remove_prev_messages(text_content: str) -> str:
        '''
        Ignore previous communications in an email
        :param text_content:
        :return:
        '''
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
        '''
        Google translate
        :param text:
        :return:
        '''
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
                    Logger.spit(f"Email translation fails with exception", debug=True, caller_prefix=EmailDataset._CallerPrefix)
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        Logger.spit("Max retries reached. Returning original text", debug=True,
                                    caller_prefix=EmailDataset._CallerPrefix)
                        return text  # Return the original text if translation fails after retries
        return text

    def extract_subject(self, email_content: Message) -> str:
        '''
        get subject
        :param email_content:
        :return:
        '''
        subject = email_content.get('subject', '')
        if len(subject):
            try:
                subject = self.decode_header(subject)
            except UnicodeDecodeError: # unicode error
                subject = email_content.get('subject', '')

        return subject

    def extract_sender(self, email_content: Message) -> Tuple[Optional[str], Optional[str]]:
        '''
        get sender name and sender address
        :param email_content:
        :return:
        '''
        sender_name, sender_address = parseaddr(email_content.get('From', ''))
        if sender_name:
            sender_name = self.decode_header(sender_name)
        return sender_name, sender_address

    def extract_recipients(self, email_content: Message) -> Tuple[List[str], List[str]]:
        '''
        get recipient(s) names and addresses
        :param email_content:
        :return:
        '''
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
        '''
        get reply-to (destination address if the recipient reply to this email, can be different from the sender address)
        :param email_content:
        :return:
        '''
        reply_to = email_content.get('Reply-To', '').strip()
        if reply_to:
            reply_to_addresses = getaddresses([reply_to])  # Handles potential list of addresses
            return reply_to_addresses[0][1]  # Return the first parsed email address
        return None

    def extract_text_content(self, part: Message) -> Tuple[str, str]:
        '''
        Recursively extracts text content from an email part with nested multiparts.
        :param part:
        :return:
        '''
        text_content = ""
        html_content = ""
        unwanted_extensions = {'.css', '.js', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx'}
        unwanted_attachment = {'mailto:', 'tel:', '#', 'javascript', "mso["}
        url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')  # Simple URL pattern

        # Check if the part is a multipart
        if part.is_multipart():
            for subpart in part.get_payload():
                content = self.extract_text_content(subpart)
                text_content += content[0]
                html_content += content[1]
        else:
            if self.translate_on: ## fixme: When the email is in Chinese, need to turn this flag on, no decoding, direct translate
                raw_email_content = part.get_payload(decode=False)
                decoded_content = self.auto_translate(raw_email_content)
                content_type = "text"
            else:
                content_type = part.get_content_type()
                # if not content_type or "text" in content_type:
                #     decoded_content = part.get_payload(decode=False)  # decoding is handled here
                # else:
                charset = part.get_content_charset('utf-8')
                raw_email_content = part.get_payload(decode=True)  # decoding is handled here
                transfer_encoding = part.get('Content-Transfer-Encoding', '').lower()
                # Handle decoded content directly if available
                if raw_email_content:
                    # Decode content based on encoding specified
                    if transfer_encoding == 'quoted-printable':
                        decoded_content = self.decode_quoted_printable(raw_email_content, charset)
                    else:
                        try:
                            decoded_content = raw_email_content.decode(charset, 'replace')
                        except LookupError:  # in case charset is not recognized
                            decoded_content = raw_email_content.decode('utf-8', 'replace')
                else:
                    decoded_content = raw_email_content  # handle cases where payload is not encoded

            # if self.translate_on:  ## fixme
            #     try:
            #         decoded_content = self.auto_translate(decoded_content)
            #     except TypeError:
            #         decoded_content = self.auto_translate(decoded_content.decode('ISO-8859-1'))

            ### Extract text from HTML
            if 'html' in content_type:
                try:
                    soup = BeautifulSoup(decoded_content, 'html.parser')
                except bs4.builder.ParserRejectedMarkup:
                    soup = BeautifulSoup(decoded_content, 'lxml')
                except TypeError:
                    return "", ""

                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                # Remove all inline styles (style attributes)
                for tag in soup.find_all(True):  # `True` finds all tags
                    if 'style' in tag.attrs:
                        del tag['style']

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
                        text = element.get_text(strip=True)
                        is_text_url = url_pattern.match(text)

                        # fixme: how to handle the embedded urls?
                        if href and (not any(href.lower().endswith(ext) for ext in unwanted_extensions)) and \
                            (not any(noisy in href.lower() for noisy in unwanted_attachment)):
                            if is_text_url:
                                text_parts.append(href)
                            else:
                                text_parts.append(f'{text}')
                        else:
                            text_parts.append(text)

                text_content = '\n'.join(text_parts)  # Join all parts into a single string
                html_content = decoded_content

            else:  ### plain text
                if decoded_content:
                    text_content = decoded_content.strip()
                else:
                    text_content = ""

        return str(text_content), str(html_content)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.load_email_content(email_file_path)

        headers = str(email_content._headers)

        sender_name, sender_address = self.extract_sender(email_content)
        if self.translate_on:
            sender_name = self.auto_translate(sender_name)
            if sender_address and "@" not in sender_address:
                translatedaddress = self.auto_translate(sender_address)
                if translatedaddress is not None:
                    sender_name += translatedaddress

        to_names, to_addresses = self.extract_recipients(email_content)
        reply_to_address = self.extract_reply_to_address(email_content)
        if reply_to_address is None:
            reply_to_address = sender_address

        subject = self.extract_subject(email_content)
        if self.translate_on:
            subject = self.auto_translate(subject)

        text_content, html_content = self.extract_text_content(email_content)
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

