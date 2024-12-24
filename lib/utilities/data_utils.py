import random
import re
from langdetect import detect, DetectorFactory, detect_langs
DetectorFactory.seed = 42
import langdetect
from typing import *
from transformers import AutoTokenizer
from tldextract import tldextract
import difflib
import json
from unidecode import unidecode
import requests
from bs4 import BeautifulSoup
import os
import mailbox
from tqdm import tqdm
from email import message_from_string
from email.utils import parseaddr
from bs4 import BeautifulSoup, NavigableString, Comment, Doctype, ProcessingInstruction


def mbox_to_eml(mbox_path: str, output_dir: str):
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
        try:
            with open(eml_filename, 'w', encoding="utf-8") as eml_file:
                eml_file.write(msg_str)
        except UnicodeEncodeError:
            os.remove(eml_filename)


def pst_to_eml(pff_file_path: str, output_dir: str):
    import pypff

    def process_email(email, output_dir):
        # Create .eml filename based on subject or a unique identifier
        eml_filename = os.path.join(output_dir, f"{email.identifier}.eml")

        with open(eml_filename, 'w', encoding='utf-8') as eml_file:
            # # Write headers
            if email.transport_headers:
                eml_file.write(email.transport_headers)
                eml_file.write("\n")
            else:
                headers = ''
                sender_name = email.get_sender_name() if hasattr(email, 'get_sender_name') else "Unknown"
                sender_attributes = parseaddr(message_from_string(email.transport_headers).get("From", ""))
                if len(sender_attributes):
                    sender_email = sender_attributes[1]
                else:
                    sender_email = ""
                headers += f"From: {sender_name} <{sender_email}>\n"
                headers += f"Date: {email.delivery_time}\n" if hasattr(email, 'delivery_time') else 'Date: \n'
                headers += f"Subject: {email.subject}\n" if hasattr(email, 'subject') else 'Subject: \n'

                # Determine content type and charset
                content_type = 'text/plain'
                charset = 'utf-8'  # Assuming UTF-8 for simplicity; adjust as necessary

                # Add content type and charset to headers
                headers += f"Content-Type: {content_type}; charset={charset}\n"
                headers += f"Content-Transfer-Encoding: 8bit\n"  # or 'base64' for binary content
                headers += '\n'  # Ensure there's a blank line after headers
                eml_file.write(headers)

            # Write email body
            # Check if the email has an HTML body and decode appropriately
            if email.html_body:
                soup = BeautifulSoup(email.html_body, 'html.parser')
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
                email_body = '. '.join(text_parts)

            elif email.plain_text_body:
                try:
                    # Decode plain text body if it's binary
                    email_body = email.plain_text_body.decode('utf-8', errors='replace')
                except UnicodeDecodeError:
                    email_body = email.plain_text_body.decode('iso-8859-1', errors='replace')  # Or 'windows-1252', 'shift_jis', etc.
                except AttributeError:
                    # Use the plain text body as is if it's already a string
                    email_body = email.plain_text_body
            else:
                # Default to an empty body if none exists
                email_body = ''

            eml_file.write(email_body)

    def extract_emails_and_folders(folder, output_dir):
        # Iterate through items in the folder
        for item in tqdm(folder.sub_items, desc="Exporting pst to eml files"):
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




class DomainUtils:
    def __init__(self):
        self.regional_tlds = self.get_regional_tlds_from_iana()

    def get_regional_tlds_from_iana(self):
        """
        Scrapes IANA's Root Zone Database to identify regional TLDs.

        Returns:
            set: A set of regional TLDs.
        """
        url = "https://www.iana.org/domains/root/db"
        response = requests.get(url, proxies={
            "http": os.environ.get("http_proxy", None),
            "https": os.environ.get("https_proxy", None)
        })
        regional_tlds = set()

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'tld-table'})
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        tld = cols[0].text.strip('.').lower()
                        tld_type = cols[1].text.strip().lower()
                        # Define criteria for regional TLDs based on type or sponsoring organization
                        if tld_type in {'geographic', 'internationalized', 'sponsored'}:
                            regional_tlds.add(tld)
        return regional_tlds

    def is_regional_tld(self, suffix: str) -> bool:
        return '.' in suffix

    @staticmethod
    def extract_domain_set(official_emails: List[str]) -> Set[str]:
        return set([tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix for x in official_emails])

    @staticmethod
    def extract_domain_components(domain: str) -> tuple:
        extracted = tldextract.extract(domain)
        return (extracted.domain, extracted.suffix)

    @staticmethod
    def contains_domain(text: str) -> Tuple[bool, Optional[Set[str]]]:

        # Search for potential domain patterns in the given text
        matches = re.findall(r'\b(?:[a-zA-Z0-9-]+\s?\.\s?)+[a-zA-Z]{2,}\b', text)

        # Initialize tldextract with the latest TLDs list
        tld_extract = tldextract.TLDExtract(cache_dir=True,
                                            suffix_list_urls=tldextract.PUBLIC_SUFFIX_LIST_URLS)

        # Clean up the matches by removing any spaces and checking if the TLD is valid
        cleaned_matches = set()
        for match in matches:
            # Remove any spaces within the domain
            cleaned_match = match.replace(' ', '')
            # Extract the subdomain, domain, and suffix (TLD)
            ext = tld_extract(cleaned_match)
            tld = ext.suffix

            # Check if the TLD is valid (non-empty and recognized)
            if tld:
                # Reconstruct the full domain (excluding subdomains if needed)
                full_domain = '.'.join(part for part in [ext.domain, tld] if part)
                cleaned_matches.add(full_domain)

        # Return True if a domain name with a valid TLD is found, otherwise False
        return bool(cleaned_matches), cleaned_matches if cleaned_matches else None

    @staticmethod
    def domain_set_overlap(query_domain_set: Set[str], reference_domain_set: Set[str]) -> bool:
        normalized_query_set = {domain.lower() for domain in query_domain_set}
        normalized_ref_set = {domain.lower() for domain in reference_domain_set}

        extracted_set1 = {DomainUtils.extract_domain_components(domain) for domain in normalized_query_set}
        extracted_set2 = {DomainUtils.extract_domain_components(domain) for domain in normalized_ref_set}

        for domain1, suffix1 in extracted_set1:
            for domain2, suffix2 in extracted_set2:
                # if DomainUtils.is_regional_tld(suffix1): # fixme: I only check the domain for now
                #     if domain1 == domain2:
                #         return True
                # else:
                #     if domain1 == domain2 and suffix1 == suffix2:
                #         return True
                if (not len(suffix2)) or (not len(suffix1)):
                    continue
                elif domain1 == domain2:
                    return True
        # If no overlap found
        return False

def find_all_token_indices(all_tokens: List[str], entity_tokens: List[str]):
    entity_len = len(entity_tokens)
    indices = []

    lower_all_tokens = [token.lower() for token in all_tokens]
    lower_entity_tokens = [token.lower() for token in entity_tokens]

    for i in range(len(lower_all_tokens) - entity_len + 1):
        if lower_all_tokens[i:i + entity_len] == lower_entity_tokens:
            indices.append((i, i + entity_len))

    return indices

def find_first_last_indices(all_tokens: List[str], entity_tokens: List[str]):
    entity_len = len(entity_tokens)
    indices = []

    # Track the first and last occurrence indices
    first_index = None
    last_index = None

    for i in range(len(all_tokens) - entity_len + 1):
        if [token.lower() for token in all_tokens[i:i + entity_len]] == entity_tokens:
            if first_index is None:
                first_index = (i, i + entity_len)
            last_index = (i, i + entity_len)

    # Check if the last occurrence is within the signatures
    if last_index and last_index[0] >= len(all_tokens) - 100:
        indices = [first_index, last_index] if first_index != last_index else [first_index]
    elif first_index:
        indices = [first_index]

    return indices

def tokenize_and_map(text: str, annotations: List[Dict], tokenizer: AutoTokenizer, label_to_id: Dict):
    # Tokenize text using a custom tokenizer function
    tokens = tokenizer.tokenize(text, truncation=False, add_special_tokens=False)

    # Initialize tags with "O"
    tags = [label_to_id["O"]] * len(tokens)

    for annot in annotations:
        entity = annot['text']
        entity_cls = annot['labels'][0]
        entity_tokens = tokenizer.tokenize(entity, truncation=False, add_special_tokens=False)

        if entity_cls == "identity":
            all_indices = find_first_last_indices(tokens, entity_tokens)
        else:
            all_indices = find_all_token_indices(tokens, entity_tokens)
        for start_index, end_index in all_indices:
            # Check if the range already has a tag other than "O"
            if all(tag == label_to_id["O"] for tag in tags[start_index:end_index]) or entity_cls == "relation":
                # Label the found entity
                tags[start_index] = label_to_id["B-" + entity_cls]
                for i in range(start_index + 1, end_index):
                    tags[i] = label_to_id["I-" + entity_cls]

        if len(all_indices) == 0:
            continue
    return tokens, tags

def filter_duplicates(strings: Union[List[str], Set[str]], threshold: float=0.5) -> List[List[str]]:
    clusters = []
    for string in strings:
        # Remove "from:" or any case variation with optional spaces
        string = re.sub(r'(?i)from\s*:?', '', string)
        # Remove leading "##"
        string = re.sub(r'^\s*#+', '', string)
        string = string.strip()
        if len(string) <= 1:
            continue

        # Check similarity with already filtered strings
        similar_found = False
        for i, f in enumerate(clusters):
            if difflib.SequenceMatcher(None, string, f[0]).ratio() > threshold: # group into a cluster
                similar_found = True
                # Add the string to the existing cluster
                clusters[i].append(string)
                break
        if not similar_found:
            clusters.append([string])

    return clusters

def process_entries(data: List[Dict], tokenizer: AutoTokenizer, label_to_id: Dict):
    seen_texts = set()
    processed_data = []
    for entry in data:
        text = entry['text']
        if text in seen_texts:
            continue  # Skip duplicates
        seen_texts.add(text)

        annotations = entry.get('annotations', [])
        if len(annotations) == 0:
            continue

        entities = []
        for annot in annotations:
            entity_cls = annot['labels'][0]
            entity = annot['text']
            if entity_cls == 'identity':
                if entity.startswith('From:'):
                    annot['text'] = re.sub(r"From:\s*", "", entity)
                entities.append(annot['text'])

        entity_clusters = filter_duplicates(entities, threshold=0.3)
        if len(entity_clusters) > 2:
            print()

        tokens, tags = tokenize_and_map(text, annotations, tokenizer, label_to_id)

        processed_data.append({
            "id": entry['Id'],
            "tokens": tokens,
            "ner_tags": tags,
            "metadata": entry.get("Path", ""),  # Add metadata field
            "text": text
        })

    return processed_data


def repeat_char(text, idx:Optional[int]):
    """Add a repeating character."""
    if len(text) < 3:
        return text
    if idx is None or idx < 1 or idx >= len(text) - 2:
        idx = random.randint(1, len(text) - 2)
    char = text[idx]
    return text[:idx] + char + char + text[idx+1:]

def delete_char(text, idx:Optional[int]):
    """Delete a character."""
    if len(text) < 3:
        return text
    if idx is None or idx < 1 or idx >= len(text) - 2:
        idx = random.randint(1, len(text) - 2)
    return text[:idx] + text[idx+1:]

def switch_chars(text, idx:Optional[int]):
    """Switch two adjacent characters."""
    if len(text) < 3:
        return text
    if idx is None or idx < 1 or idx >= len(text) - 2:
        idx = random.randint(1, len(text) - 2)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

def replace_with_similar(text, idx:Optional[int]):
    """Replace a character with a similar-looking Latin/Unicode character."""
    similar_chars = {
        '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
        "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ',
        'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս',
        'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'
    }
    if len(text) < 3:
        return text
    if idx is None or idx < 1 or idx >= len(text) - 2:
        idx = random.randint(1, len(text) - 2)
    char = text[idx]
    return text[:idx] + similar_chars.get(char, char) + text[idx+1:]


def match_case(word, pattern):
    """Match the case of the pattern in the original word."""
    if pattern.islower():
        return word.lower()
    elif pattern.isupper():
        return word.upper()
    elif pattern[0].isupper():
        return word.capitalize()
    else:
        return word


def to_sentence_case(text):
    """
    Converts the given text into sentence case.
    Capitalizes the first word and lowers the rest, except for proper nouns or acronyms.
    """
    # Tokenize based on whitespace or specific parse tokens
    tokens = re.split(r'(\s+|\(|\)|,|\.|\;)', text)  # split but keep delimiters

    # Capitalize first VB (or other sentence-starting token)
    capitalized = False
    new_tokens = []
    for token in tokens:
        if not capitalized and token.isalpha():
            new_tokens.append(token.capitalize())  # capitalize first alphabetic word
            capitalized = True
        else:
            new_tokens.append(token.lower())  # lowercase the rest
    return ''.join(new_tokens)


def remove_trailing_digits(s):
    # Use regex to replace trailing _digits with an empty string
    return re.sub(r'_\d+$', '', s)



def clean_string(s: str) -> str:
    s = s.strip()
    # Define the pattern for unnecessary special characters
    special_chars_pattern = r'[()\[\]:\*\^]'
    # Remove the special characters from the string
    s = re.sub(special_chars_pattern, '', s)
    # normalization: todo
    # mеtамаsk
    return s

def remove_urls(text):
    pattern = r'\((https?://[^)]+)\)'
    return re.sub(pattern, '', text)

def normalization(text):
    transliterated = unidecode(text)
    return transliterated


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
