import nltk
from nltk.corpus import stopwords
import email
from email import message_from_string
from email.utils import parseaddr
import re
from bs4 import BeautifulSoup
import requests
from io import BytesIO, StringIO
import urllib.parse
from requests.exceptions import Timeout, ConnectionError
import base64
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
from tqdm import tqdm
import json
from email.parser import BytesParser
from email.policy import default
import re
import socket
import re
from ipaddress import ip_address
from typing import Tuple

stop_words = list(STOP_WORDS)
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'cc', 'email', 'bcc', 'subject']) # add some email stopwords


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

def get_body_text(text):
    try:
        return text.split("Body: ")[1]
    except IndexError:
        return ""

'''Extract all images from email body'''
def get_img_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get image links from <img> tags
    img_links = [img['src'] for img in soup.find_all('img', src=True)]

    # Get image links from <picture> tags
    picture_links = [source['srcset'] for source in soup.find_all('source', srcset=True)]

    # Get image links from inline styles (background-image)
    inline_style_links = []
    for tag in soup.find_all(style=True):
        style = tag['style']
        match = re.search("url\(['\"]?(.*?)['\"]?\)", style)
        if match:
            inline_style_links.append(match.group(1))

    # Get image links from <object> tags
    object_links = [obj['data'] for obj in soup.find_all('object', data=True)]

    # Get image links from <embed> tags
    embed_links = [embed['src'] for embed in soup.find_all('embed', src=True)]

    # Get SVG images
    svg_links = [str(svg) for svg in soup.find_all('svg')]

    # Get Canvas elements
    canvas_links = [str(canvas) for canvas in soup.find_all('canvas')]

    # Get Data URIs
    data_uris = []
    for tag in soup.find_all(['img', 'source']):
        src = tag.get('src', '')
        if src.startswith('data:image'):
            data_uris.append(src)

    # Combine all links
    all_links = img_links + picture_links + inline_style_links + object_links + embed_links + svg_links + canvas_links + data_uris

    return all_links

'''Load the image links in a safe way'''
def load_images(img_links, proxies=None, max_retries=1):
    img_bytes_list = []
    for link in img_links:
        if link.startswith('data:image/'): # Handle Data URIs
            try:
                # Extract the base64 encoded part
                base64_data = link.split(',')[1]
                img_bytes = BytesIO(base64.b64decode(base64_data))
                img_bytes_list.append(img_bytes)
            except Exception as e:
                pass
        elif link.startswith('<svg'):
            # Handle SVGs
            try:
                svg_bytes = StringIO(link)  # SVG is XML text, so we use StringIO
                img_bytes_list.append(svg_bytes)
            except Exception as e:
                pass
        else:
            retries = 0
            while retries <= max_retries:
                try:
                    response = requests.get(link, proxies=proxies, timeout=5)
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if content_type.startswith('image/'):
                            img_bytes = BytesIO(response.content)
                            img_bytes_list.append(img_bytes)
                            break  # Successfully downloaded, exit the retry loop
                        else:
                            break  # Not an image, exit the retry loop
                    else:
                        break  # HTTP error, exit the retry loop
                except (Timeout, ConnectionError) as e:
                    retries += 1  # Increment the retry counter and try again
                except Exception as e:
                    break  # Unknown error, exit the retry loop

    return img_bytes_list

'''Extract all links from email body'''
def get_links(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    links = []

    # Standard <a> tags
    for tag in soup.find_all('a', href=True):
        text = tag.string if tag.string else tag.text
        url = tag['href']
        if urllib.parse.urlparse(url).scheme:
            links.append((text, url))

    # JavaScript links
    for tag in soup.find_all(onclick=True):
        onclick = tag['onclick']
        match = re.search(r"window\.location\.href='(.*?)'|window\.open\('(.*?)'\)", onclick)
        if match:
            url = match.group(1) or match.group(2)
            text = tag.string if tag.string else tag.text
            if urllib.parse.urlparse(url).scheme:
                links.append((text, url))

    # Form action links
    for tag in soup.find_all('form', action=True):
        url = tag['action']
        if urllib.parse.urlparse(url).scheme:
            text = 'Form: ' + (tag.get('name') or tag.get('id') or '')
            if urllib.parse.urlparse(url).scheme:
                links.append((text, url))

    # Area map links
    for tag in soup.find_all('area', href=True):
        url = tag['href']
        text = tag.get('alt', 'Area Map')
        if urllib.parse.urlparse(url).scheme:
            links.append((text, url))

    # SVG Links
    for tag in soup.find_all('a', {'xlink:href': True}):
        url = tag['xlink:href']
        text = tag.string if tag.string else tag.text
        if urllib.parse.urlparse(url).scheme:
            links.append((text, url))

    return links


def extract_data(feature, df):
    column= []
    for row in df:
        e = email.message_from_string(row)
        column.append(e.get(feature))
    return column

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

def sanitize_text(text):
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
        email_body_text = sanitize_text(email_body_text)  # Assuming sanitize_text is defined

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

def extract_sender_ip(email_headers: Tuple) -> str:
    # Regular expression pattern to extract IP addresses from "Received" lines
    ip_regex = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    ips = []
    for header in email_headers:
        if header[0].startswith('Received'):
            found_ips = ip_regex.findall(header[1])
            ips.extend(found_ips)

    # Iterate over the found IPs in reverse order (start from the original sender)
    for ip in reversed(ips):
        try:
            # Check if the IP address is a private address
            if not ip_address(ip).is_private:
                return ip
        except ValueError:
            # If IP address is invalid, continue to the next
            continue

    return "NotFound"

def reverse_dns(ip_address):
    try:
        return socket.gethostbyaddr(ip_address)[0]
    except (socket.herror, socket.gaierror):
        return "No domain associated with the IP address."
