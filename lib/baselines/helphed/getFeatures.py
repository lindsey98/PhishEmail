import email
import email.policy
import email.message
from gensim.models import KeyedVectors
import ipaddress
import os
import re
import numpy as np
from pathlib import Path
import lxml.html
import nltk
import spacy
import tldextract
from urllib.parse import urlparse
from nltk.corpus import stopwords
from . import html_util
import configparser

'''
Four main parts: Body/Part features, Text features, Header features and URL 
Body: needs some form of HTML parsing
Text: raw text parsing (parse text from html?)
Header: simple header parsing
URL: needs HTML parsing

Existing fields:
['nurls', 'encoding', 'nparts', 'hasHTML', 'attachments',
       'badwords', 'ipurls', 'diffhref', 'forms', 'scripts', 'ndots', 'nports',
       'nrecs', 'checkdomains', 'subject_badwords',
       'distinct_words', 'char_count', 'word_count', 'richness', 'RE_presence',
       'link_images', 'named_urls']
need to check: script encoding
'''
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/helphed/conf.cfg')
MODEL_DIR_PATH = os.path.abspath(cfg.get('env', 'model_dir_path'))
word2vec_kv = KeyedVectors.load(f"{MODEL_DIR_PATH}/word2vec_model.kv", mmap='r')

class Result():
    def __init__(self, label):
        # removed script_parts because not in paper
        self.results_dict = dict.fromkeys(['label', 'nurls', 'encoding', 'nparts', 'hasHTML', 'attachments',
                                           'badwords', 'ipurls', 'diffhref', 'forms', 'scripts', 'ndots', 'nports',
                                           'nrecs', 'checkdomains', 'subject_badwords',
                                           'distinct_words', 'char_count', 'word_count', 'richness', 'RE_presence',
                                           'link_images', 'named_urls', 'Word2vec'], False)
        self.results_dict['subject_badwords'] = 0
        self.results_dict['badwords'] = 0
        self.results_dict['ndots'] = 0
        self.results_dict['attachments'] = 0
        self.results_dict['nrecs'] = 0
        self.results_dict['nurls'] = 0
        self.results_dict['diffhref'] = 0
        self.results_dict['label'] = label

    def __setitem__(self, key, value):
        self.results_dict[key] = value

    def __getitem__(self, key):
        return self.results_dict[key]

    def get_dict(self):
        # do check to see if any none?
        return self.results_dict


def exceed_length(some):
    return False


def count_email_parts(msg):
    if msg.is_multipart():
        parts = 0
        for _ in msg.iter_parts():
            parts += 1
        return parts
    else:
        return 1


def count_recepients(msg):
    recipients = []
    for header in ['To', 'Cc', 'Bcc']:
        if msg[header]:
            recipients += email.utils.getaddresses(msg.get_all(header))
    return len(recipients)


def count_subject_badwords(text: str):
    text = text.lower()
    bad_words = ['verify', 'bank', 'debit', 'payment', 'suspend']
    return sum([text.count(word) for word in bad_words])


def count_body_badwords(text: str):
    # body text should be lowered already
    bad_words = ['link', 'click', 'confirm', 'user', 'customer', 'client', 'suspend', 'restrict', 'verify', 'payment',
                 'protect']
    return sum([text.count(word) for word in bad_words])


def check_ip_address(url):
    parsed_url = parse_url(url)
    hostname = parsed_url.hostname
    try:
        # If the hostname can be converted to an IP address, it's valid
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def parse_url(url: str):
    if not urlparse(url).scheme:
        url = 'http://' + url

    # Parse the URL
    parsed_url = urlparse(url)

    return parsed_url


def get_domain_without_subdomain(url: str):
    extracted = tldextract.extract(url)
    if extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    else:
        return extracted.domain


def process_links(html, result: Result, sender_domain: str):
    links = html.xpath('//a')
    result['nurls'] = len(links)

    for link in links:
        url = link.get('href')
        try:
            a = parse_url(url).port
        except Exception as e:
            continue
        if url:
            result['ndots'] = max(result['ndots'], url.count('.'))
            result['nports'] = True if parse_url(url).port else result['nports']
            result['ipurls'] = True if check_ip_address(url) else result['ipurls']
            result['checkdomains'] = True if tldextract.extract(url).suffix and sender_domain != get_domain_without_subdomain(url).lower() else result['checkdomains']

            if link.text:
                if len(link.text.strip()) > 0:
                    result['named_urls'] += 1
                link_text = link.text.replace('\\n', '').replace('\\t', '')

                # for diffhref, make sure the domain exists (has at least a domain)
                if tldextract.extract(link_text).suffix and get_domain_without_subdomain(url).lower() != get_domain_without_subdomain(link_text).lower():
                    result['diffhref'] += 1

        img_child = link.xpath('.//img')
        if img_child:
            result['link_images'] = True


def get_text_from_html(html):
    html_util.clean_script_style_comments(html)
    text = html.xpath('//text()')
    return ' '.join(text)


def preprocess_text(text: str):
    text = text.replace('\\n', ' ').replace('\t', ' ').replace("\\'", "'").replace(",", " ")
    e = text.lower().split()
    w_tolemma = []
    w_normal = []
    for w in e:
        w = w.rstrip('.')
        # Match stop words and special characters
        if re.match(r'\W', w) or "b'" in w:
            continue

        if w in stop_words:
            w_normal.append(w)
            continue

        # Replace link with hyperlinks
        if re.match(r'http[s]?://', w):
            w = "LINKLINK"

        w_normal.append(w)
        w_tolemma.append(w)
    try:
        w_doc = nlp(' '.join(w_tolemma))
        w_lemmatized = [token.lemma_ for token in w_doc]
    except Exception as e:
        w_lemmatized = w_normal

    return w_normal, w_lemmatized


def parse_email_parts(filename, label):
    result = Result(label)
    try:
        with open(filename, 'r', encoding='UTF-8') as f:
            mail_parsed = email.message_from_file(f, _class=email.message.EmailMessage, policy=email.policy.default)
    except UnicodeDecodeError:
        with open(filename, 'r', encoding="ISO-8859-1") as f:
            mail_parsed = email.message_from_file(f, _class=email.message.EmailMessage, policy=email.policy.default)

    # Header-based features
    result['encoding'] = mail_parsed['Content-Transfer-Encoding']
    result['nparts'] = count_email_parts(mail_parsed)
    result['nrecs'] = count_recepients(mail_parsed)
    if mail_parsed['subject']:
        result['subject_badwords'] = count_subject_badwords(mail_parsed['subject'])
        result['RE_presence'] = 'RE' in mail_parsed['subject']

    # Content-based features
    html_text = []
    plain_text = []
    for part in mail_parsed.walk():
        if part['Content-Transfer-Encoding'] and not result['encoding']:
            result['encoding'] = mail_parsed['Content-Transfer-Encoding']
        if part.is_attachment():
            result['attachments'] = result['attachments'] + 1 if result['attachments'] else 1
        if 'script' in part.get_content_subtype():
            result['scripts'] = True
        if part.get_content_maintype() != 'text' and part.get_content_maintype() != 'multipart':
            continue

        payload = str(part.get_payload(decode=True))
        if part.get_content_type() == 'text/html':
            result['hasHTML'] = True
            html_text.append(payload)
        elif payload is not None:
            payload_lower = payload.lower()
            if '<html>' in payload_lower or '<body>' in payload_lower:
                result['hasHTML'] = True
                html_text.append(payload)
            else:
                plain_text.append(payload)

    # Content-based features: URL links
    final_text = ''
    try:
        sender_domain = get_domain_without_subdomain(mail_parsed['From'].split('@')[-1].replace('>', '')).lower()
    except Exception as e:
        sender_domain = ''
    for text in html_text:
        html_lxml = lxml.html.fromstring(text)
        process_links(html_lxml, result, sender_domain)
        result['scripts'] = True if html_lxml.xpath('//script') else result['scripts']
        result['forms'] = True if html_lxml.xpath('//form') else result['forms']
        final_text += get_text_from_html(html_lxml) + '\n\n'
    for text in plain_text:
        final_text += text + '\n\n'

    # Content-based features: Word2vec
    processed_text, lemmatized_text = preprocess_text(final_text)

    result['badwords'] = count_body_badwords(' '.join(lemmatized_text))
    result['word_count'] = len(processed_text)
    result['char_count'] = sum(len(s) for s in processed_text)
    if result['word_count'] == 0:
        result['richness'] = 0
    else:
        result['richness'] = result['char_count'] / result['word_count']
    result['distinct_words'] = len(list(set(processed_text)))

    lemmatized_words = [word for word in lemmatized_text if word in word2vec_kv.key_to_index]
    if not lemmatized_words:
        result['Word2vec'] = np.zeros(word2vec_kv.vector_size, dtype=np.float32)
    else:
        word_vectors = [word2vec_kv[word] for word in lemmatized_words]
        result['Word2vec'] = np.mean(word_vectors, axis=0)
    return result.get_dict()
