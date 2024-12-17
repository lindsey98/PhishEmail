import email
import features_hdr
import features_text
import features_html
import re
from bs4 import BeautifulSoup as bs
from .utils import *
import time
from transformers import BertTokenizer, BertModel
import torch

'''
This file extracts and store the features from each email, for input to the different modules 
'''

cfg = configparser.ConfigParser()
cfg.read('./lib/baselines/dfence/conf.cfg')

TEXT_EMBEDDING_MODEL = cfg.get('env', 'text_embedding_model')
NUM_OF_FEATURES = int(cfg.get('env', 'num_of_features'))
INCLUDE_CSS_FEATURES = bool(int(cfg.get('env', 'include_css_features')))
INCLUDE_JS_FEATURES = bool(int(cfg.get('env', 'include_js_features')))
INCLUDE_DOM_FEATURES = bool(int(cfg.get('env', 'include_dom_features')))
INCLUDE_LINK_FEATURES = bool(int(cfg.get('env', 'include_link_features')))
INCLUDE_CHARSET = bool(int(cfg.get('env', 'include_charset')))
INCLUDE_MESSAGEID = bool(int(cfg.get('env', 'include_messageid')))
INCLUDE_MAILADDR = bool(int(cfg.get('env', 'include_mailaddr')))
INCLUDE_SECTION_BOUNDARY = bool(int(cfg.get('env', 'include_section_boundary')))
BERT_GLOBAL_LOOKUP = bool(int(cfg.get('env', 'bert_global_lookup')))
DEBUG_BYPASS = int(int(cfg.get('env', 'debug_bypass')))


def processEmails(dataset, batch_size):
    """
    Entry point for feature extraction.
    This function extracts features for all four feature types in batches and returns the full DataFrames.
    """

    # Initialize lists to accumulate results from all batches
    all_header_features = []
    all_html_features = []
    all_text_features = []
    all_urls = []
    total_time = 0

    # Initialize feature usage dictionary
    dict_feature_use = {
        'css_features': INCLUDE_CSS_FEATURES,
        'js_features': INCLUDE_JS_FEATURES,
        'dom_features': INCLUDE_DOM_FEATURES,
        'link_features': INCLUDE_LINK_FEATURES,
        'charset': INCLUDE_CHARSET,
        'messageid': INCLUDE_MESSAGEID,
        'mailaddr': INCLUDE_MAILADDR,
        'section_boundary': INCLUDE_SECTION_BOUNDARY
    }

    header_colnames = features_hdr.get_column_list(dict_feature_use)
    html_colnames = features_html.get_column_list(dict_feature_use)

    embedding_model = TEXT_EMBEDDING_MODEL
    tokenizer = BertTokenizer.from_pretrained(embedding_model)
    model = BertModel.from_pretrained(embedding_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size  # Calculate number of batches

    for batch_num in range(num_batches):

        # Determine start and end indices for the current batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        # Initialize lists for the current batch
        batch_header_features = []
        batch_html_features = []
        batch_text_features = []
        batch_urls = []
        filename_list = []

        # Process emails in the current batch
        for it in range(start_idx, end_idx):
            email_file_path = dataset.file_list[it]

            try:
                with open(email_file_path, 'r', encoding='UTF-8') as fp:
                    mail_parsed = email.message_from_file(fp)
            except UnicodeDecodeError:
                with open(email_file_path, 'r', encoding='ISO-8859-1') as fp:
                    mail_parsed = email.message_from_file(fp)

            header_data = []
            html_data = []
            plain_text_data = []
            for part in mail_parsed.walk():
                if part.get_content_maintype() == 'application':
                    continue
                if part.get_content_maintype() == 'image':
                    continue

                if part.get_content_maintype() != 'text' and part.get_content_maintype() != 'multipart':
                    print(part.get_content_maintype())
                    print(str(part.get_payload(decode=True)))

                payload = str(part.get_payload(decode=True))
                if part.get_content_type() == 'text/html':
                    html_data.append(payload)
                elif part.get_content_type() == 'text/plain':
                    plain_text_data.append(payload)
                elif payload is not None:
                    plain_text_data.append(payload)

            header_data = mail_parsed
            plain_text_data = ' '.join(plain_text_data)
            html_data = ' '.join(html_data)

            start_time = time.time()
            filename_list.append(email_file_path)
            batch_header_features.append(processHeader(header_data, dict_feature_use))
            batch_html_features.append(processHTML(html_data))
            batch_text_features.append(processText(plain_text_data, html_data, tokenizer, model, NUM_OF_FEATURES))
            batch_urls.append(processURL(plain_text_data, html_data))
            total_time += time.time() - start_time

        # Create DataFrames for the current batch
        header_features_df = pd.DataFrame(batch_header_features, columns=header_colnames)
        header_features_df.fillna(-1, inplace=True)
        header_features_df['ID'] = filename_list

        html_features_df = pd.DataFrame(batch_html_features, columns=html_colnames)
        html_features_df.fillna(-1, inplace=True)
        html_features_df['ID'] = filename_list

        text_features_df = pd.DataFrame(batch_text_features)
        text_numfeat = list(np.arange(0, NUM_OF_FEATURES))
        text_colnames = ['features_plain' + str(string) for string in text_numfeat] + ['text_lang_plain'] + [
            'features_html' + str(string) for string in text_numfeat] + ['text_lang_html']
        text_features_df.columns = text_colnames
        text_features_df['ID'] = filename_list

        URLs_df = pd.DataFrame(batch_urls, columns=['url_list'])
        URLs_df.columns = ['url_list']
        URLs_df['ID'] = filename_list

        # Adjust column names
        header_col = header_features_df.columns
        header_features_df.columns = ['features_' + str(string) if string != 'ID' and string != 'label' else string for
                                  string in header_col]

        html_col = html_features_df.columns
        html_features_df.columns = ['features_' + str(string) if string != 'ID' and string != 'label' else string for string
                                in html_col]

        # Accumulate results from the current batch
        all_header_features.append(header_features_df)
        all_html_features.append(html_features_df)
        all_text_features.append(text_features_df)
        all_urls.append(URLs_df)

    # Concatenate all batches into full DataFrames
    full_header_features_df = pd.concat(all_header_features, ignore_index=True)
    full_html_features_df = pd.concat(all_html_features, ignore_index=True)
    full_text_features_df = pd.concat(all_text_features, ignore_index=True)
    full_URLs_df = pd.concat(all_urls, ignore_index=True)


    return full_header_features_df, full_html_features_df, full_text_features_df, full_URLs_df, total_time



def processHeader(header_data, dict_feature_use):
    """
    This function takes header section data and returns extracted header features for each email
    header data is a email message object (can use the keys to check for field)
    output format:
    """

    header_features = features_hdr.extract_features(header_data, dict_feature_use)
    return header_features


def processURL(plain_text_data, html_data):
    """
    This function takes in both plain text data string and html data string, and returns extracted URLs for each email
    output format:
    """
    text = str(plain_text_data) + '\n' + str(html_data)
    list_url = re.findall(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[\/\-?=_:;@.&+]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    URLs = list(set(list_url))
    return [URLs]


def processHTML(html_data):
    """
    This function takes in html body data string, and returns extracted html structural data for each email
    output format:
    """
    html_features = features_html.extract_features(html_data)
    return html_features


# def processText(plain_text_data, html_data, tokenizer, model, num_of_features,htmltext_dict, plaintext_dict):
def processText(plain_text_data, html_data, tokenizer, model, num_of_features):
    """
    This function takes in plain text body string and html body data string, and returns extracted Bert features and text language for each email. Note that the htmltext processing needs to be cleaned to remove other texts
    output format: [plainfeature0..plainfeature767,plaintext_textlang, htmlfeature0..htmlfeature767, htmltext_textlang]
    """

    # if plaintext_data_hash in dict_text_hash_plaintext:
    html_data = html_data.replace('\\n', '').replace('\\t', '')
    utils.g_cur_work_task = 'soup = bs(str(html_data)'
    soup = bs(str(html_data), 'lxml')
    html_text_data = soup.text.replace('\\n', '').replace('\\t', '')
    html_text_data = html_text_data[2:-1]
    # preprocessing
    plain_text_preprocessed = features_text.preprocessing_text(plain_text_data)
    utils.g_cur_work_task = 'BERT'
    plain_text_features = features_text.create_text_feat(plain_text_preprocessed, tokenizer, model, num_of_features,
                                                         words_range=100)
    html_text_features = features_text.create_text_feat(html_text_data, tokenizer, model, num_of_features,
                                                        words_range=100)

    text_features = plain_text_features + html_text_features

    return text_features
