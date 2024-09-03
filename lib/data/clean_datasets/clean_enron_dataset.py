import os, sys, email,re
import nltk
nltk.set_proxy('http://127.0.0.1:7890')
nltk.download("stopwords")
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
import gensim
from gensim.utils import simple_preprocess
import matplotlib
from lib.data.utils import remove_extra_spaces
matplotlib.use('Agg')
import pandas as pd

stop_words = list(STOP_WORDS)
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'cc', 'email', 'bcc', 'subject']) # add some email stopwords
def extract_all_features(df, features):
    # Initialize a dictionary to hold the extracted data for each feature
    extracted_data = {feature: [] for feature in features}

    # Iterate through each row (message) only once
    for row in df['message']:
        # Parse the email message
        e = email.message_from_string(row)

        # Extract each feature and append to the corresponding list
        for feature in features:
            extracted_data[feature].append(e.get(feature))

    # Update the DataFrame with the extracted data for each feature
    for feature, data in extracted_data.items():
        df[feature] = data

def get_email_body(data):
    column = []
    col_len = []
    for msg in data:
        e = email.message_from_string(msg)
        column.append(e.get_payload())
        col_len.append(len(e.get_payload().split()))
    return column, col_len

def get_address(from_field, strict):
    regex = re.compile('[\w\.-]+@[\w\.-]+\.\w+')
    match = re.findall(regex, from_field)
    if match:
        return match[0]
    else:
        if strict:
            return None
        else:
            return from_field

def normalize_address(address):
    if not address:
        return None
    elif "</O=ENRON" in address:
        names = re.findall(r'([a-zA-Z]+)', address)
        names_concatenated = ''.join(names)
        normalized_address = f"{names_concatenated}@enron.com"
        return normalized_address.lower()
    elif '@' in address:
        return address.lower()
    else:
        return None

if __name__ == '__main__':
    '''data cleaning'''
    df = pd.read_csv('./datasets/enron_mail_2015/emails.csv')
    print(df.shape)
    eng_stopwords = set(stopwords.words('english'))
    features = ["Date", "Subject", "X-From", "X-To", "X-Folder"]
    extract_all_features(df, features)

    df['body'], df['body_token_len'] = get_email_body(df['message'])
    df.to_csv('./datasets/enron_mail_2015/emails_processed.csv')

    # Remove punctuation, dropna
    df = pd.read_csv('./datasets/enron_mail_2015/emails_processed.csv')
    df = (df[(~df['body'].str.contains('Forwarded')) &
             (~df['body'].str.contains('Original Message')) &
             (~df['Subject'].fillna('').str.contains('Re:|Fw:', case=False))])
    regex = "\n[Tt][Oo]:[^\n]*|\n[CC][cc]:[^\n]*|\n[Bb][Cc][Cc]:[^\n]*|\n[Ss]ubject:[^\n]*"
    df['body'] = df['body'].apply(remove_extra_spaces)
    df['body'] = df['body'].apply(lambda x: re.sub(regex, '', re.sub('[^\w \n]', '', x)))
    df.dropna(subset=['body', 'X-From', 'X-To'], how='any', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv('./datasets/enron_mail_2015/emails_processed.csv', index=False)
    print(df.describe())

    df = pd.read_csv('./datasets/enron_mail_2015/emails_processed.csv')
    df['From'] = df['X-From'].apply(get_address, strict=True).apply(normalize_address)
    df['To'] = df['X-To'].apply(get_address, strict=False)
    df.dropna(subset=['From'], how='any', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv('./datasets/enron_mail_2015/emails_processed_clean.csv', index=False)
    print(df)
    exit()
