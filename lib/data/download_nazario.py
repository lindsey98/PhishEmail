import shutil

from lib.web_utils.CustomDriver import CustomWebDriver
import time
from selenium.webdriver.common.by import By
import pandas as pd
import json
import re
import bs4 as BeautifulSoup
import os
import requests
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def extract_emails(text):
    # Regex pattern to identify email sections
    # This assumes emails are separated by a specific separator or a new line
    email_pattern = re.compile(r"^From(?!:)[^\n]*\n(.*?)(?=^From(?!:)|\Z)", re.MULTILINE | re.DOTALL)

    # Find all matches in the text
    emails = re.findall(email_pattern, text)

    # Optionally, strip extra whitespace
    emails = [email.strip() for email in emails]

    return emails



if __name__ == '__main__':
    '''Nazario recent'''
    text = open('./datasets/phishing-2023.txt', encoding="ISO-8859-1").read()
    os.makedirs('./datasets/nazario-recent/2023', exist_ok=True)

    emails = extract_emails(text)
    for it, email in enumerate(emails):
        with open(os.path.join('./datasets/nazario-recent/2023', f"{it}.eml"), 'w', encoding='utf-8') as f: # must set this encoding
            f.write(email)

    '''CSDMC2010'''
    # labels = open('./datasets/CSDMC2010/SPAMTrain.label').readlines()
    # # 1 is ham
    # ct = 0
    # for l in labels:
    #     if l.split()[0] == '1':
    #         ct += 1
    #         os.makedirs('./datasets/CSDMC2010/Ham', exist_ok=True)
    #         shutil.copyfile(f'./datasets/CSDMC2010/TRAINING/{l.split()[1].strip()}',
    #                         f'./datasets/CSDMC2010/Ham/{l.split()[1].strip()}')
    # print(ct)