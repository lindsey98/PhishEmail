from lib.web_utils.CustomDriver import CustomWebDriver
import time
from selenium.webdriver.common.by import By
import pandas as pd
import json
import re
import bs4 as BeautifulSoup
import os
import requests

def extract_emails(text):
    # Regex pattern to identify email sections
    # This assumes emails are separated by a specific separator or a new line
    email_pattern = re.compile(r"^From(?!:)[^\n]*\n(.*?)(?=^From(?!:)|\Z)", re.MULTILINE | re.DOTALL)

    # Find all matches in the text
    emails = re.findall(email_pattern, text)

    # Optionally, strip extra whitespace
    emails = [email.strip() for email in emails]

    return emails


def download_eml_files(base_url, folder_url):
    # Create a session object to persist certain parameters across requests
    session = requests.Session()
    # Get the page
    response = session.get(folder_url)
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links on the page
    for link in soup.find_all('a'):
        href = link.get('href')
        # Check if the link is to an .eml file
        if href and href.endswith('.eml'):
            download_url = base_url + href
            # Get the file name
            file_name = href.split('/')[-1]
            # Download the .eml file
            file_response = session.get(download_url)
            if file_response.status_code == 200:
                # Save the file
                with open(file_name, 'wb') as file:
                    file.write(file_response.content)
                print(f"Downloaded {file_name}")
            else:
                print(f"Failed to download {file_name}")

if __name__ == '__main__':
    '''Nazario recent'''
    # text = open('./datasets/phishing-2023.txt', encoding="ISO-8859-1").read()
    # os.makedirs('./datasets/nazario-recent/2023', exist_ok=True)
    #
    # emails = extract_emails(text)
    # for it, email in enumerate(emails):
    #     with open(os.path.join('./datasets/nazario-recent/2023', f"{it}.eml"), 'w', encoding='utf-8') as f: # must set this encoding
    #         f.write(email)

    '''phishing pot'''
    base_url = 'https://github.com'
    folder_url = 'https://raw.githubusercontent.com/rf-peixoto/phishing_pot/main/email'

    # Create a directory to save the downloaded files
    os.makedirs('./datasets/phishing_pot', exist_ok=True)
    os.chdir('./datasets/phishing_pot')

    # Call the function
    download_eml_files(base_url, folder_url)