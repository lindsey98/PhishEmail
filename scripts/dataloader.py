from torch.utils.data import Dataset
import os
import re
from email import message_from_string
from email.header import decode_header
from email.utils import parseaddr
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter

def question_template_brand(email_subject, email_body):
    question = f'''
        Given the email subject <start>{email_subject}<end>, and the email body <start>{email_body}<end>,
        Question: What is the brand? Answer: 
    '''
    return {
            "role": "user",
            "content": question
    }

def question_template_motivation(email_subject, email_body):
    question = f'''
        Given the email subject <start>{email_subject}<end>, and the email body <start>{email_body}<end>,
        Question: Select one option that is appropriate to describe the email from the following: 
        A. The email triggers a sense of urgency, fear, or greed.
        B. The email doesn’t trigger a sense of urgency, fear, or greed. Answer: 
    '''
    return {
        "role": "user",
        "content": question
    }

def question_template_action(email_subject, email_body):
    question = f'''
        Given the email subject <start>{email_subject}<end>, and the email body <start>{email_body}<end>,
        Question: Select one option that is appropriate to describe the email from the following: 
        A. The email requires certain subsequent actions to take.
        B. The email doesn’t require any subsequent action to take. Answer: 
    '''
    return {
        "role": "user",
        "content": question
    }


class Nazario(Dataset):
    def __init__(self, root_path):
        file_list = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith('eml'):
                    continue
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)

        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = open(email_file_path, encoding="ISO-8859-1").read()

        # Parse the email
        email_message = message_from_string(email_content)

        # Get 'From' header
        from_header = email_message['From'] # fixme: from header can be spoofed
        from_email_address = parseaddr(from_header)[1]

        # Get 'Subject' header and decode if needed
        subject = email_message['Subject']

        # Initialize an empty string to store HTML content
        text_content = email_message.get_payload()

        # Loop through each part in the email to find the HTML part
        for part in email_message.walk():
            if part.get_content_type() == "text/html":
                html_content = part.get_payload(decode=True).decode('ISO-8859-1')
                # Using BeautifulSoup to extract text from HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text()
                text_content = text_content.replace('\xa0', ' ')
                text_content = text_content.replace('\n', ' ')
                text_content = re.sub(' +', ' ', text_content)
                break  # No need to look further

        return email_file_path, (from_email_address, subject, text_content)


class CSDMC(Dataset):
    def __init__(self, root_path):
        label_file = [x.strip() for x in open(os.path.join(root_path, 'SPAMTrain.label')).readlines()]
        label_list = []
        file_list = []
        for line in label_file:
            label, file = line.split(' ')
            if file.lower().startswith('train'):
                file_list.append(os.path.join(root_path, 'TRAINING', file))
            else:
                file_list.append(os.path.join(root_path, 'TESTING', file))
            label_list.append(int(label))

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith('eml'):
                    continue
                full_path = os.path.join(dirpath, filename)
                if full_path not in file_list:
                    file_list.append(full_path)
                    label_list.append(-1) # unlabeled

        self.file_list = file_list
        self.labels = label_list

    @property
    def get_dist(self):
        return Counter(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        label = self.labels[idx]
        email_content = open(email_file_path, encoding="ISO-8859-1").read()

        # Parse the email
        email_message = message_from_string(email_content)

        # Get 'From' header
        from_header = email_message['From'] # fixme: from header can be spoofed
        from_email_address = parseaddr(from_header)[1]

        # Get 'Subject' header and decode if needed
        subject = email_message['Subject']

        text_content = email_message.get_payload()

        # Loop through each part in the email to find the HTML part
        for part in email_message.walk():
            if part.get_content_type() == "text/html":
                html_content = part.get_payload(decode=True).decode('ISO-8859-1')
                # Using BeautifulSoup to extract text from HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text()
                text_content = text_content.replace('\xa0', ' ')
                text_content = text_content.replace('\n', ' ')
                text_content = re.sub(' +', ' ', text_content)
                break  # No need to look further

        return email_file_path, (from_email_address, subject, text_content), label

class SpamAssassin(Dataset):
    def __init__(self, root_path):
        file_list = []
        label_list = []
        difficulty_list = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if 'tar.bz' in filename:
                    continue
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)
                if 'ham' in dirpath:
                    label_list.append(1)
                    if 'easy' in dirpath:
                        difficulty_list.append('easy')
                    else:
                        difficulty_list.append('hard')
                else:
                    label_list.append(0)
                    difficulty_list.append('')

        self.file_list = file_list
        self.labels = label_list
        self.difficulty_list = difficulty_list

    @property
    def get_dist(self):
        return Counter(self.labels), Counter(self.difficulty_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        label, difficulty_level = self.labels[idx], self.difficulty_list[idx]
        email_content = open(email_file_path, encoding="ISO-8859-1").read()

        # Parse the email
        email_message = message_from_string(email_content)

        # Get 'From' header
        from_header = email_message['From'] # fixme: from header can be spoofed
        from_email_address = parseaddr(from_header)[1]

        # Get 'Subject' header and decode if needed
        subject = email_message['Subject']

        # Initialize an empty string to store HTML content
        text_content = email_message.get_payload()

        # Loop through each part in the email to find the HTML part
        for part in email_message.walk():
            if part.get_content_type() == "text/html":
                html_content = part.get_payload(decode=True).decode('ISO-8859-1')
                # Using BeautifulSoup to extract text from HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text()
                text_content = text_content.replace('\xa0', ' ')
                text_content = text_content.replace('\n', ' ')
                text_content = re.sub(' +', ' ', text_content)
                break  # No need to look further

        return email_file_path, (from_email_address, subject, text_content), label, difficulty_level

class Enron(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.file_list = list(df['file'])
        self.message_list = list(df['message'])

    def __len__(self):
        return len(self.message_list)

    def __getitem__(self, idx):
        email_file_path = self.file_list[idx]
        email_content = self.message_list[idx]
        parsed_email = message_from_string(email_content)

        # Extract header information
        from_email_address = parsed_email.get("From")
        subject = parsed_email.get("Subject")

        # Extract body content
        text_content = parsed_email.get_payload()
        return email_file_path, (from_email_address, subject, text_content)

if __name__ == '__main__':
    # dataset = Nazario(root_path = './datasets/Nazario_2005')
    # print(len(dataset))
    # print(dataset[2500])

    dataset = CSDMC(root_path = './datasets/CSDMC2010_SPAM/CSDMC2010_SPAM')
    print(dataset.get_dist) # 1 stands for a HAM and 0 stands for a SPAM.
    # print(dataset[0])

    # dataset = SpamAssassin(root_path = './datasets/spamassassin_2005')
    # classes, difficulty_levels = dataset.get_dist
    # print(classes, difficulty_levels)
    # print(dataset[500])

    # dataset = Enron(csv_path = './datasets/enron_mail_2015/emails.csv')
    # print(len(dataset))
    # print(dataset[500])