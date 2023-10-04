from torch.utils.data import Dataset
import os
from email import message_from_string
from email.utils import parseaddr
from collections import Counter
from scripts.utils import *

def question_template_brand(email_subject, email_body):
    question = f'''
        Given the email subject <start>{email_subject}<end>, and the email body <start>{email_body}<end>,
        Question: What is the brand's domain? Answer: 
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
        A. The email has a clear call to action, urging the recipient to click a link or provide sensitive information.
        B. The email doesn’t have a clear call to action.
    '''
    return {
        "role": "user",
        "content": question
    }


def process_email_parts(email_part, email_content_collection):
    html_tags = ['<table', '<tr', '<td', '<font', '<a', '<html', '<body', '<meta']

    content_type = email_part.get_content_type()

    # If it's plain text, check for HTML tags
    if content_type == "text/plain":
        payload = email_part.get_payload(decode=True).decode('ISO-8859-1')

        if any(tag in payload.lower() for tag in html_tags): # html inside the plain text
            content_type = "text/html"
            soup = BeautifulSoup(payload, 'html.parser')
            text_content = soup.get_text()
        else:
            text_content = payload

        text_content = remove_specific_special_chars(text_content)

        return email_content_collection + [(content_type, payload, text_content)]

    # If it's HTML, extract text content
    elif content_type.startswith("text/html"):
        payload = email_part.get_payload(decode=True).decode('ISO-8859-1')

        soup = BeautifulSoup(payload, 'html.parser')
        text_content = soup.get_text()
        text_content = remove_specific_special_chars(text_content)

        return email_content_collection + [(content_type, payload, text_content)]

    # Nested content
    elif content_type.startswith('multipart') or content_type.startswith('message/rfc'):
        try:
            subpart = email_part.get_payload(0)
        except TypeError: # sometimes the email_part._payload is a str, not a list, just return the payload directly
            text_content = email_part._payload
            text_content = remove_specific_special_chars(text_content)
            return email_content_collection + [(content_type, text_content, text_content)]

        return email_content_collection + process_email_parts(subpart, email_content_collection)

    # Embedded image
    elif content_type.startswith('image'):
        payload = email_part.get_payload(decode=True)
        return email_content_collection + [(content_type, payload, payload)]

    elif content_type.startswith('application'):
        return email_content_collection  # dangerous attachments, dont open

    elif content_type.startswith('message/') or  content_type.startswith('text/') or content_type.startswith('video/'):
        return email_content_collection # unimportant
    else:
        raise UnsupportedContentTypeError(f"Unsupported content type: {content_type}. Cannot process this part.")


def parse_email_content(email_content_collection):
    email_body_text = ''
    email_image = []
    email_links = []
    for content in email_content_collection:
        content_type, original, text = content
        if 'image' not in content_type:
            email_body_text += text
        else:
            attached_images = BytesIO(original)
            email_image.append(attached_images)

        if "text/html" in content_type:
            img_links = get_img_links(original)
            intext_images = load_images(img_links)
            email_image.extend(intext_images)

            links = get_links(original)
            email_links.extend(links)

    return email_body_text, email_image, email_links


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

        content_collection = []

        # Walk through each part of the email to find text or HTML body
        for part in email_message.walk():
            content_collection = process_email_parts(part, content_collection)

        # remove duplicates
        content_collection = sorted(set(content_collection), key=content_collection.index)

        return email_file_path, (from_email_address, subject, content_collection)

class SpamAssassin(Dataset):
    def __init__(self, root_path):
        file_list = []
        label_list = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if 'tar.bz' in filename:
                    continue
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)
                if 'ham' in dirpath: # benign
                    label_list.append(1)
                else: # spam, not phishing
                    label_list.append(0)

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
        from_header = email_message['From']  # fixme: from header can be spoofed
        from_email_address = parseaddr(from_header)[1]

        # Get 'Subject' header and decode if needed
        subject = email_message['Subject']

        content_collection = []

        # Walk through each part of the email to find text or HTML body
        for part in email_message.walk():
            content_collection = process_email_parts(part, content_collection)

        # remove duplicates
        content_collection = sorted(set(content_collection), key=content_collection.index)

        return email_file_path, (from_email_address, subject, content_collection), label

# class CSDMC(Dataset):
#     def __init__(self, root_path):
#         label_file = [x.strip() for x in open(os.path.join(root_path, 'SPAMTrain.label')).readlines()]
#         label_list = []
#         file_list = []
#         for line in label_file:
#             label, file = line.split(' ')
#             if file.lower().startswith('train'):
#                 file_list.append(os.path.join(root_path, 'TRAINING', file))
#             else:
#                 file_list.append(os.path.join(root_path, 'TESTING', file))
#             label_list.append(int(label))
#
#         for dirpath, dirnames, filenames in os.walk(root_path):
#             for filename in filenames:
#                 if not filename.endswith('eml'):
#                     continue
#                 full_path = os.path.join(dirpath, filename)
#                 if full_path not in file_list:
#                     file_list.append(full_path)
#                     label_list.append(-1) # unlabeled
#
#         self.file_list = file_list
#         self.labels = label_list
#
#     @property
#     def get_dist(self):
#         return Counter(self.labels)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         email_file_path = self.file_list[idx]
#         label = self.labels[idx]
#         email_content = open(email_file_path, encoding="ISO-8859-1").read()
#
#         # Parse the email
#         email_message = message_from_string(email_content)
#
#         # Get 'From' header
#         from_header = email_message['From'] # fixme: from header can be spoofed
#         from_email_address = parseaddr(from_header)[1]
#
#         # Get 'Subject' header and decode if needed
#         subject = email_message['Subject']
#
#         text_content = email_message.get_payload()
#
#         # Loop through each part in the email to find the HTML part
#         for part in email_message.walk():
#             if part.get_content_type() == "text/html":
#                 html_content = part.get_payload(decode=True).decode('ISO-8859-1')
#                 # Using BeautifulSoup to extract text from HTML
#                 soup = BeautifulSoup(html_content, 'html.parser')
#                 text_content = soup.get_text()
#                 text_content = text_content.replace('\xa0', ' ')
#                 text_content = text_content.replace('\n', ' ')
#                 text_content = re.sub(' +', ' ', text_content)
#                 break  # No need to look further
#
#         return email_file_path, (from_email_address, subject, text_content), label
#

#
# class Enron(Dataset):
#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path)
#         self.file_list = list(df['file'])
#         self.message_list = list(df['message'])
#
#     def __len__(self):
#         return len(self.message_list)
#
#     def __getitem__(self, idx):
#         email_file_path = self.file_list[idx]
#         email_content = self.message_list[idx]
#         parsed_email = message_from_string(email_content)
#
#         # Extract header information
#         from_email_address = parsed_email.get("From")
#         subject = parsed_email.get("Subject")
#
#         # Extract body content
#         text_content = parsed_email.get_payload()
#         return email_file_path, (from_email_address, subject, text_content)

if __name__ == '__main__':
    dataset = Nazario(root_path = './datasets/Nazario_2005')
    print(len(dataset))
    print(dataset[2500])

    # dataset = CSDMC(root_path = './datasets/CSDMC2010_SPAM/CSDMC2010_SPAM')
    # print(dataset.get_dist) # 1 stands for a HAM and 0 stands for a SPAM.
    # print(dataset[0])

    # dataset = SpamAssassin(root_path = './datasets/spamassassin_2005')
    # classes, difficulty_levels = dataset.get_dist
    # print(classes, difficulty_levels)
    # print(dataset[500])

    # dataset = Enron(csv_path = './datasets/enron_mail_2015/emails.csv')
    # print(len(dataset))
    # print(dataset[500])