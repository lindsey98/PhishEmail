import string
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from lib.data import RenderDataset, OCR
from lib.reference_db import IdentityMatcher
from inference import Config
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from label_eml import label_html_file, label_headers
from nltk.corpus import stopwords
import base64
import re


app = Flask(__name__)
CORS(app)
matcher_cls = IdentityMatcher(brand_index_db=Config.brand_index_db,
                                  internal_relation_index_db=Config.internal_relation_index_db,
                                  embed_model=Config.matching_model,
                                  brand_domain_map_path=Config.brand_domain_map_path,
                                  knowledge_base_expansion=False,
                                  gpt_client=None,
                                  gpt_assistant=None,
                                  check_action=True,
                                  threshold=Config.thre,
                                  relax_match=True)
ocr_model = OCR()
stop_words = set(stopwords.words('english'))
stop_words.add('edu')
stop_words.add('sg')

# TODO: make safe renderable version for HTML
def email_address_format(email):
    if 'displayName' in email:
        return f"{email['displayName']} <{email['emailAddress']}>"
    else:
        return email['emailAddress']
    
def fix_image_refs(body, name):
    pattern = rf'{re.escape(name)}(@[\w.]+(?=["\']))'
    match = re.findall(pattern, body)[0]
    return body.replace(match, '')

def build_mime_email(email_json):

    # Create the root message
    msg = MIMEMultipart('related')
    if 'sender' in email_json and len(email_json['sender']):
        msg['From'] = email_address_format(email_json['sender'])

    if 'to' in email_json and len(email_json['to']):
        to_emails = []
        for email in email_json['to']:
            to_emails.append(email_address_format(email))
        msg['To'] = ', '.join(to_emails)

    if 'cc' in email_json and len(email_json['cc']):
        cc_emails = []
        for email in email_json['cc']:
            cc_emails.append(email_address_format(email))
        msg['Cc'] = ', '.join(cc_emails)

    if 'bcc' in email_json and len(email_json['bcc']):
        bcc_emails = []
        for email in email_json['bcc']:
            bcc_emails.append(email_address_format(email))
        msg['Bcc'] = ', '.join(bcc_emails)

    if 'subject' in email_json:
        msg['Subject'] = email_json['subject']

    # Create the HTML part with inline images
    if len(email_json['attachments']):
        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)

        # Attach the HTML content
        body = email_json['body']
        if len(email_json['attachments']):
            body = fix_image_refs(email_json['body'], email_json['attachments'][0]['name'])
        msg_alternative.attach(MIMEText(body, 'html'))

        # Attach images as inline content
        for image in email_json['attachments']:
                mime_image = MIMEImage(base64.b64decode(image['content']))
                mime_image.add_header('Content-ID', f"<{image['name']}>")  # Reference as <image0>, <image1>, etc.
                mime_image.add_header('Content-Disposition', 'inline', filename=image['name'])
                msg.attach(mime_image)
    
    else:
        body = email_json['body']
        msg.attach(MIMEText(body, 'html'))

    with open('./addin/temp.eml', 'w') as f:
        f.write(msg.as_string())

    return msg['From'], msg['To'], msg['Subject']

def process_email():
    dataset = RenderDataset('./addin', ocr_model=ocr_model, dumpDir= './addin_imgs', save_imgs=True, translate_on = False)
    email_file_path, (sender_name, sender_address), \
            (to_names, to_addresses), reply_to_address, \
            subject, email_body_text, header = dataset[0]
    print(email_body_text)
    sender_domains = dataset.domain_parsing(sender_address)

    parsed_email = f'Subject: {subject} \n From: {sender_name} \n Body: {email_body_text}'
    identities, actions, relations, urls_after_actions, identity_recog_runtime = Config.identity_model(parsed_email)

    if len(urls_after_actions):
        next_step_of_engagement = urls_after_actions
    else:
        next_step_of_engagement = reply_to_address

    next_step_of_engagement_domains = dataset.domain_parsing(next_step_of_engagement)

    recipient_domains = dataset.domain_parsing(to_addresses)
    sender_domains = sender_domains.union(next_step_of_engagement_domains)

    is_inconsistent, matched_identity, identity_matching_runtime = matcher_cls(identities=identities,
                                                                                       actions=actions,
                                                                                       relations=relations,
                                                                                       sender_domains=sender_domains,
                                                                                       recipient_domains=recipient_domains)
    

    def remove_trailing_punctuation_and_whitespace(s):
        return s.rstrip(string.punctuation + string.whitespace)
    identities_clean = [identity for identity in identities if identity not in stop_words and identity not in string.punctuation]
    actions_clean = [remove_trailing_punctuation_and_whitespace(action) for action in actions if action not in stop_words and action not in string.punctuation]
    
    print(identities_clean)
    print(actions_clean)

    return is_inconsistent, matched_identity, actions_clean, identities_clean

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()

    try:
        sender, recipient, subject = build_mime_email(data)
        is_inconsistent, matched_identity, actions, identities = process_email()
        labelled_eml = label_html_file('addin/temp.eml', identities, actions)
        labelled_sender, labelled_recipient, labelled_subject = label_headers(sender, recipient, subject, identities, actions, is_inconsistent, matched_identity)
        response = jsonify({"status": "success", 
                            "isInconsistent": is_inconsistent, 
                            "matchedIdentity": matched_identity if isinstance(matched_identity, str) else list(matched_identity)[0],
                            "identities": list(identities),
                            "actions": list(actions),
                            "labelledHTML": labelled_eml,
                            "labelledSender": labelled_sender,
                            "labelledRecipient": labelled_recipient,
                            "labelledSubject": labelled_subject 
                            })
        return response
    except Exception as e:
        response = jsonify({"status": "error", "message": str(e)})
        return response

@app.route('/addin/<path:path>')
def send_report(path):
    return send_from_directory('addin', path)

if __name__ == '__main__':
    app.run(debug=True)
