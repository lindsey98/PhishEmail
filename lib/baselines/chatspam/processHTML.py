
import email
import lxml.html
from nltk import tokenize
import htmlUtil

TOKEN_LIMIT = 3000

def exceed_length(text):
    #return len(encoding.encode(text)) > TOKEN_LIMIT
    return len(text) > TOKEN_LIMIT * 4

def filter_headers(text):
    filtered_headers = ['date', 'from', 'to', 'subject', 'content-type']
    for header in filtered_headers:
        if text.lower().startswith(header):
            return True
    return False

def clean_plain(plain_text):
    sentences = tokenize.sent_tokenize(plain_text)
    while exceed_length(' '.join(sentences)):
        sentences.pop(len(sentences) // 2)
    return ' '.join(sentences)

def get_email_text(filename):
    try:
        with open(filename, 'r', encoding='UTF-8') as f:
            mail_parsed = email.message_from_file(f)
    except UnicodeDecodeError:
        with open(filename, 'r', encoding="ISO-8859-1") as f:
            mail_parsed = email.message_from_file(f)

    html_text = []
    plain_text = []

    # if no content type provided, default is text/plain
    for part in mail_parsed.walk():
        if part.get_content_maintype() != 'text' and part.get_content_maintype() != 'multipart':
            continue

        payload = str(part.get_payload(decode=True))
        if part.get_content_type() == 'text/html':
            html_text.append(payload)
        elif part.get_content_maintype() == 'text':
            plain_text.append(payload)
        elif payload is not None:
            plain_text.append(payload)

    if len(html_text) + len(plain_text) == 0:
        print(f"{filename} missing!")
        print(html_text)
        print(plain_text)

    final_text = ''

    # Step 0: delete useless headers
    for mail_key in mail_parsed.keys():
        if not filter_headers(mail_key):
            mail_parsed.__delitem__(mail_key)

    # Step 1: prioritize html_text, do not take any plain_text if over limit
    for text in html_text:
        final_text = final_text + text + '\n\n'

    for text in plain_text:
        if exceed_length(final_text):
            break
        final_text = final_text + text + '\n\n'

    # Step 2: remove tags if html
    if exceed_length(final_text) and html_text:
        final_text = clean_html(final_text)

    # Step 3: strip text if plain
    elif exceed_length(final_text) and plain_text:
        final_text = clean_plain(final_text)

    # print(len(encoding.encode(final_text)))
    mail_parsed.set_payload(final_text)
    return mail_parsed


def clean_html(html_text):
    html = lxml.html.fromstring(html_text)

    htmlUtil.clean_script_style_comments(html)
    if not exceed_length(lxml.html.tostring(html).decode('utf-8')):
        return lxml.html.tostring(html).decode('utf-8')

    htmlUtil.clean_attributes(html)
    if not exceed_length(lxml.html.tostring(html).decode('utf-8')):
        return lxml.html.tostring(html).decode('utf-8')

    htmlUtil.clean_textless_tag(html)
    if not exceed_length(lxml.html.tostring(html).decode('utf-8')):
        return lxml.html.tostring(html).decode('utf-8')

    htmlUtil.unwrap_tag(html)
    if not exceed_length(lxml.html.tostring(html).decode('utf-8')):
        return lxml.html.tostring(html).decode('utf-8')

    htmlUtil.shorten_url(html)
    if not exceed_length(lxml.html.tostring(html).decode('utf-8')):
        return lxml.html.tostring(html).decode('utf-8')

    while exceed_length(lxml.html.tostring(html).decode('utf-8')):
        htmlUtil.remove_middle(html)

    return lxml.html.tostring(html).decode('utf-8')
