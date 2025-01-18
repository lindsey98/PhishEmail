

import re
from bs4 import BeautifulSoup
from email import message_from_file

css_block = """
.entity {
    display: inline-block;
    position: relative;
    padding: 2px 4px;
    margin: 0 2px;
    line-height: 1.5;
    border-radius: 4px;
}
.entity::after {
    content: attr(data-class); /* Use the data-class attribute for annotation */
    position: absolute;
    top: -12px; /* Move slightly above the text */
    right: -30px; /* Shift further to the right */
    background: white; /* Background for better visibility */
    color: black; /* Text color */
    font-size: 10px; /* Small font size */
    padding: 2px 4px; /* Padding for better visibility */
    border: 1px solid #ccc; /* Light border */
    border-radius: 4px; /* Rounded corners */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Light shadow */
    white-space: nowrap; /* Prevent wrapping */
    z-index: 1000; /* Ensure it is on top */
}
.entity.identity {
    background-color: #DFF4E1; /* Lighter Medium Slate Blue */
}
.entity.action {
    background-color: #F4D1C1; /* Lighter Indian Red */
}
.entity.sender {
    background-color: #DFF4E1; /* Lighter Lime Green */
}
.popup {
    visibility: hidden;
    opacity: 0;
    width: 200px;
    background-color: #333;
    color: #fff;
    text-align: center;
    padding: 10px;
    border-radius: 5px;
    position: absolute;
    bottom: 120%; /* Position above the span */
    left: 50%;
    transform: translateX(-50%);
    transition: opacity 0.3s ease, visibility 0.3s ease;
    z-index: 1;
}

/* Arrow below the popup */
.popup::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 5px;
    border-style: solid;
    border-color: #333 transparent transparent transparent;
}

/* Show the popup on hover */
.entity:hover .popup {
    visibility: visible;
    opacity: 1;
}
""" 

# Function to wrap matches with <strong> tags
def wrap_identity(match):
    return f"<span class='entity identity' data-class='Identity'>{match.group(1)}</span>"

def wrap_actions(match):
    return f"<span class='entity action' data-class='Action'>{match.group(1)}<span class='popup'>Asks the recepient to '{match.group(1)}'</span></span>"

def wrap_sender(text, identity, is_inconsistent):
    if is_inconsistent:
        return f"<span class='entity sender' data-class='Sender'>{text}<span class='popup alert'>Claims to be {identity} but sent from domain '{text.split('@')[-1].removesuffix('>')}'</span></span>"
    else:
        return f"<span class='entity sender' data-class='Sender'>{text}</span>"
    
def append_css_styling(html_content, css_block):
    # Check if there's a <head> section
    if "<head>" in html_content:
        # Insert CSS inside the <head> section
        modified_html = re.sub(r"(<head>)", r"\1\n<style>\n" + css_block + r"\n</style>", html_content, count=1, flags=re.IGNORECASE)
    else:
        # Prepend CSS at the start of the document if no <head> exists
        modified_html = f"<style>\n{css_block}\n</style>\n" + html_content
    return modified_html

def unwrap_links(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for a_tag in soup.find_all('a'):
        a_tag.unwrap()
    for span_tag in soup.find_all('span'):
        span_tag.unwrap()
    
    return str(soup)

def remove_images(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for img_tag in soup.find_all('img'):
        img_tag.decompose()

    return str(soup)

# Function to process and modify the text/html part
def label_html(html_content, identities, actions):
    # Escape strings to handle special characters
    escaped_identities = [re.escape(s) for s in identities]
    escaped_actions = [re.escape(s) for s in actions]
    # Create a regex pattern to match any of the target strings
    pattern_identities = r'(' + '|'.join(escaped_identities) + r')'
    pattern_actions = r'(' + '|'.join(escaped_actions) + r')'

    # Use re.sub to wrap matches
    html_content = re.sub(pattern_identities, wrap_identity, html_content, flags=re.IGNORECASE)
    html_content = re.sub(pattern_actions, wrap_actions, html_content, flags=re.IGNORECASE)
    return html_content

def label_headers(sender, recipient, subject, identities, actions, is_inconsistent, matched_identity):
    escaped_identities = [re.escape(s) for s in identities]
    escaped_actions = [re.escape(s) for s in actions]

    pattern_identities = r'(' + '|'.join(escaped_identities) + r')'
    pattern_actions = r'(' + '|'.join(escaped_actions) + r')'

    sender = wrap_sender(sender, matched_identity, is_inconsistent)

    recipient = re.sub(pattern_identities, wrap_identity, recipient, flags=re.IGNORECASE)
    recipient = re.sub(pattern_actions, wrap_actions, recipient, flags=re.IGNORECASE)

    subject = re.sub(pattern_identities, wrap_identity, subject, flags=re.IGNORECASE)
    subject = re.sub(pattern_actions, wrap_actions, subject, flags=re.IGNORECASE)

    return sender, recipient, subject
# Open and process the .eml file
def label_eml_file(file: str, identities, actions):
    eml_file_path = file  # Replace with the path to your .eml file

    # Read the .eml file
    with open(eml_file_path, "r") as eml_file:
        message = message_from_file(eml_file)

    # Iterate through the parts of the email
    for part in message.walk():
        if part.get_content_type() == "text/html":  # Only modify text/html parts
            html_content = part.get_payload(decode=True).decode(part.get_content_charset())
            html_content = append_css_styling(html_content, css_block)
            html_content = unwrap_links(html_content)
            html_content = label_html(html_content, identities, actions)
            
            part.set_payload(html_content)

    # Save the modified .eml file
    output_eml_path = file.removesuffix('.eml') + '_labelled.eml'
    with open(output_eml_path, "w") as output_file:
        output_file.write(message.as_string())

    output_mht_path = file.removesuffix('.eml') + '_labelled.mht'
    with open(output_mht_path, "w") as output_file:
        output_file.write(message.as_string())

    print(f"Modified .eml file saved to {output_eml_path}")

    return output_mht_path

# extract only the html, removing any image tags
def label_html_file(file: str, identities, actions):
    html_file_path = file  # Replace with the path to your .eml file

    # Read the .eml file
    with open(html_file_path, "r") as eml_file:
        message = message_from_file(eml_file)

    # Iterate through the parts of the email
    for part in message.walk():
        if part.get_content_type() == "text/html":  # Only modify text/html parts
            html_content = part.get_payload(decode=True).decode(part.get_content_charset())
            html_content = unwrap_links(html_content)
            html_content = remove_images(html_content)
            html_content = label_html(html_content, identities, actions)
            
    output_html_path = file.removesuffix('.eml') + '_labelled.html'
    with open(output_html_path, "w") as output_file:
        html_content_with_css = append_css_styling(html_content, css_block)
        output_file.write(html_content_with_css)

    return html_content

if __name__ == '__main__':
    identities = ['university of toronto', 'university of toronto summer programme 2025', 'sci ug comms', 'hkust and', 'thehongkong universityofscience andtechnology', 'fos', 'abroad team']
    actions = ['apply by 7 jan 2025', 'refer to our website for details of the summer programme', 'please read the guide for student programme application attached before you login to edurec to complete your application', "search for under external study type ' summer / winter '", 'submit application online via edurec', 'deemed eligible']

    label_html_file("addin/temp.eml", identities, actions)