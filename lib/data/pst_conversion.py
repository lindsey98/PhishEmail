import os
import pypff
import email
from email import policy
from email.generator import BytesGenerator


def sanitize_filename(name):
    # Remove or replace characters that are invalid in filenames
    if name:
        invalid_chars = r'<>:"/\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()
    else:
        return ''

def extract_emails_and_folders(folder, output_dir):
    # Iterate through items in the folder
    for item in folder.sub_items:
        if isinstance(item, pypff.folder):
            # Recursively process subfolder
            subfolder_output_dir = os.path.join(output_dir, item.name)
            os.makedirs(subfolder_output_dir, exist_ok=True)
            extract_emails_and_folders(item, subfolder_output_dir)
        elif isinstance(item, pypff.message):
            # Process email
            process_email(item, output_dir)


def process_email(email, output_dir):
    # Create .eml filename based on subject or a unique identifier
    eml_filename = os.path.join(output_dir, f"{email.identifier}.eml")

    with open(eml_filename, 'w', encoding='utf-8') as eml_file:
        # Write headers
        eml_file.write(f"{email.transport_headers}\n")
        eml_file.write("\n")  # Blank line to separate headers from body

        # Write email body
        try:
            email_body = email.html_body.decode('utf-8')
        except:
            email_body = str(email.html_body)
        eml_file.write(email_body)


def pst_to_eml(pff_file_path, output_dir):
    # Open the pst file
    pff_file = pypff.file()
    pff_file.open(pff_file_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process root folder
    root_folder = pff_file.get_root_folder()
    extract_emails_and_folders(root_folder, output_dir)

    pff_file.close()

