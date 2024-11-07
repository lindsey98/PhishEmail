import mailbox
import email
import os


def mbox_to_eml(mbox_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the MBOX file
    mbox = mailbox.mbox(mbox_path)

    # Loop through all messages in the MBOX file
    for i, message in enumerate(mbox):
        # Extract the message as a string
        msg_str = message.as_string()

        # If no Message-ID, fall back to a fallback identifier (e.g., email index)
        safe_message_id = f"email_{i + 1}"

        # Create a filename for the .eml file using the Message-ID or fallback
        eml_filename = os.path.join(output_dir, f"{safe_message_id}.eml")

        # Save the message to an .eml file
        with open(eml_filename, 'w') as eml_file:
            eml_file.write(msg_str)
