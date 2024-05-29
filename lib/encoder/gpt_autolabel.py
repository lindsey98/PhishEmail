
from openai import OpenAI
import os
import json
from tqdm import tqdm
import tiktoken

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKENS = 3900  # Reserve tokens for the function call and other messages

def truncate_text(text, max_tokens):
    tokens = encoder.encode(text)
    return encoder.decode(tokens[:max_tokens])

schema = {
  "name": "email_annotation",
  "description": "Annotate sender's identity ('organization'), sender's relation to the recipient, e.g. colleague, student, etc, ('relation'), and the required action for the recipient ('action'). Annotate conservatively, do not output the phrase if the label is empty.",
  "parameters": {
    "type": "object",
    "properties": {
      "annotations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "text": {
              "type": "string",
              "description": "the relevant phrases identified"
            },
            "labels": {
              "type": "array",
              "items": {
                "type": "string",
                "description": "organization, relation or action"
              }
            }
          },
          "required": ["text", "labels"]
        }
      }
    },
    "required": ["annotations"]
  }
}


example_response_1 = "\"annotations\": [{\"text\": \"From: Chase Bank\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"from Chase Bank\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"Chase Bank Online Banking Support, N.A\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"Chase Bank, N.A. Member FDIC\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"Copyright 2006 Chase Bank\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"Chase customer\", \"labels\": [\"relation\"]}," \
                                        "{\"text\": \"you must renew (overview) the service(s) listed below or it will be deactivated and deleted\", \"labels\": [\"action\"]}," \
                                        "{\"text\": \"Login to your chase account\", \"labels\": [\"action\"]}]"

example_response_2 = "\"annotations\": [{\"text\": \"From: lanefinancialllc.com\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"Best regards Morton Lane Roger Beckwith Lane Financial LLC\", \"labels\": [\"organization\"]}," \
                                        "{\"text\": \"Dear Colleague\", \"labels\": [\"relation\"]}," \
                                        "{\"text\": \"access the paper under the PublicationsComparative or Arbitrage Analysis of Insurance Securities and Derivatives section of our site wwwlanefinancialllccom or directly through the following link\", \"labels\": [\"action\"]}]"

dataset_name = 'spam_archive_2023'
with open(f'./datasets/{dataset_name}_unique_annotation/all.json', 'r', encoding='utf-8') as f:
    raw_data_json = json.load(f)

# Check if the annotated file already exists and load it if it does
annotated_file_path = f'./datasets/{dataset_name}_unique_annotation/annotated_all.json'
if os.path.exists(annotated_file_path):
    with open(annotated_file_path, 'r', encoding='utf-8') as f:
        updated_entries = json.load(f)
else:
    updated_entries = []

for entry in tqdm(raw_data_json[:500]):
    if entry["Path"] in [e["Path"] for e in updated_entries]:
        continue

    parsed_email = entry['text']
    truncated_email = truncate_text(parsed_email, MAX_TOKENS - 500)  # Adjust 500 as needed based on your prompt

    few_shot_function_calling_example = client.chat.completions.create(
        model = "gpt-3.5-turbo-0613",
            messages = [
                {"role": "system", "content": "You are an assistant that annotates an email with phrases that are related to the sender's identity ('organization'), sender's relation to the recipient such as colleague or subordinate, etc ('relation'), and the required action for the recipient ('action'). Annotate at most 2 most important actions. Annotate conservatively, do not output the phrase if the label is empty/unsure."},
                {"role": "user", "content": "Subject: Your Chase Bank Account May Have Been Accessed By An Unauthorized\n Third Party \n From: Chase Bank \n Recipient address: nobody@login.example.com \n Body: Dear Chase customer , This is your official notification from Chase Bank that the service(s) listed below will be deactivated and deleted if not renewed immediately. Previous notifications have been sent to the Chase Online SM Contact assigned to this account. As the Primary Contact, you must renew (overview) the service(s) listed below or it will be deactivated and deleted. 1. SERVICE : Chase Bank Chase Online SM with Bill Payment. EXPIRATION: February 06, 2006 2. We recently reviewed your account, and suspect that your chase account may have been accessed by an unauthorized third party. Protecting the security of your account and of the chase network is our primary concern. Login to your chase account . In case you are not enrolled yet for Internet Banking, you will have to use your Social Security Number as both your Personal ID and Password and fill in the required information. How can I restore my account access? Please visit the Resolution Center and complete the \"Steps to Remove Limitations.\" Thank you, Chase Bank Online Banking Support, N.A. ************************************************************** IMPORTANT CUSTOMER SUPPORT INFORMATION ************************************************************** Please do not reply to this message. For any inquiries, contact Customer Service. Document Reference: (92051208). Chase Bank, N.A. Member FDIC. Equal Housing Lender. Copyright 2006 Chase Bank, N.A. All rights reserved."},
                {"role": "function", "name": "email_annotation", "content": example_response_1},
                # {"role": "user", "content": "Subject: USAA PRIME: CHOICE CATS FOR DIVERSIFYING INVESTORS \n From: lanefinancialllc.com \n Recipient address: dlouise@lanefinancialllc.com \n Body: Dear Colleague Please find posted on our web site our latest paper USAA PRIME Choice Cats for Diversifying Investors This paper  which contrasts two recent securitizations  is a contributed chapter to an upcoming risk management text by Prof Chris Culp of the University of Chicago You may access the paper under the PublicationsComparative or Arbitrage Analysis of Insurance Securities and Derivatives section of our site wwwlanefinancialllccom or directly through the following link httpwwwlanefinancialllccompubsec2USAA_Prime_9301pdf Best regards Morton Lane Roger Beckwith Lane Financial LLC If you wish to be removed from this notice list regarding publications on our web site please send a reply to this email with Unsubscribe in the subject field"},
                #  {"role": "function", "name": "email_annotation", "content": example_response_2},
                {"role": "user", "content": truncated_email}],
        functions=[schema],
        function_call={"name": "email_annotation"},
        temperature=0
    )

    returned_annotations = few_shot_function_calling_example.choices[0].message.function_call.arguments
    try:
        entry['annotations'] = json.loads(returned_annotations)['annotations']
        print(json.loads(returned_annotations)['annotations'])
    except json.decoder.JSONDecodeError:
        continue

    updated_entries.append(entry)

    with open(annotated_file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_entries, f, indent=4, ensure_ascii=False)

