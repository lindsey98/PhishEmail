import warnings
warnings.filterwarnings('ignore')
import argparse
from lib.baselines.helphed.getFeatures import parse_email_parts
from lib.data.Dataset import EmailDataset
import configparser
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import time
from openai import OpenAI
from lib.baselines.chatspam.processHTML import get_email_text

'''
Main function
'''
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open(f'./lib/baselines/chatspam/prompt.json', 'rb') as file:
    SYSTEM_PROMPT = json.load(file)

def test(email_dir, results_save_dir):

    os.makedirs(results_save_dir, exist_ok=True)
    test_dataset = EmailDataset(email_dir)
    for it in tqdm(range(len(test_dataset))):

        email_file_path = test_dataset.file_list[it]
        parsed_email = get_email_text(email_file_path)

        # parsed_email = f'Subject: {subject} \n From: {sender_name} \n Body: {email_body_text}'

        result_save_path = os.path.join(results_save_dir, os.path.basename(email_file_path) + '.json')
        if os.path.exists(result_save_path) and len(open(result_save_path).read()):
            continue

        # Combine system and sample-specific prompts
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Email: ```{parsed_email}```"""}
        ]

        inference_done = False
        while not inference_done:
            try:
                # Make the API call
                start_time = time.time()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                runtime = time.time() - start_time

                prediction = response.choices[0].message.content
                # Get token usage
                total_tokens = response.usage.total_tokens

                # Calculate cost based on OpenAI's pricing (e.g., $0.002 per 1k tokens)
                cost_per_1000_tokens = 0.002
                cost = (total_tokens / 1000) * cost_per_1000_tokens
                inference_done = True
            except Exception as e:
                messages[-1]['content'] = messages[-1]['content'][:len(messages[-1]['content']) // 2] # cut
                time.sleep(1)

        results = {
            "runtime": runtime,
            "prediction": prediction,
            "total_tokens": total_tokens,
            "cost": cost
        }
        with open(result_save_path, "w") as handle:
            json.dump(results, handle)

        # Print the cost
        print(f"Time taken: {runtime:.4f} seconds")
        print(f"Total tokens used: {total_tokens}")
        print(f"Cost of the API call: ${cost:.4f}")


def results_calculation(results_save_dir):
    report_ct = 0
    total = 0
    time_list = []
    for file in os.listdir(results_save_dir):
        with open(os.path.join(results_save_dir, file), "r") as handle:
            result = json.load(handle)

        if file.replace(".json", "") not in open(f'./datasets/nazario_noisy_list.txt').read():
            total += 1
            # if "Yes, it is phishing" in result['prediction']:
            if "\"is_phishing\": true" in (result['prediction']):
                report_ct += 1

        time_list.append(float(result['runtime']))

    print(f'Reported = {report_ct}, % = {report_ct / total}')
    print(f'Median time = {np.median(time_list)}')

if __name__ == '__main__':
    # desc_folder = './datasets/GPT_V6/v6'
    # test(desc_folder, results_save_dir='./datasets/GPT_results_prompt3_ChatSpamDetector')

    # results_calculation('./datasets/GPT_results_prompt3_ChatSpamDetector')

    # desc_folder = './datasets/nazario-recent'
    # test(desc_folder, results_save_dir='./datasets/nazario_prompt3_ChatSpamDetector')
    # results_calculation('./datasets/nazario_prompt3_ChatSpamDetector')

    desc_folder = './datasets/field/ruofan_honeypot/Mail/Starred'
    test(desc_folder, results_save_dir='./datasets/field/ruofan_honeypot/honeypot_ChatSpamDetector')
    # results_calculation('./datasets/field/ruofan/ruofanINBOX_ChatSpamDetector')
