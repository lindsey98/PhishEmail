import openai
import json
import torch
import time
from scripts.dataloader import *
from scripts.utils import *
from honeypot.web_utils.Logger import Logger
from honeypot.web_utils.CustomDriver import CustomWebDriver
import argparse
import yaml
import pandas
import os
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

class TestLLM():

    def __init__(self, param_dict: dict, proxies: dict = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proxies = proxies
        self.LLM_model = param_dict["LLM_model"]
        self.brand_prompt = param_dict['brand_recog']['prompt_path']
        self.brand_recog_temperature, self.brand_recog_max_tokens = param_dict['brand_recog']['temperature'], param_dict['brand_recog']['max_tokens']
        self.brand_recog_sleep = param_dict['brand_recog']['sleep_time']

        self.action_prompt = param_dict['action']['prompt_path']
        self.action_temperature, self.action_max_tokens = param_dict['action']['temperature'], param_dict['action']['max_tokens']
        self.action_sleep = param_dict['action']['sleep_time']

    def recognize_brand(self, email_subject: str, email_body: str) -> str:
        question = question_template_brand(email_subject, email_body)

        with open(self.brand_prompt, 'rb') as f:
            prompt = json.load(f)

        new_prompt = prompt
        new_prompt.append(question)

        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=self.brand_recog_temperature,
                    max_tokens=self.brand_recog_max_tokens,
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=CustomWebDriver._caller_prefix, warning=True)
                prompt[-1]['content'] = prompt[-1]['content'][:len(prompt[-1]['content']) // 2] # truncate the email content
                time.sleep(self.brand_recog_sleep)

        brand = ''.join([choice["message"]["content"] for choice in response['choices']])
        print(brand)
        if len(brand) > 0 and is_valid_domain(brand):
            return brand
        else:
            return ''

    def require_action(self, email_subject: str, email_body: str) -> bool:
        question = question_template_action(email_subject, email_body)

        with open(self.action_prompt, 'rb') as f:
            prompt = json.load(f)

        new_prompt = prompt
        new_prompt.append(question)

        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=self.action_temperature,
                    max_tokens=self.action_max_tokens,
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=CustomWebDriver._caller_prefix, warning=True)
                prompt[-1]['content'] = prompt[-1]['content'][:len(prompt[-1]['content']) // 2] # truncate the email content
                time.sleep(self.action_sleep)

        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        print(answer)
        if len(answer):
            if 'A.' in answer:
                return True
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./scripts/param_dict.yaml', help="Config .yaml path")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.proxy = "http://127.0.0.1:7890" # set openai proxy

    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    llm_cls = TestLLM(param_dict=param_dict,
                      # proxies={"http": "http://127.0.0.1:7890",
                      #          "https": "http://127.0.0.1:7890",
                      #         }
                      )

    # df = pandas.read_csv('./datasets/junk-mail.csv')
    # df = df.iloc[:, :8]
    # df.columns = ["subject", "body", "sender_name", "sender_address",
    #               "sender_type", "recipient_name", "recipient_address", "recipient_type"]
    #
    # for index, row in df.iterrows():
    #     email_subject = row["subject"]
    #     email_subject = email_subject.replace('[SPAM]', '')
    #     email_body = row["body"]
    #     email_sender_name = row["sender_name"]
    #     email_sender_address = row["sender_address"]
    #     recipient_name = row["recipient_name"]
    #     recipient_address = row["recipient_address"]
    #
    #     brand = llm_cls.recognize_brand(email_subject, email_body)
