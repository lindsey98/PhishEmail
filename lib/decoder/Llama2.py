from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Tuple, Set, List
import torch
from ..utilities import Timer, Logger
import re
import numpy as np
import torch
import random
import numpy as np
import ast
import json

class IdentityLlama:
    _CallerPrefix = "IdentityLlama"

    def __init__(self, identity_checkpoint_path: str):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(identity_checkpoint_path, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)
        self.max_seq_len = 2048
        self.gen_config = GenerationConfig.from_pretrained('NousResearch/Llama-2-7b-hf',
                                                  temperature=0.001,
                                                  max_new_tokens=50)

    def prepare_prompt(self, raw_input):
        instruction = "Step 1: Identify phrases that claim the identity of the sender (brand name or relation to the recipient such as colleague)." \
                      "Step 2: Identify the call-to-action phrases." \
                      "Step 3: Identify the phrases that suggest the relation of sender to the recipient. " \
                      "Step 1-3 must cite the phrases from the original email. No new phrases, no extra explanation."
        length_predefined_prompt = len(
            self.tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

        # Tokenize the input and truncate to the max_seq_len
        tokens = self.tokenizer.tokenize(raw_input)[:self.max_seq_len - length_predefined_prompt]  # Reserving space for an EOS token, if necessary
        cleaned_input = self.tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

        return f"### Instruction:\n{instruction}\n\n### Input:\n{cleaned_input}\n\n### Response:\n"

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=tokenized_prompt,
                                        generation_config=self.gen_config)
            generation_ids = output[0][len(tokenized_prompt[0]):]
            num_gen_tokens = len(generation_ids)
            generation = self.tokenizer.decode(generation_ids, skip_special_tokens=True)
        return dict(generation=generation,
                    generation_ids=generation_ids.tolist(),
                    num_gen_tokens=num_gen_tokens)

    @staticmethod
    def literal_parsing(regex_match):
        if regex_match:
            raw = regex_match.group(1).strip()
            # Fix single quotes inside the content
            fixed = re.sub(r"(?<=\w)'(?=\w)", "", raw)
            # Safely evaluate the fixed content
            try:
                text = ast.literal_eval(fixed)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing Step 1: {e}")
                text = []
        else:
            text = []

        return text

    def results_parsing(self, results):
        # Extract Step 1
        step_1_match = re.search(r"Step 1:\s*(\[.*?\])", results, re.DOTALL)
        step_1_text = self.literal_parsing(step_1_match)

        # Extract Step 2
        step_2_match = re.search(r"Step 2:\s*(\[.*?\])", results, re.DOTALL)
        step_2_text = self.literal_parsing(step_2_match)

        # Extract Step 3
        step_3_match = re.search(r"Step 3:\s*(\[.*?\])", results, re.DOTALL)
        step_3_text = self.literal_parsing(step_3_match)
        return step_1_text, step_2_text, step_3_text

    @staticmethod
    def remove_urls(text):
        pattern = r'\((https?://[^)]+)\)'
        return re.sub(pattern, '', text)

    def get_next_step_of_engagement(self, raw_text: str, actions: Set[str]):
        # Define a regex to extract URLs enclosed in parentheses
        url_pattern = re.compile(r'\((http[s]?://[^)]+)\)')

        # Split the email into sentences
        sentences = re.split(r'\.\s+', raw_text)  # Split on period followed by space

        # Find sentences with actions and the subsequent URLs
        action_sentences_indices = []
        for i, sentence in enumerate(sentences):
            if any(action.lower() in sentence.lower() for action in actions):  # Case insensitive check
                action_sentences_indices.append(i)

        urls_after_actions = set()
        for index in action_sentences_indices:
            if index + 1 < len(sentences):  # Check if there is a next sentence
                next_sentence = sentences[index + 1]
                match = url_pattern.search(next_sentence)
                if match:
                    urls_after_actions.add(match.group(1))  # Store URL found after the action sentence

        return urls_after_actions


    @torch.inference_mode()
    def __call__(self, raw_text: str) -> Tuple[List[str], Set[str], List[str], Set[str], float]:

        # fixme: I dont want the URL during prediction
        processed_text = self.remove_urls(raw_text)
        processed_text = self.prepare_prompt(processed_text)
        with Timer() as timer:
            results = self.generate(processed_text)

        identities, actions, relations  = self.results_parsing(results['generation'])
        actions = set(actions)

        # # Adhoc fix for special identity: admin or domain address claimed in the sender name part
        if not identities:
            pattern = r'^\s*From:\s*(admin[\w-]*)'
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = regex.findall(raw_text)
            if matches:
                # Adding in order of matches; since lists preserve insertion order
                for match in matches:
                    if match not in identities:
                        identities.append(match)

        if not identities:
            pattern = r'^\s*From:\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)'
            # Compile the regex with case-insensitive and multiline flags
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = regex.findall(raw_text)
            if matches:
                for match in matches:
                    if match not in identities:
                        identities.append(match)

        Logger.spit(f"Recognized identities = {identities}, "
                    f"recognized actions = {actions}, "
                    f"potential relations to the sender = {relations}",
                    debug=True,
                    caller_prefix=IdentityLlama._CallerPrefix)
        #
        ## Beta: get next-step-of-engagement
        urls_after_actions = self.get_next_step_of_engagement(raw_text, actions)

        return identities, actions, relations, urls_after_actions, timer.interval


if __name__ == '__main__':
    model = IdentityLlama("./checkpoints/llama2/checkpoint-3304")
    raw_text = """Subject: Black Hat organization USA 2023: Invitation for Leading Cybersecurity Experts. \n From: Black Hat organization \n Body: 
    Dear Davide,Black Hat USA organization is thrilled to invite you to join us at our renowned cybersecurity conference this year. As a leading researcher with a distinguished background in CTF competitions, particularly your involvement in the Order of the Overflow and DefCon CTF, we believe your expertise would be invaluable to our community.
    Black Hat organization USA 2023 will feature a diverse range of sessions, workshops, and networking opportunities focused on the latest advancements in cybersecurity. We are particularly excited about our dedicated track focused on offensive security, where you could share your insights and engage with fellow cybersecurity professionals.
    We are confident that your participation would enrich the conference experience for all attendees. To learn more about Black Hat USA organization 2023 and to register, please visit: https://tinyurl.com/ABCDEFG
    We eagerly await your confirmation and hope to welcome you to Black Hat USA 2023.
    Sincerely, The Black Hat organization"""

    model(raw_text)