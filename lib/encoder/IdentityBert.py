from transformers import pipeline, AutoTokenizer
from typing import Tuple, Set
import torch
from lib.utilities.logger import Timer
import re


class IdentityBert:
    _CallerPrefix = "IdentityBert"

    def __init__(self, identity_checkpoint_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier_pipeline = pipeline("ner", model=identity_checkpoint_path, device=self.device, aggregation_strategy="simple")
        self.tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)

    @staticmethod
    def remove_urls(text):
        pattern = r'\((https?://[^)]+)\)'
        return re.sub(pattern, '', text)

    def _tokenize(self, text: str) -> str:
        tokens = self.tokenizer(text, return_offsets_mapping=True, truncation=True)
        tokenized_email = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        tokenized_email_str = self.tokenizer.convert_tokens_to_string(tokenized_email)
        return tokenized_email_str

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
    def __call__(self, raw_text: str) -> Tuple[Set[str], Set[str], Set[str], Set[str], float]:

        # fixme: I dont want the URL during prediction
        processed_text = self.remove_urls(raw_text)
        with Timer() as timer:
            entities = self.classifier_pipeline(processed_text)

        identities = set()
        relations = set()
        actions = set()

        for ent in entities:
            ent_label = ent['entity_group']
            ent_text = ent['word']
            if ent_label == 'identity':
                identities.add(ent_text)
            elif ent_label == 'relation':
                relations.add(ent_text)
            else:
                actions.add(ent_text)

        ## Beta: get next-step-of-engagement
        urls_after_actions = self.get_next_step_of_engagement(raw_text, actions)

        return identities, actions, relations, urls_after_actions, timer.interval


