from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import Tuple, Set
import torch
from lib.utilities.logger import Timer, Logger
import re
import numpy as np

class IdentityBert:
    _CallerPrefix = "IdentityBert"

    def __init__(self,
                 identity_checkpoint_path: str,
                 aggregation_strategy="simple"):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForTokenClassification.from_pretrained(identity_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)
        self.classifier_pipeline = pipeline("ner",
                                            tokenizer=self.tokenizer,
                                            model=self.model,  # Use the already loaded model
                                            device=self.device,
                                            aggregation_strategy=aggregation_strategy)

    def get_pred(self, input_ids):
        probs = self.get_prob(input_ids)
        pred_classes = probs.argmax(axis=-1)
        return pred_classes

    @torch.inference_mode()
    def get_prob(self, input_ids):
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids)[0][0].cpu().numpy()
        return np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True) # B x C

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

        Logger.spit(f"Recognized identities = {identities}, recognized actions = {actions}", debug=True, caller_prefix=IdentityBert._CallerPrefix)
        ## Beta: get next-step-of-engagement
        urls_after_actions = self.get_next_step_of_engagement(raw_text, actions)

        return identities, actions, relations, urls_after_actions, timer.interval


