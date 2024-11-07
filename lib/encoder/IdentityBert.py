from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import Tuple, Set, List
import torch
from ..utilities import Timer, Logger
import re
import numpy as np
import torch
import random
import numpy as np

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# fixme: several things to adjust:
#  1. What kind of aggregation strategy to use?
#  2. Whether to do tokenization first before the pipeline? By right, pipeline should handle the tokenization internally, however I find this extra pre-tokenization step can affect the final results.
#  3. How to rank the reported entities?

class IdentityBert:
    _CallerPrefix = "IdentityBert"

    def __init__(self,
                 identity_checkpoint_path: str,
                 aggregation_strategy="first"):
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

    # Optimized rank_entities function
    def rank_entities(self, temp_entities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not temp_entities:
            return []

        # Sort by average score descending
        temp_entities.sort(key=lambda x: x[1], reverse=True)
        return temp_entities

    @torch.inference_mode()
    def __call__(self, raw_text: str) -> Tuple[List[str], Set[str], List[str], Set[str], float]:

        # fixme: I dont want the URL during prediction
        processed_text = self.remove_urls(raw_text)
        # processed_text = self._tokenize(processed_text) # fixme: I find the results will be different if I do tokenization here
        with Timer() as timer:
            entities = self.classifier_pipeline(processed_text)

        # Temporary lists to store entities with their confidence scores
        temp_identities: List[Tuple[str, float]] = []
        relations: List[str] = []
        actions: Set[str] = set()

        for ent in entities:
            ent_label = ent['entity_group']
            ent_text = ent['word']
            ent_score = ent.get('score', 0.0)  # Get the confidence score
            if ent_label == 'identity':
                if ent_text not in ['[CLS]', '[UNK]']: # cannot be CLS token
                    temp_identities.append((ent_text, ent_score))
            elif ent_label == 'relation':
                relations.append(ent_text)
            else:
                actions.add(ent_text)

        # Rank entities
        ranked_identities = self.rank_entities(temp_identities)

        # Initialize list for identities (to preserve order)
        identities: List[str] = []

        # Add entities to the list, preserving order and avoiding duplicates
        for entity, avg_score in ranked_identities:
            if entity not in identities:
                identities.append(entity)

        # Adhoc fix for special identity: admin or domain address claimed in the sender name part
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
                    caller_prefix=IdentityBert._CallerPrefix)

        ## Beta: get next-step-of-engagement
        urls_after_actions = self.get_next_step_of_engagement(raw_text, actions)

        return identities, actions, relations, urls_after_actions, timer.interval


