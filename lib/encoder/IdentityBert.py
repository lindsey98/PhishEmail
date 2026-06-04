import random
import re
from typing import List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from ..utilities import Logger, Timer
from ..utilities.data_utils import remove_urls

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class IdentityBert:
    _CallerPrefix = "IdentityBert"

    def __init__(self, identity_checkpoint_path: str, aggregation_strategy="first"):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForTokenClassification.from_pretrained(identity_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)
        self.max_length = self.tokenizer.model_max_length
        self.pad_to_max_length = False
        self.classifier_pipeline = pipeline(
            "ner",
            tokenizer=self.tokenizer,
            model=self.model,  # Use the already loaded model
            device=self.device,
            aggregation_strategy=aggregation_strategy,
        )

    def get_pred(self, input_ids: torch.Tensor) -> torch.Tensor:
        probs = self.get_prob(input_ids)
        pred_classes = probs.argmax(axis=-1)
        return pred_classes

    @torch.inference_mode()
    def get_prob(self, input_ids: torch.Tensor) -> np.ndarray:
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids)[0][0].cpu().numpy()
        return np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)  # B x C

    def _tokenize(self, text: str) -> List[str]:
        """
        Perform separate tokenization
        :param text:
        :return:
        """
        prefix = "Subject: \nFrom: \nBody: \n"
        prepend_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        prepend_length = len(prepend_ids)
        reserved_tokens = prepend_length + 2  # prepend_ids + BOS + EOS
        max_tokens_subsequent = self.max_length - reserved_tokens
        bos_id = 101
        eos_id = 102

        tokens = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=False,  # Disable automatic truncation
            add_special_tokens=True,
        )

        input_ids = tokens["input_ids"]
        total_length = len(input_ids)
        batches = []

        # First batch: handle separately
        first_batch_ids = input_ids[: self.max_length]
        if len(first_batch_ids) >= self.max_length:
            first_batch_ids = input_ids[: self.max_length - 1] + [eos_id]

        first_batch_tokens = self.tokenizer.convert_ids_to_tokens(first_batch_ids)
        first_batch_str = self.tokenizer.convert_tokens_to_string(first_batch_tokens)
        batches.append(first_batch_str)

        for i in range(self.max_length, total_length, max_tokens_subsequent):
            # Slice the input_ids for the current batch
            batch_ids = input_ids[i : i + max_tokens_subsequent]

            # Combine prepend_ids with the current batch_ids
            combined_ids = prepend_ids + batch_ids

            # Add BOS and EOS tokens
            combined_ids = [bos_id] + combined_ids + [eos_id]

            # Calculate the number of tokens after adding prepend and special tokens
            current_length = len(combined_ids)

            # Truncate if necessary
            if current_length > self.max_length:
                combined_ids = combined_ids[: self.max_length]

            combined_tokens = self.tokenizer.convert_ids_to_tokens(combined_ids)
            combined_str = self.tokenizer.convert_tokens_to_string(combined_tokens)
            batches.append(combined_str)

        return batches

    def get_next_step_of_engagement(self, raw_text: str, actions: Set[str]) -> Set[str]:
        """
        Get the URLs near the call-to-actions phrases, if any
        :param raw_text:
        :param actions:
        :return:
        """
        # Define a regex to extract URLs enclosed in parentheses
        url_pattern = re.compile(r"\((http[s]?://[^)]+)\)")

        # Split the email into sentences
        sentences = re.split(r"\.\s+", raw_text)  # Split on period followed by space

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

    def rank_entities(self, temp_entities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Rank entities based on confidences
        :param temp_entities:
        :return:
        """
        if not temp_entities:
            return []

        # Sort by average score descending
        temp_entities.sort(key=lambda x: x[1], reverse=True)
        return temp_entities

    @torch.inference_mode()
    def __call__(self, raw_text: str) -> Tuple[List[str], Set[str], List[str], Set[str], float]:

        # fixme: I dont want the URL during prediction
        processed_text = remove_urls(raw_text)
        processed_text = self._tokenize(
            processed_text
        )  # fixme: I find the results will be different if I do tokenization here
        with Timer() as timer:
            entities = self.classifier_pipeline(processed_text)

        entities = [y for x in entities for y in x]
        # Temporary lists to store entities with their confidence scores
        temp_identities: List[Tuple[str, float]] = []
        relations: List[str] = []
        actions: Set[str] = set()

        for ent in entities:
            ent_label = ent["entity_group"]
            ent_text = ent["word"]
            ent_score = ent.get("score", 0.0)  # Get the confidence score
            if ent_text not in ["[CLS]", "[SEP]", "[PAD]"]:  # cannot be CLS token
                if ent_label == "identity":
                    if ent_text not in ["[UNK]"]:
                        temp_identities.append((ent_text, ent_score))
                elif ent_label == "relation":
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
            pattern = r"^\s*From:\s*(admin[\w-]*)"
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = regex.findall(raw_text)
            if matches:
                # Adding in order of matches; since lists preserve insertion order
                for match in matches:
                    if match not in identities:
                        identities.append(match)

        if not identities:
            pattern = r"^\s*From:\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)"
            # Compile the regex with case-insensitive and multiline flags
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = regex.findall(raw_text)
            if matches:
                for match in matches:
                    if match not in identities:
                        identities.append(match)

        if not identities:
            with Timer() as timer:
                entities = self.classifier_pipeline([raw_text])
            entities = [y for x in entities for y in x]
            temp_identities: List[Tuple[str, float]] = []

            for ent in entities:
                ent_label = ent["entity_group"]
                ent_text = ent["word"]
                ent_score = ent.get("score", 0.0)  # Get the confidence score
                if ent_text not in ["[CLS]", "[SEP]", "[PAD]"]:  # cannot be CLS token
                    if ent_label == "identity":
                        if ent_text not in ["[UNK]"]:
                            temp_identities.append((ent_text, ent_score))
            # Rank entities
            ranked_identities = self.rank_entities(temp_identities)
            identities = list(set([entity for entity, score in ranked_identities]))

        Logger.spit(
            f"Recognized identities = {identities}, "
            f"recognized actions = {actions}, "
            f"potential relations to the sender = {relations}",
            caller_prefix=IdentityBert._CallerPrefix,
        )

        ## Beta: get next-step-of-engagement
        urls_after_actions = self.get_next_step_of_engagement(raw_text, actions)

        return identities, actions, relations, urls_after_actions, timer.interval
