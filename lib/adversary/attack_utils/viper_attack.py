import copy
import random
from OpenAttack.attackers import VIPERAttacker
from .base_attack import SuperAttacker
from tqdm import tqdm
from lib.encoder.IdentityBert import IdentityBert
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Set, Tuple
import re
import torch
from ...utilities.data_utils import find_all_token_indices

class MyViperAttacker(VIPERAttacker, SuperAttacker):
    _CallerPrefix = "VIPER Attacker"

    def transform(self, word: str) -> List[str]:
        candidates = []
        for idx in range(1, len(word) - 2):
            if word[idx].isalpha():
                s = self.substitute(word[idx])[0][0]
                candidate = copy.deepcopy(word)
                candidate = candidate.replace(word[idx], s)
                candidates.append(candidate)

        return candidates

    def process_entries(self, data: List[Dict], model: Optional[IdentityBert], tokenizer: Optional[AutoTokenizer]) -> List[Dict]:
        seen_texts = set()
        processed_data = []

        for entry in tqdm(data):
            orig_text = entry['text']
            if orig_text in seen_texts:
                continue  # Skip duplicates
            seen_texts.add(orig_text)

            annotations = entry.get('annotations', [])
            rephrased = {}
            if len(annotations) == 0:
                continue

            for annot in annotations:
                annot['text'] = re.sub(r"From:\s*", "", annot['text'], re.IGNORECASE) # very ad-hoc fix

            have_attacked = False
            for annot in annotations:
                entity = annot['text']
                entity_cls = annot['labels'][0]

                if have_attacked == False and entity_cls == "identity":
                    orig_tokens = tokenizer.tokenize(orig_text, truncation=True, add_special_tokens=True)
                    entity_tokens = tokenizer.tokenize(entity)

                    entity_range = find_all_token_indices(orig_tokens, entity_tokens)
                    if not len(entity_range):
                        continue
                    entity_starting_index = entity_range[0][0]
                    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(orig_tokens)])
                    orig_probs = torch.Tensor(model.get_prob(input_ids))
                    orig_prob = orig_probs[entity_starting_index][1] # 1 stands for the class I-Identity

                    candidates = self.transform(entity)

                    most_gap = 0
                    best_cand = None

                    for candidate in candidates:
                        new_text_copy = copy.deepcopy(orig_text)
                        new_text_copy = new_text_copy.replace(entity, candidate)

                        new_tokens = tokenizer.tokenize(new_text_copy, truncation=True, add_special_tokens=True)
                        candidate_tokens = tokenizer.tokenize(candidate)

                        entity_range = find_all_token_indices(new_tokens, candidate_tokens)
                        if not len(entity_range):
                            continue

                        new_entity_starting_index = entity_range[0][0]
                        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(new_tokens)])
                        new_probs = torch.Tensor(model.get_prob(input_ids))
                        new_prob = new_probs[new_entity_starting_index][1]  # 1 stands for the class I-Identity

                        gap = orig_prob - new_prob  # decrease in confidence
                        if gap > most_gap:
                            most_gap = gap
                            best_cand = candidate

                    if most_gap > 0:
                        orig_text = orig_text.replace(entity, best_cand)
                        rephrased[entity] = best_cand
                        annot['text'] = best_cand
                        have_attacked = True

            processed_data.append({
                "id": entry['Id'],
                "text": orig_text,
                "annotations": annotations,
                "metadata": entry.get("Path", ""),  # Add metadata field
                "rephrased_text": rephrased
            })

        return processed_data



