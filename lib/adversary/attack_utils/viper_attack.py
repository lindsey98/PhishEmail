import copy
import random

from OpenAttack.attackers import VIPERAttacker
from .base_attack import SuperAttacker
from lib.adversary.utils import *
from tqdm import tqdm
import numpy as np
from lib.encoder.IdentityBert import IdentityBert
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Set, Tuple
import re

class MyViperAttacker(VIPERAttacker):

    def transform(self, word: str):
        random_pos = random.choice(range(len(word)))
        s = self.substitute(word[random_pos])[0][0]
        new_word = copy.deepcopy(word)
        new_word = new_word.replace(word[random_pos], s)
        return new_word

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
                    typoed_entity = self.transform(entity)
                    orig_text = orig_text.replace(entity, typoed_entity)
                    rephrased[entity] = typoed_entity
                    annot['text'] = typoed_entity
                    have_attacked = True

            processed_data.append({
                "id": entry['Id'],
                "text": orig_text,
                "annotations": annotations,
                "metadata": entry.get("Path", ""),  # Add metadata field
                "rephrased_text": rephrased
            })

        return processed_data



