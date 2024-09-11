from OpenAttack.attackers import DeepWordBugAttacker
from .base_attack import SuperAttacker
from lib.adversary.utils import *
from tqdm import tqdm
import numpy as np
from lib.encoder.IdentityBert import IdentityBert
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Set, Tuple

class MyDeepWordBugAttacker(DeepWordBugAttacker, SuperAttacker):


    def transform(self, type_: str, word: str) -> str:
        if type_ == 'repeat':
            typoed_text = repeat_char(word)
        elif type_ == 'delete':
            typoed_text = delete_char(word)
        elif type_ == 'switch':
            typoed_text = switch_chars(word)
        elif type_ == 'replace':
            typoed_text = replace_with_similar(word)
        else:
            raise NotImplementedError

        return typoed_text

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

            have_attacked = False
            for annot in annotations:
                entity = annot['text']
                entity_cls = annot['labels'][0]

                if have_attacked == False and entity_cls == "identity":
                    typoed_entity = self.transform(self.transformer, entity)
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



