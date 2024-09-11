import copy
from OpenAttack.attackers import SCPNAttacker
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
from tqdm import tqdm
from .base_attack import SuperAttacker
from transformers import AutoTokenizer, BertForMaskedLM
from lib.encoder.IdentityBert import IdentityBert
from typing import List, Dict, Optional
import re

class MySCPNAttacker(SCPNAttacker, SuperAttacker):
    """
        Generates paraphrases of the input sentence,
        re-mapping the original ground truth and making
        sure named entities are preserved.

        Based on (as implemented in OpenAttack):

        Adversarial Example Generation with Syntactically Controlled
        Paraphrase Networks.

        Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer.
        NAACL-HLT 2018.

        `[pdf] <https://www.aclweb.org/anthology/N18-1170.pdf>`__
        `[code] <https://github.com/miyyer/scpn>`__
    """

    def _select_template(self) -> str:
        pick_one = random.choice(self.templates)
        return pick_one

    @staticmethod
    def to_sentence_case(text):
        """
        Converts the given text into sentence case.
        Capitalizes the first word and lowers the rest, except for proper nouns or acronyms.
        """
        # Tokenize based on whitespace or specific parse tokens
        tokens = re.split(r'(\s+|\(|\)|,|\.|\;)', text)  # split but keep delimiters

        # Capitalize first VB (or other sentence-starting token)
        capitalized = False
        new_tokens = []
        for token in tokens:
            if not capitalized and token.isalpha():
                new_tokens.append(token.capitalize())  # capitalize first alphabetic word
                capitalized = True
            else:
                new_tokens.append(token.lower())  # lowercase the rest
        return ''.join(new_tokens)

    def process_entries(self, data: List[Dict], model: Optional[IdentityBert], tokenizer: Optional[AutoTokenizer]) -> List[Dict]:
        processed_data = []
        seen_texts = set()

        for entry in tqdm(data):
            text = entry['text']
            if text in seen_texts:
                continue  # Skip duplicates
            seen_texts.add(text)

            annotations = entry.get('annotations', [])
            rephrased = {}
            if len(annotations) == 0:
                continue

            have_attacked = False
            for annot in annotations:
                entity_cls = annot['labels'][0]
                entity = annot['text']

                if have_attacked==False and entity_cls == 'action':
                    try:
                        rephrased_text = self.gen_paraphrase(
                            sent=entity,
                            templates=[self._select_template()]
                        )[0]  # generate only ONE candidate
                    except KeyError:
                        rephrased_text = entity
                    rephrased_text = self.to_sentence_case(rephrased_text)
                    text = text.replace(entity, rephrased_text)
                    rephrased[entity] = rephrased_text
                    annot['text'] = rephrased_text
                    have_attacked = True

            processed_data.append({
                "id": entry['Id'],
                "text": text,
                "annotations": annotations,
                "metadata": entry.get("Path", ""),  # Add metadata field
                "rephrased_text": rephrased
            })

        return processed_data

