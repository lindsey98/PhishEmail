import copy
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
from tqdm import tqdm
from .base_attack import SuperAttacker
from ...utilities.data_utils import to_sentence_case, match_case
from lib.encoder.IdentityBert import IdentityBert
from typing import List, Dict, Optional
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from OpenAttack.attackers import TextFoolerAttacker
import nltk
from nltk.tokenize import word_tokenize
from nltk import StanfordTagger
import string
import spacy

nlp = spacy.load('en_core_web_sm')


class MyTextFoolerAttacker(TextFoolerAttacker, SuperAttacker):
    _CallerPrefix = "TextFooler Attacker"

    def tokenize_pos_tag(self, sent: str):
        # Tokenize the sentence into words (for POS tagging)
        pos_tags = []
        doc = nlp(sent)
        for token in doc:
            pos_tags.append((token.text, token.pos_))
        return pos_tags

    def get_verb_token_indices(self, sent: str):
        pos_tags = self.tokenize_pos_tag(sent)

        # Identifying indices of subwords that correspond to verbs
        verb_subword_indices = []
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB') or tag.startswith('VERB'):  # Checking if the POS tag is for a verb
                verb_subword_indices.append(i)

        return verb_subword_indices

    def gen_paraphrase(self, sent: str):
        words = [token.text for token in nlp(sent)]
        verb_indices = self.get_verb_token_indices(sent) # FIxme: Note that I am using the word tokenizer

        for idx in verb_indices:
            candidates = self.substitute(words[idx], "verb")
            if len(candidates): # take 1 and only 1 candidate
                words[idx] = candidates[0][0]

        sent = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()
        return sent

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
                        )  # generate only ONE candidate
                    except KeyError:
                        rephrased_text = entity
                    rephrased_text = to_sentence_case(rephrased_text)
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

