import copy
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
from tqdm import tqdm
from .base_attack import SuperAttacker
from ..utils import to_sentence_case
from transformers import AutoTokenizer, BertForMaskedLM
from lib.encoder.IdentityBert import IdentityBert
from typing import List, Dict, Optional
import re

class MyConcatSentAttacker(SuperAttacker):

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

            sentences_with_delimiters = re.findall(r'([^.!?]+)([.!?]*)', text)
            sentences = []
            delimiters = []
            for item in sentences_with_delimiters:
                sent, delimit = item
                sentences.append(sent)
                delimiters.append(delimit)

            have_attacked = False
            for annot in annotations:
                entity_cls = annot['labels'][0]
                entity = annot['text']

                if have_attacked == False and entity_cls == 'action':
                    sent_index = -1
                    for idx, sentence in enumerate(sentences):
                        if entity in sentence:
                            sent_index = idx
                            break
                    if sent_index != -1:
                        concat_sent = to_sentence_case(sentences[sent_index-1] + ' ' + entity)
                        rephrased[entity] = concat_sent
                        # Combine the entity with its previous sentence
                        sentences[sent_index-1] = concat_sent
                        # Remove the entity in its original sentence
                        sentences[sent_index] = to_sentence_case(sentences[sent_index].replace(entity, ""))

                        # Apply case preservation and append delimiters back
                        corrected_sentences = []
                        for correct_sent, delimit in zip(sentences, delimiters):
                            corrected_sentences.append(correct_sent + delimit)

                        text = " ".join(corrected_sentences)

                        have_attacked = True

            processed_data.append({
                "id": entry['Id'],
                "text": text,
                "annotations": annotations,
                "metadata": entry.get("Path", ""),  # Add metadata field
                "rephrased_text": rephrased
            })

        return processed_data

