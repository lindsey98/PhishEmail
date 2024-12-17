import copy
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
from tqdm import tqdm
from .base_attack import SuperAttacker
from ...utilities.data_utils import to_sentence_case
from transformers import AutoTokenizer, BertForMaskedLM
from ...encoder.IdentityBert import IdentityBert
from typing import List, Dict, Optional
import re

class MyConcatSentAttacker(SuperAttacker):
    _CallerPrefix = "ConcatSentence Attacker"
    _Help = "Concatenate the entity with its preceding sentence"

    def process_entries(self, data: List[Dict], model: Optional[IdentityBert], tokenizer: Optional[AutoTokenizer]) -> List[Dict]:
        '''
        Conduct attack on a list of data, each data is a dict with 'text' and 'annotations'
        :param data:
        :param model:
        :param tokenizer:
        :return:
        '''
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
            sentences, delimiters = zip(*sentences_with_delimiters)
            sentences = list(sentences)

            have_attacked = False
            for annot in annotations:
                entity_cls = annot['labels'][0]
                entity = annot['text']

                if have_attacked == False and entity_cls == 'action':
                    for idx, sentence in enumerate(sentences):
                        if entity in sentence:
                            concat_sent = to_sentence_case(sentences[idx-1] + ' ' + entity)
                            rephrased[entity] = concat_sent
                            # Combine the entity with its previous sentence
                            sentences[idx-1] = concat_sent
                            # Remove the entity in its original sentence
                            sentences[idx] = to_sentence_case(sentences[idx].replace(entity, ""))

                            # Apply case preservation and append delimiters back
                            text = " ".join(to_sentence_case(sent + delim) for sent, delim in zip(sentences, delimiters))

                            have_attacked = True
                            break

            processed_data.append({
                "id": entry['Id'],
                "text": text,
                "annotations": annotations,
                "metadata": entry.get("Path", ""),  # Add metadata field
                "rephrased_text": rephrased
            })

        return processed_data

