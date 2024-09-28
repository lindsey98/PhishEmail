import copy
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
from tqdm import tqdm
from .base_attack import SuperAttacker
from ...utilities.data_utils import to_sentence_case
from transformers import AutoTokenizer, BertForMaskedLM
from lib.encoder.IdentityBert import IdentityBert
from typing import List, Dict, Optional
import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

class MyBartParaphraseAttacker(SuperAttacker):
    _CallerPrefix = "BART Paraphrasing Attacker"

    def __init__(self):
        super().__init__()  # Call the superclass constructor
        self.model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

    def gen_paraphrase(self, sent: str):
        batch = self.tokenizer(sent, return_tensors='pt', truncation=True)
        batch = batch.to(self.device)
        generated_ids = self.model.generate(batch['input_ids'], max_new_tokens=512)
        generated_sentence = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_sentence

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

                if have_attacked == False and entity_cls == 'action':
                    try:
                        rephrased_text = self.gen_paraphrase(
                            sent=entity,
                        )[0]  # generate only ONE candidate
                    except KeyError:
                        rephrased_text = entity # no rephrasing happens

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

