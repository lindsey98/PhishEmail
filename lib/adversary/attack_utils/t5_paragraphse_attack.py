import copy
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
from tqdm import tqdm
from .base_attack import SuperAttacker
from lib.encoder.IdentityBert import IdentityBert
from typing import List, Dict, Optional
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class MyT5ParaphraseAttacker(SuperAttacker):

    def __init__(self):
        super().__init__()  # Call the superclass constructor
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

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

    def gen_paraphrase(self, sent: str):
        text = "paraphrase: " + sent + " </s>"

        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")

        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=200,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=5
        )

        generated_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
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

                if have_attacked==False and entity_cls == 'action':
                    try:
                        rephrased_text = self.gen_paraphrase(
                            sent=entity,
                        )  # generate only ONE candidate
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

