from .base_attack import SuperAttacker
from tqdm import tqdm
import os
from lib.encoder.IdentityBert import IdentityBert
from ...utilities.gpt_utils import chat_completion
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Set, Tuple

class MyGPTAttacker(SuperAttacker):
    _CallerPrefix = "GPT Paraphrasing Attacker"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the superclass constructor
        from openai import OpenAI
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.proxy = os.getenv("http_proxy")  # proxy
        self.client = OpenAI()

    def gen_paraphrase(self, sent: str) -> str:
        rephrased_text = chat_completion(self.client,
                                   query=sent,
                                   context="""Paraphrase the following phrase with a special focus on diversifying the sentence patterns. 
                                    Only output the rephrased text with no extra explanations.""")
        return rephrased_text

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
                    rephrased_text = self.gen_paraphrase(
                        sent=entity,
                    )  # generate only ONE candidate
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


