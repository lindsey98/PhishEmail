from transformers import pipeline, AutoTokenizer
from typing import Tuple, Set
import time
import torch

class IdentityBert():
    def __init__(self, identity_checkpoint_path: str) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classifier_pipeline = pipeline("ner", model=identity_checkpoint_path,
                                            device=self.device, aggregation_strategy="simple")
        self.tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)

    def _tokenize(self, text):
        tokens = self.tokenizer(text, return_offsets_mapping=True, truncation=True)
        tokenized_email = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        tokenized_email_str = self.tokenizer.convert_tokens_to_string(tokenized_email)
        return tokenized_email_str

    @torch.inference_mode()
    def __call__(self, raw_text: str) -> Tuple[Set[str], Set[str], Set[str], float]:
        tokenized_email_str = self._tokenize(raw_text)
        start_time = time.time()
        entities = self.classifier_pipeline(tokenized_email_str)
        runtime = time.time() - start_time

        identities = set()
        relations = set()
        actions = set()

        for ent in entities:
            ent_label = ent['entity_group']
            ent_text = ent['word']
            if ent_label == 'identity':
                identities.add(ent_text)
            elif ent_label == 'relation':
                relations.add(ent_text)
            else:
                actions.add(ent_text)

        return identities, actions, relations, runtime
