import copy
from OpenAttack.attackers import SCPNAttacker
import ssl
from typing import List
import random
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error
import json
import re
from lib.encoder.model_utils.log_dataset import tokenize_and_map
import os
from tqdm import tqdm
from .base_attack import SuperAttacker

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

    def _select_template(self):
        pick_one = random.choice(self.templates)
        return pick_one

    def get_transformations(self, tokens, ground_truth, tokenizer, cls_to_modify: List[int]=[5]):
        """
            Returns paraphrases of the input text, mapping named entities to the new ground truth.
        """
        sentences_to_be_rephrased, other_sentences, all_sentences, sentences_to_be_rephrased_entity_id = \
            self.sentence_segmentation(tokens=tokens, ground_truth=ground_truth, cls_to_modify=cls_to_modify)

        # Collect all other entities first with a context window (bigram or trigram) => we need to keep track
        entities = {}
        cls_to_modify_plus_one = set([x + 1 for x in cls_to_modify])

        for idx, (token, truth) in enumerate(zip(tokens, ground_truth)):
            if truth == 0 or truth in cls_to_modify or truth in cls_to_modify_plus_one:
                continue
            # Create context window (trigram with the token and neighboring words)
            context = ' '.join(tokens[max(0, idx - 1):min(len(tokens), idx + 2)]).lower()
            entities[context] = {  # track entities by their context
                "token": token,
                "truth": truth
            }

        # Faster lookup of sentence indices by using a dictionary
        sentence_index_map = {sent: idx for idx, sent in enumerate(all_sentences)}

        all_sentence_candidates = {}
        rephrased_entities = {}
        # Generate k candidates for each sentence
        for i, sent in enumerate(sentences_to_be_rephrased):
            if sent.strip():  # Skip empty sentences
                sent_entity_id = sentences_to_be_rephrased_entity_id[i]
                sent_candidate = self.gen_paraphrase(
                    sent=sent,
                    templates=[self._select_template()]
                )[0]  # generate only ONE candidate
                sent_index = sentence_index_map[sent]  # Get the sentence index efficiently
                # remove EOS token
                sent_candidate = sent_candidate.replace('EOS', '').strip()
                all_sentence_candidates[(sent, sent_index)] = sent_candidate  # record the rephrased versions

                rephrased_tokens = sent_candidate.split(' ')
                for j, token in enumerate(rephrased_tokens):
                    # Use the rephrased sentence's context (trigram) for entity matching
                    rephrased_context = ' '.join(rephrased_tokens[max(0, j - 1):min(len(rephrased_tokens), j + 2)]).lower()
                    rephrased_entities[rephrased_context] = {
                        "token": token,
                        "truth": sent_entity_id if j == 0 else sent_entity_id + 1
                    }

        # Efficient string concatenation
        candidate_paragraphs = []
        last_index = 0

        # Concatenate the aligned sentence candidates to form k different paragraphs
        for (sent, sent_index), cdn_sent in all_sentence_candidates.items():
            if last_index == len(all_sentences):  # reached the end
                candidate_paragraphs.append(cdn_sent)
            else:
                candidate_paragraphs.append(' '.join(all_sentences[last_index:sent_index]) + ' ' + cdn_sent)
            last_index = sent_index + 1

        if last_index < len(all_sentences):
            candidate_paragraphs.append(' '.join(all_sentences[last_index:]))

        candidate_paragraphs = ' '.join(candidate_paragraphs)

        # Remap the ground-truth
        out_texts = []
        cnd_tokens = candidate_paragraphs.split(" ") # the altered tokens
        final_cnd_tokens, cnd_truth = [], []
        used_entities = set()  # Track which entities have been used to avoid duplicates

        for idx, cnd_token in enumerate(cnd_tokens):
            # Use the context window (trigram) around the rephrased token for matching
            cnd_context = ' '.join(cnd_tokens[max(0, idx - 1):min(len(cnd_tokens), idx + 2)]).lower()
            left_cdn_context = ' '.join(cnd_tokens[max(idx-1, 0):idx+1]).lower() # including the token itself
            right_cdn_context = ' '.join(cnd_tokens[idx:min(len(cnd_tokens), idx + 2)]).lower()

            # Check both entities and rephrased entities in the context windows
            contexts_to_check = [cnd_context, left_cdn_context, right_cdn_context]

            token, truth = self.get_entity_from_context(contexts_to_check, entities, used_entities)
            if token is None:
                token, truth = self.get_entity_from_context(contexts_to_check, rephrased_entities, used_entities)

            # Append the matched token and truth, or the default values if no entity is found
            if token is not None:
                final_cnd_tokens.append(token)
                cnd_truth.append(truth)
            else:
                final_cnd_tokens.append(cnd_token)
                cnd_truth.append(0)

        final_text = " ".join(final_cnd_tokens)

        return {
            "perturbed_text": final_text,
            "perturbed_tokens": final_cnd_tokens,
            "perturbed_ground_truth": cnd_truth,
            "rephrased_dict": {k[0]:v for k, v in all_sentence_candidates.items()}
        }


    def process_entries(self, data, model, tokenizer):
        seen_texts = set()
        processed_data = []

        for entry in tqdm(data):
            tokens = entry['tokens']
            tags = entry['ner_tags']
            text = ' '.join(tokens)
            if text in seen_texts:
                continue  # Skip duplicates
            seen_texts.add(text)

            rephrased_result = self.get_transformations(tokens, tags, tokenizer=tokenizer, cls_to_modify=[5])
            rephrased_tokens = rephrased_result["perturbed_tokens"]
            rephrased_tags = rephrased_result["perturbed_ground_truth"]

            processed_data.append({
                "id": entry['id'],
                "tokens": rephrased_tokens,
                "ner_tags": rephrased_tags,
                'rephrased_dict': rephrased_result["rephrased_dict"],
                "metadata": entry["metadata"]  # Add metadata field
            })

        return processed_data

if __name__ == '__main__':
    label_list = [
        "O",
        "B-identity",
        "I-identity",
        "B-relation",
        "I-relation",
        "B-action",
        "I-action",
    ]
    label_to_id = {label: idx for idx, label in enumerate(label_list)}

    with open('./datasets/ner_training_augmented/test.json', 'r') as json_file:
        data = json.load(json_file)

    transformation = MySCPNAttacker(
            templates=[
                        '( ROOT ( S ( VP ( VB ) ( NP ) ) ) ) EOP', # do xx
                        '( ROOT ( S ( VP ( TO ( VB ) ) ) ( , ) ( VP ( VB ) ) ) ) EOP', # to do xx, do xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( VP ( TO ( VB ) ( NP ) ) ) ) ) ) EOP', # do xx to do xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( SBAR ( IN ( S ( VP ( VB ) ( NP ) ) ) ) ) ) ) ) EOP', # do xx so that xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( VP ( IN ( TO ( VB ) ( ADVP ) ) ) ) ) ) ) EOP', # do xx in order to xx
                        '( ROOT ( S ( SBAR ( IN ( S ) ) ) ( , ) ( VP ( VB ) ) ) ) EOP', # if xx, do xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( SBAR ( IN ( ADJP ) ) ) ) ) ) EOP', # do xx if xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ) ( , ) ( CC ) ( VP ( VB ) ( NP ) ) ) ) EOP' # do xx, then xx
                    ]
    )
    processed_data = []
    seen_texts = set()
    transformation.process_entries(data, None, None)

    output_dir = f'./datasets/ner_training_adversarial/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'adversarial_rephrase_scpn.json'), 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)
