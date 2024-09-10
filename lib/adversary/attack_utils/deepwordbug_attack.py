import copy
import random
from typing import List
from OpenAttack.attackers import DeepWordBugAttacker
from .base_attack import SuperAttacker
from lib.adversary.utils import *
from tqdm import tqdm
import json
import os
import numpy as np
from transformers import AutoTokenizer

class MyDeepWordBugAttacker(DeepWordBugAttacker, SuperAttacker):

    def group_consecutive_subsequences(self, interested_indices):
        if not interested_indices:
            return []

        subsequences = []
        current_subseq = [interested_indices[0]]

        # Group consecutive numbers into subsequences
        for i in range(1, len(interested_indices)):
            if interested_indices[i] == interested_indices[i - 1] + 1:
                current_subseq.append(interested_indices[i])
            else:
                subsequences.append(current_subseq)
                current_subseq = [interested_indices[i]]

        subsequences.append(current_subseq)  # Append the last subsequence

        return subsequences

    def transform(self, type_, word):
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

    def get_transformations(self, tokens, ground_truth, tokenizer, cls_to_modify: List[int]=[1, 2]):

        adv_tokens = []
        adv_ground_truth = []
        interested_indices = [i for i in range(len(ground_truth)) if (ground_truth[i] in cls_to_modify)]
        grouped_indices = self.group_consecutive_subsequences(interested_indices)

        typoed_tokens = {}
        prev_idx = 0
        for group_idx in grouped_indices:
            if len(typoed_tokens.keys()) >= self.power: # limit the number of typos created
                break
            group_start = group_idx[0]
            group_end = group_idx[-1]

            group_ground_truth = ground_truth[group_start]
            adv_tokens.extend(tokens[prev_idx:group_start])
            adv_ground_truth.extend(ground_truth[prev_idx:group_start])

            subseq = np.array(tokens)[group_idx]
            subseq = ' '.join(subseq)

            adv_subseq = self.transform(self.transformer, subseq)
            adv_subseq_tokens = tokenizer.tokenize(adv_subseq)
            adv_subseq_ground_truth = [group_ground_truth] + [group_ground_truth+1]*(len(adv_subseq_tokens)-1)

            adv_tokens.extend(adv_subseq_tokens)
            adv_ground_truth.extend(adv_subseq_ground_truth)
            prev_idx = group_end + 1
            typoed_tokens[subseq] = tokenizer.convert_tokens_to_string(adv_subseq_tokens)

        return {
            "perturbed_tokens": adv_tokens,
            "perturbed_ground_truth": ground_truth,
            "rephrased_dict": typoed_tokens
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

            rephrased_result = self.get_transformations(tokens=tokens, ground_truth=tags,
                                                        tokenizer=tokenizer,
                                                        cls_to_modify=[1, 2])
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
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/identity-model")

    for typo_type in ['delete', 'switch', 'replace', 'repeat']:
        with open('./datasets/ner_training_augmented/test.json', 'r') as json_file:
            data = json.load(json_file)

        transformation = MyDeepWordBugAttacker(transform=typo_type, power=1)
        processed_data = []
        seen_texts = set()
        transformation.process_entries(data, None, tokenizer)

        output_dir = f'./datasets/ner_training_adversarial/'
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'adversarial_rephrase_deepwordbug_{typo_type}.json'), 'w') as outfile:
            json.dump(processed_data, outfile, indent=4)

