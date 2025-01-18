import copy
import json
from ...adversary.attack_utils import MyDeepWordBugAttacker
from tqdm import tqdm
import random
from transformers import AutoTokenizer
from typing import *
import re
import os

def get_entity_spans(text: str, entities: List[str]) -> List[Tuple[int, int]]:
    """
    Finds the start and end character indices of each entity in the text.
    Assumes that entities do not overlap and appear in the text as whole words.
    """
    spans = []
    for entity in entities:
        # Use regex to find whole word matches to avoid partial matches
        for match in re.finditer(r'\b{}\b'.format(re.escape(entity)), text):
            start, end = match.start(), match.end()
            spans.append((start, end))
    return spans


def process_adversarial_training_entries(data: List[Dict]):
    # Initialize lists to store augmented data
    augmented_data = []

    # Process each entry in the training data
    for entry in tqdm(data, desc="Augmenting Data"):
        tokens, tags = entry['tokens'], entry['ner_tags']

        # Filter by language if necessary
        if entry.get("lang") and entry.get("lang") != "en":
            continue

        # Extract entities
        entity_pos = []
        entities = []
        start_idx = 0
        while start_idx < len(tags):
            if tags[start_idx] == label_to_id['B-identity']:
                end_idx = start_idx + 1
                while end_idx < len(tags) and tags[end_idx] == label_to_id['I-identity']:
                    end_idx += 1
                entity = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx])
                entities.append(entity)
                entity_pos.append((start_idx, end_idx))
                start_idx = end_idx
            else:
                start_idx += 1

        # Inject typos into entities
        offset = 0
        new_tokens = copy.deepcopy(tokens)
        new_tags = copy.deepcopy(tags)

        for it, entity in enumerate(entities):
            perturb_type = random.choice(['repeat', 'delete', 'switch', 'replace'])
            perturbed_choices = attacker.transform(perturb_type, entity)
            if perturbed_choices:
                perturbed_entity = random.choice(perturbed_choices)
                perturb_entity_tokens = tokenizer.tokenize(perturbed_entity)
                orig_start_idx, orig_end_idx = entity_pos[it]

                orig_entity_token_counts = orig_end_idx - orig_start_idx
                perturb_entity_token_counts = len(perturb_entity_tokens)

                new_tokens = new_tokens[:orig_start_idx+offset] + perturb_entity_tokens + new_tokens[orig_end_idx+offset:]
                new_tags = new_tags[:orig_start_idx+offset] + [label_to_id['B-identity']] + [label_to_id['I-identity']]*(perturb_entity_token_counts-1) + new_tags[orig_end_idx+offset:]

                offset += perturb_entity_token_counts - orig_entity_token_counts

        augmented_data.append({
            "id": entry['id'],
            "tokens": new_tokens,
            "ner_tags": new_tags,
            "metadata": entry["metadata"]  # Add metadata field
        })

    return augmented_data


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
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    output_dir = f'./datasets/ner_training_augmented/'
    train_file = output_dir + 'train.json'
    test_file = output_dir + 'test.json'

    # Save the processed data
    # Save the training data
    with open(train_file, 'rb') as outfile:
        train_data_raw = json.load(outfile)

    # Save the testing data
    with open(test_file, 'rb') as outfile:
        test_data_raw = json.load(outfile)

    attacker = MyDeepWordBugAttacker()
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased", use_fast=True)

    train_data = process_adversarial_training_entries(train_data_raw)
    test_data = process_adversarial_training_entries(test_data_raw)

    output_dir = './datasets/ner_adversarial_training/'
    os.makedirs(output_dir, exist_ok=True)
    train_file = output_dir + 'train.json'
    test_file = output_dir + 'test.json'

    # Save the training data
    with open(train_file, 'w') as outfile:
        json.dump(train_data, outfile, indent=4)

    # Save the testing data
    with open(test_file, 'w') as outfile:
        json.dump(test_data, outfile, indent=4)






