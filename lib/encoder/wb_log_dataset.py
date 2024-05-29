import json
import re
from nltk.tokenize import word_tokenize
import nltk
import os
import numpy as np
from sklearn.model_selection import train_test_split
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
nltk.download('punkt')

# Custom tokenizer function
def custom_tokenize(text):
    # Use regex to split the text into tokens, considering words and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens


# Function to find token indices for a given entity in the tokenized text (case-insensitive)
def find_all_token_indices(tokens, entity_text):
    entity_tokens = custom_tokenize(entity_text.lower())
    entity_len = len(entity_tokens)
    indices = []

    for i in range(len(tokens) - entity_len + 1):
        if [token.lower() for token in tokens[i:i + entity_len]] == entity_tokens:
            indices.append((i, i + entity_len))
    return indices


def tokenize_and_map(text, annotations, label_to_id):
    # Tokenize text using a custom tokenizer function
    tokens = custom_tokenize(text)

    # Initialize tags with "O"
    tags = [label_to_id["O"]] * len(tokens)

    for annotation in annotations:
        entity_text = annotation['text']
        label = annotation['labels'][0]

        all_indices = find_all_token_indices(tokens, entity_text)
        for start_index, end_index in all_indices:
            # Check if the range already has a tag other than "O"
            if all(tag == label_to_id["O"] for tag in tags[start_index:end_index]):
                # Label the found entity
                tags[start_index] = label_to_id["B-" + label]
                for i in range(start_index + 1, end_index):
                    tags[i] = label_to_id["I-" + label]

        if len(all_indices) == 0:
            raise

    return tokens, tags


if __name__ == '__main__':

    # Load the JSON file
    dataset_1_name = "spam_archive_2023"
    with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_all.json', 'r') as json_file:
        data_1 = json.load(json_file)

    dataset_2_name = "Nazario_2005"
    with open(f'./datasets/{dataset_2_name}_unique_annotation/annotated_all.json', 'r') as json_file:
        data_2 = json.load(json_file)

    label_list = [
        "O",
        "B-organization",
        "I-organization",
        "B-relation",
        "I-relation",
        "B-action",
        "I-action",
    ]

    # Create a mapping from labels to integers
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    # Process the data
    processed_data = []

    for entry in data_1:
        text = entry['text']
        annotations = entry['annotations']

        tokens, tags = tokenize_and_map(text, annotations, label_to_id)

        processed_data.append({
            "id": entry['Id'],
            "tokens": tokens,
            "ner_tags": tags,
            "metadata": entry["Path"]  # Add metadata field

        })

    for entry in data_2:
        text = entry['text']
        annotations = entry['annotations']

        tokens, tags = tokenize_and_map(text, annotations, label_to_id)

        processed_data.append({
            "id": entry['Id'],
            "tokens": tokens,
            "ner_tags": tags,
            "metadata": entry["Path"]  # Add metadata field
        })

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

    output_dir = f'./datasets/ner_training/'
    os.makedirs(output_dir, exist_ok=True)
    train_file = output_dir + 'train.json'
    test_file = output_dir + 'test.json'

    # Save the processed data
    # Save the training data
    with open(train_file, 'w') as outfile:
        json.dump(train_data, outfile, indent=4)

    # Save the testing data
    with open(test_file, 'w') as outfile:
        json.dump(test_data, outfile, indent=4)



