import json
import re
import nltk
import os
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
            if all(tag == label_to_id["O"] for tag in tags[start_index:end_index]) or label == "relation":
                # Label the found entity
                tags[start_index] = label_to_id["B-" + label]
                for i in range(start_index + 1, end_index):
                    tags[i] = label_to_id["I-" + label]

        if len(all_indices) == 0:
            # raise
            continue
    return tokens, tags


def process_entries(data):
    global processed_data, seen_texts
    for entry in data:
        text = entry['text']
        if text in seen_texts:
            continue  # Skip duplicates
        seen_texts.add(text)

        annotations = entry.get('annotations', [])
        if len(annotations) == 0:
            continue
        for annot in annotations:
            if annot['labels'][0] == 'identity':
                if annot['text'].startswith('From:'):
                    annot['text'] = re.sub(r"From:\s*", "", annot['text'])
        tokens, tags = tokenize_and_map(text, annotations, label_to_id)

        processed_data.append({
            "id": entry['Id'],
            "tokens": tokens,
            "ner_tags": tags,
            "metadata": entry.get("Path", "")  # Add metadata field
        })




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
    #
    # # Create a mapping from labels to integers
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    # Load the JSON file
    dataset_1_name = "spam_archive_2023"
    with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_all_augmented_50.json', 'r') as json_file:
        data_1 = json.load(json_file)

    dataset_2_name = "Nazario_2005"
    with open(f'./datasets/{dataset_2_name}_unique_annotation/annotated_all_augmented_50.json', 'r') as json_file:
        data_2 = json.load(json_file)

    dataset_3_name = "annotated_datasets_from_paul"
    with open(f'./datasets/{dataset_3_name}/annotated_all_augmented_50.json', 'r') as json_file:
        data_3 = json.load(json_file)

    dataset_4_name = "spam_archive_2023"
    with open(f'./datasets/{dataset_4_name}_unique_annotation/annotated_jiafan_augmented_50.json', 'r') as json_file:
        data_4 = json.load(json_file)

    dataset_5_name = "augmented_internal_emails"
    with open(f'./datasets/{dataset_5_name}/annotate_all.json', 'r') as json_file:
        data_5 = json.load(json_file)

    with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_internal.json', 'r') as json_file:
        data_6 = json.load(json_file)

    with open(f'./datasets/Enron_2015_unique_annotation/annotated_internal.json', 'r') as json_file:
        data_7 = json.load(json_file)

    # Process the data
    processed_data = []
    seen_texts = set()
    last_length = 0
    process_entries(data_1)
    print(f"Length of dataset {dataset_1_name} = {len(processed_data) - last_length}")
    last_length = len(processed_data)
    process_entries(data_2)
    print(f"Length of dataset {dataset_2_name} = {len(processed_data) - last_length}")
    last_length = len(processed_data)
    process_entries(data_3)
    print(f"Length of dataset {dataset_3_name} = {len(processed_data) - last_length}")
    last_length = len(processed_data)
    process_entries(data_4)
    print(f"Length of dataset {dataset_4_name} = {len(processed_data) - last_length}")
    last_length = len(processed_data)
    process_entries(data_5)
    print(f"Length of dataset {dataset_5_name} = {len(processed_data) - last_length}")

    last_length = len(processed_data)
    process_entries(data_6)
    print(f"Length of dataset Internal spamassassin = {len(processed_data) - last_length}")
    last_length = len(processed_data)
    process_entries(data_7)
    print(f"Length of dataset Internal Enron = {len(processed_data) - last_length}")

    # # Split the data into training and testing sets
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

    # output_dir = f'./datasets/ner_training/'
    output_dir = f'./datasets/ner_training_augmented/'
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




