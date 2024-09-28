import json
import re
import nltk
import os
from sklearn.model_selection import train_test_split
from ...utilities.data_utils import process_entries
from transformers import AutoTokenizer
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
nltk.download('punkt')

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
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

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

    data = data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7
    processed_data = process_entries(data=data, tokenizer=tokenizer, label_to_id=label_to_id)

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




