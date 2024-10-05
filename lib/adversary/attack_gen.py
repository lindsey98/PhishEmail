
import click
from ..encoder import IdentityBert
from transformers import AutoTokenizer
from .attack_utils import MySCPNAttacker, MyBAEAttacker, MyDeepWordBugAttacker, MyGPTAttacker, \
    MyViperAttacker, MyBartParaphraseAttacker, MyT5ParaphraseAttacker, MyConcatSentAttacker, MyTextFoolerAttacker
import os
import json
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

def pick_attacker(attacker_name, typo_type):
    if attacker_name == "bae":
        return MyBAEAttacker()
    elif attacker_name == "scpn":
        return MySCPNAttacker(templates=[
                        '( ROOT ( S ( VP ( VB ) ( NP ) ) ) ) EOP', # do xx
                        '( ROOT ( S ( VP ( TO ( VB ) ) ) ( , ) ( VP ( VB ) ) ) ) EOP', # to do xx, do xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( VP ( TO ( VB ) ( NP ) ) ) ) ) ) EOP', # do xx to do xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( SBAR ( IN ( S ( VP ( VB ) ( NP ) ) ) ) ) ) ) ) EOP', # do xx so that xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( VP ( IN ( TO ( VB ) ( ADVP ) ) ) ) ) ) ) EOP', # do xx in order to xx
                        '( ROOT ( S ( SBAR ( IN ( S ) ) ) ( , ) ( VP ( VB ) ) ) ) EOP', # if xx, do xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ( SBAR ( IN ( ADJP ) ) ) ) ) ) EOP', # do xx if xx
                        '( ROOT ( S ( VP ( VB ) ( NP ) ) ( , ) ( CC ) ( VP ( VB ) ( NP ) ) ) ) EOP' # do xx, then xx
                    ])
    elif attacker_name == "deepwordbug":
        return MyDeepWordBugAttacker(transform=typo_type, power=1)
    elif attacker_name == "gpt":
        return MyGPTAttacker()
    elif attacker_name == 'viper':
        return MyViperAttacker()
    elif attacker_name == 'bart':
        return MyBartParaphraseAttacker()
    elif attacker_name == 't5':
        return MyT5ParaphraseAttacker()
    elif attacker_name == 'concatsent':
        return MyConcatSentAttacker()
    elif attacker_name == 'textfooler':
        return MyTextFoolerAttacker()

@click.command()
@click.option('--attacker', required=True, type=click.Choice(['bae', 'scpn', 'deepwordbug', 'gpt', 'viper', 'bart', 't5', 'concatsent', 'textfooler'], case_sensitive=False), help="Specify the attacker type (e.g., 'bae')")
@click.option('--typo_type', help="Specify the typo type, only for the DeepWordBug attacking method", type=click.Choice(['repeat', 'delete', 'replace', 'switch'], case_sensitive=False))
def main(attacker, typo_type):
    # Initialize the transformation based on the attacker argument
    transformation = pick_attacker(attacker, typo_type)

    ### All data
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
    # get test split
    with open('./datasets/ner_training_augmented/test.json', 'r') as outfile:
        test_split = json.load(outfile)
    test_text = [x['text'] for x in test_split]
    filtered_data = []
    for d in data:
        if d['text'] in test_text:
            filtered_data.append(d)
    ###
    model = IdentityBert("checkpoints/identity_adversarial_training/checkpoint-435", aggregation_strategy="none")
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/identity_adversarial_training/checkpoint-435")

    # Process data with the selected transformation
    processed_data = transformation.process_entries(filtered_data, model, tokenizer)

    output_dir = f'./datasets/ner_training_adversarial/'
    os.makedirs(output_dir, exist_ok=True)
    if typo_type:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}_{typo_type}.json'), 'w') as outfile:
            json.dump(processed_data, outfile, indent=4)
    else:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}.json'), 'w') as outfile:
            json.dump(processed_data, outfile, indent=4)


if __name__ == '__main__':
    main()
