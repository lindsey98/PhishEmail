
import click
from lib.encoder.IdentityBert import IdentityBert
from transformers import AutoTokenizer
from lib.adversary.attack_utils import MySCPNAttacker, MyBAEAttacker, MyDeepWordBugAttacker, MyGPTAttacker
import os
import json
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
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

@click.command()
@click.option('--attacker', required=True, help="Specify the attacker type (e.g., 'bae')")
@click.option('--typo_type', type=str, help="Specify the typo type, only for the deepwordbug attack")
def main(attacker, typo_type):
    # Initialize the transformation based on the attacker argument
    transformation = pick_attacker(attacker, typo_type)

    # Your processing logic
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

    model = IdentityBert("./checkpoints/identity-model", aggregation_strategy="none")
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/identity-model")

    # Process data with the selected transformation
    processed_data = transformation.process_entries(data, model, tokenizer)

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
