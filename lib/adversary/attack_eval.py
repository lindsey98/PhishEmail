import copy
import os
import csv
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer
from lib.encoder.IdentityBert import IdentityBert
import pandas as pd
import langdetect
import unicodedata
import difflib
from lib.reference_db.IdentityMatcher import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
from lib.adversary.defence_utils import SpellFixer
import click
import json
import re

class MyAttackEvaluator():
    def __init__(self):
        self.corrector = SpellFixer()

    @staticmethod
    def remove_trailing_digits(s):
        # Use regex to replace trailing _digits with an empty string
        return re.sub(r'_\d+$', '', s)

    @staticmethod
    def check_lang(parsed_email):
        try:
            detected_langs = langdetect.detect_langs(parsed_email)
        except langdetect.lang_detect_exception.LangDetectException:
            return False
        is_in_english = True
        for lang in detected_langs:
            if lang.lang != 'en':
                is_in_english = False
                break

        return is_in_english

    def model_pred(self, email: str, tokenizer: AutoTokenizer, model: IdentityBert):
        tokens = tokenizer(email, return_offsets_mapping=True, truncation=True)
        tokenized_email = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        tokenized_email_str = tokenizer.convert_tokens_to_string(tokenized_email)
        identities, actions, relations, urls_after_actions, identity_recog_runtime = model(tokenized_email_str)

        return identities, actions, identity_recog_runtime

    def results_collector(self, dataset: List[Dict], tokenizer: AutoTokenizer, model: IdentityBert, result_csv_path: str, with_defence: bool):

        if not os.path.exists(result_csv_path):
            with open(result_csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['email_file_path',
                                 'attacked_email',
                                 'identities_after_adv',
                                 'actions_after_adv',
                                 'identities_before_adv',
                                 'actions_before_adv',
                                 'rephrased_dict',
                                 'pred_time'])

        for it in tqdm(range(len(dataset))):
            entry = dataset[it]
            email_file_path = entry["metadata"]
            if email_file_path in [x.split(',')[0] for x in open(result_csv_path).readlines()]:
                continue

            attacked_email = entry["text"]
            rephrased_dict = entry['rephrased_text']

            # fixme: recover to the email before attack
            unattacked_email = copy.deepcopy(attacked_email)
            for k, v in entry["rephrased_text"].items():
                unattacked_email = unattacked_email.replace(v, k)

            if with_defence:
                attacked_email = self.corrector(attacked_email)

            identities_after_adv, actions_after_adv, identity_recog_runtime_after_adv = self.model_pred(attacked_email, tokenizer, model)
            identities_before_adv, actions_before_adv, _ = self.model_pred(unattacked_email, tokenizer, model)

            # Append the new row to the CSV file
            with open(result_csv_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([email_file_path,
                                 attacked_email,
                                 identities_after_adv,
                                 actions_after_adv,
                                 identities_before_adv,
                                 actions_before_adv,
                                 rephrased_dict,
                                 identity_recog_runtime_after_adv
                                 ])

    def metrics_collector(self, result_csv_path: str, matcher_cls: IdentityMatcher, embed_model: CharacterBERT, cls_to_check: str):
        result_df = pd.read_csv(result_csv_path)

        before_adv_reported_ct = 0
        after_adv_reported_ct = 0
        matching_ct = 0
        total_ct = 0
        seen_email_files = set()

        for it, row in tqdm(result_df.iterrows()):
            email_file_path = row['email_file_path']
            parsed_email = row["attacked_email"]
            rephrased_dict = row["rephrased_dict"]
            identities_after_adv = row["identities_after_adv"]
            actions_after_adv = row["actions_after_adv"]
            identities_before_adv = row["identities_before_adv"]
            actions_before_adv = row["actions_before_adv"]

            if email_file_path in seen_email_files:
                continue
            else:
                seen_email_files.add(email_file_path)

            if not self.check_lang(parsed_email):
                continue

            if len(eval(rephrased_dict).values()):
                total_ct += 1

                if cls_to_check == "identity":
                    # expected identities
                    expected_identities_orig = [self.remove_trailing_digits(unicodedata.normalize("NFKC", x.lower())) for x in list(eval(rephrased_dict).keys())]
                    expected_identities_rephrased = [unicodedata.normalize("NFKC", x.lower()) for x in list(eval(rephrased_dict).values())]

                    # build an index base for those expected identities
                    orig_index_db = BaseFaissIPRetriever(tags=expected_identities_orig, embed_model=embed_model)
                    ## matching acc
                    for id in expected_identities_rephrased:
                        _, closest_match = matcher_cls.find_closest_match(query=id,
                                                                          value_index_db=orig_index_db)
                        if closest_match:
                            matching_ct += 1
                            break
                    orig_index_db.add(None, expected_identities_rephrased)

                    ### after adv
                    predicted_identities = [x.lower() for x in list(eval(identities_after_adv))]
                    for pred_identity in predicted_identities:
                        _, closest_match = matcher_cls.find_closest_match(query=pred_identity,
                                                                          value_index_db=orig_index_db)
                        if closest_match:
                            after_adv_reported_ct += 1
                            break

                    ### baseline
                    predicted_identities = [x.lower() for x in list(eval(identities_before_adv))]
                    for pred_identity in predicted_identities:
                        _, closest_match = matcher_cls.find_closest_match(query=pred_identity,
                                                                          value_index_db=orig_index_db)
                        if closest_match:
                            before_adv_reported_ct += 1
                            break

                elif cls_to_check == "action":
                    matching_ct += 1
                    ### after rephrasing
                    rephrased_action = [x.lower() for x in list(eval(rephrased_dict).values())] + [x.lower() for x in list(eval(rephrased_dict).keys())]

                    predicted_action = [x.lower() for x in list(eval(actions_after_adv))]
                    for expected_action in rephrased_action:
                        closest_match = difflib.get_close_matches(expected_action, predicted_action, n=1, cutoff=0.3)
                        if closest_match:
                            after_adv_reported_ct += 1
                            break
                    ### baseline
                    predicted_action = [x.lower() for x in list(eval(actions_before_adv))]
                    for expected_action in rephrased_action:
                        closest_match = difflib.get_close_matches(expected_action, predicted_action, n=1, cutoff=0.3)
                        if closest_match:
                            before_adv_reported_ct += 1
                            break
                else:
                    raise NotImplementedError()

        return after_adv_reported_ct, before_adv_reported_ct, matching_ct, total_ct


@click.command()
@click.option('--attacker', required=True, type=click.Choice(['bae', 'deepwordbug', 'scpn', 'gpt'], case_sensitive=False), help="Specify the attacker type (e.g., 'bae')")
@click.option('--cls_to_attack', required=True, type=str, help="Attack which NER class")
@click.option('--typo_type', help="Specify the typo type, only for the DeepWordBug attacking method", type=click.Choice(['repeat', 'delete', 'replace', 'switch'], case_sensitive=False))
@click.option('--eval_only', help="Eval only or Attack+Eval", is_flag=True, show_default=True, default=False)
@click.option('--defence', help="With defence or not", is_flag=True, show_default=True, default=False)
def main(attacker, cls_to_attack, typo_type, eval_only, defence):
    evaluator = MyAttackEvaluator()
    embed_model = CharacterBERT()
    matcher_cls = IdentityMatcher(brand_index_db=None,
                                  internal_relation_index_db=None,
                                  embed_model=embed_model,
                                  brand_domain_map_path=None,
                                  knowledge_base_expansion=False,
                                  gpt_client=None, gpt_assistant=None,
                                  check_action=True,
                                  threshold=0.9)

    output_dir = f'./datasets/ner_training_adversarial/'
    if typo_type:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}_{typo_type}.json'), 'r') as json_file:
            dataset = json.load(json_file)
        result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_{typo_type}.csv'
    else:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}.json'), 'r') as json_file:
            dataset = json.load(json_file)
        result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}.csv'

    if defence:
        result_csv_path = result_csv_path.replace(".csv", "_with_defence.csv")

    model = IdentityBert("./checkpoints/identity-model", aggregation_strategy="simple")
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/identity-model")

    if eval_only:
        if not os.path.exists(result_csv_path):
            raise FileNotFoundError(f"The {result_csv_path} does not exist")
        after_adv_reported_ct, before_adv_reported_ct, matching_ct, total_ct = evaluator.metrics_collector(result_csv_path=result_csv_path, matcher_cls=matcher_cls,
                                    embed_model=embed_model, cls_to_check=cls_to_attack)

    else:
        evaluator.results_collector(dataset=dataset, tokenizer=tokenizer, model=model, result_csv_path=result_csv_path, with_defence=defence)
        after_adv_reported_ct, before_adv_reported_ct, matching_ct, total_ct = evaluator.metrics_collector(result_csv_path=result_csv_path, matcher_cls=matcher_cls,
                                                            embed_model=embed_model, cls_to_check=cls_to_attack)

    print(f"Baseline NER detection rate = {before_adv_reported_ct}/{total_ct} = {before_adv_reported_ct / total_ct}")
    print(f"Attacker = {attacker} {typo_type}, With defence = {defence} \t Attacking class = {cls_to_attack} \t NER detection rate = {after_adv_reported_ct}/{total_ct} = {after_adv_reported_ct / total_ct}")
    print(f"Identity matching rate = {matching_ct}/{total_ct} = {matching_ct / total_ct}")

if __name__ == '__main__':
    main()

# Baseline Detection rate = 1162/1410 = 0.8241134751773049
# Attacking method = deepwordbug replace 	 Attacking class = identity 	 Detection rate = 870/1410 = 0.6170212765957447

# Baseline Detection rate = 1173/1420 = 0.826056338028169
# Attacking method = deepwordbug delete 	 Attacking class = identity 	 Detection rate = 1067/1420 = 0.7514084507042254

# Baseline Detection rate = 1170/1416 = 0.826271186440678
# Attacking method = deepwordbug repeat 	 Attacking class = identity 	 Detection rate = 1067/1416 = 0.7535310734463276

# Baseline Detection rate = 1168/1417 = 0.8242766407904023
# Attacking method = deepwordbug switch 	 Attacking class = identity 	 Detection rate = 1049/1417 = 0.7402964008468595

# Baseline Detection rate = 1181/1327 = 0.8899773926149209
# Attacking method = bae None 	 Attacking class = identity 	 Detection rate = 1157/1327 = 0.8718914845516202

# Baseline NER detection rate = 796/913 = 0.8718510405257394
# Attacker = gpt None, With defence = False 	 Attacking class = action 	 NER detection rate = 744/913 = 0.8148959474260679

# --attacker deepwordbug --typo_type delete --cls_to_attack identity --defence