import copy
import os
import csv
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from ..encoder import IdentityBert
import pandas as pd
import langdetect
import unicodedata
import difflib
from ..reference_db import CharacterBERT, IdentityMatcher, BaseFaissIPRetriever
from .defence_utils import T5SpellFixer
import click
import json
import re
from ..utilities.data_utils import remove_trailing_digits, check_lang
import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

class MyAttackEvaluator():
    def __init__(self, correction_method=Optional[str]):
        if correction_method is not None and correction_method.lower() == 't5':
            self.corrector = T5SpellFixer()

    def check_is_match(self, prediction_set: List, expected_set: List, cutoff:float):
        count = 0
        predicted_identities = [x.lower() for x in prediction_set]
        for pred_identity in predicted_identities:
            closest_match = difflib.get_close_matches(pred_identity, expected_set, n=1, cutoff=cutoff)
            if closest_match:
                count = 1
                break
        return count

    def results_collector(self, dataset: List[Dict], tokenizer: AutoTokenizer, model: IdentityBert, result_csv_path: str, defender: Optional[str]):

        if not os.path.exists(result_csv_path):
            with open(result_csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['email_file_path',
                                 'attacked_email',
                                 'identities_after_adv',
                                 'actions_after_adv',
                                 'identities_after_adv_with_defence',
                                 'actions_after_adv_with_defence',
                                 'identities_before_adv',
                                 'actions_before_adv',
                                 'rephrased_dict'])

        for it in tqdm(range(len(dataset))):
            entry = dataset[it]
            email_file_path = entry["metadata"]
            attacked_email = entry["text"]
            rephrased_dict = entry['rephrased_text']

            if email_file_path in open(result_csv_path, encoding="utf-8").read():
                continue

            # fixme: recover to the email before attack
            unattacked_email = copy.deepcopy(attacked_email)
            for k, v in entry["rephrased_text"].items():
                unattacked_email = unattacked_email.replace(v, k)

            identities_after_adv_with_defence, actions_after_adv_with_defence = set(), set()
            if defender:
                corrected_attacked_email = self.corrector(attacked_email)
                identities_after_adv_with_defence, actions_after_adv_with_defence, *_ = model(corrected_attacked_email)

            identities_after_adv, actions_after_adv, *_ = model(attacked_email)
            identities_before_adv, actions_before_adv, *_ = model(unattacked_email)

            # Append the new row to the CSV file
            with open(result_csv_path, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([email_file_path,
                                 attacked_email,
                                 identities_after_adv,
                                 actions_after_adv,
                                 identities_after_adv_with_defence,
                                 actions_after_adv_with_defence,
                                 identities_before_adv,
                                 actions_before_adv,
                                 rephrased_dict,
                                 ])

    def metrics_collector(self, result_csv_path: str, cls_to_check: str):

        result_df = pd.read_csv(result_csv_path)
        before_adv_reported_ct = 0
        after_adv_reported_ct = 0
        total_ct = 0

        for it, row in tqdm(result_df.iterrows()):
            rephrased_dict = row["rephrased_dict"]
            identities_after_adv = row["identities_after_adv"]
            actions_after_adv = row["actions_after_adv"]
            identities_before_adv = row["identities_before_adv"]
            actions_before_adv = row["actions_before_adv"]

            if len(eval(rephrased_dict).values()):
                total_ct += 1

                if cls_to_check == "identity":
                    # expected identities
                    expected_identities_orig = [remove_trailing_digits(unicodedata.normalize("NFKC", x.lower())) for x in list(eval(rephrased_dict).keys())]
                    expected_identities_rephrased = [unicodedata.normalize("NFKC", x.lower()) for x in list(eval(rephrased_dict).values())]
                    expected_identities_all = expected_identities_rephrased + expected_identities_orig

                    ### after adv
                    after_adv_reported_ct += self.check_is_match(prediction_set=list(eval(identities_after_adv)),
                                                                 expected_set=expected_identities_all,
                                                                 cutoff=0.5)

                    ### baseline
                    before_adv_reported_ct += self.check_is_match(prediction_set=list(eval(identities_before_adv)),
                                                                 expected_set=expected_identities_all,
                                                                 cutoff=0.5)


                elif cls_to_check == "action":
                    ### after rephrasing
                    expected_actions_all = [x.lower() for x in list(eval(rephrased_dict).values())] + \
                                           [x.lower() for x in list(eval(rephrased_dict).keys())]

                    after_adv_reported_ct += self.check_is_match(prediction_set=list(eval(actions_after_adv)),
                                                                 expected_set=expected_actions_all,
                                                                 cutoff=0.3)

                    ### baseline
                    before_adv_reported_ct += self.check_is_match(prediction_set=list(eval(actions_before_adv)),
                                                                  expected_set=expected_actions_all,
                                                                  cutoff=0.3)

                else:
                    raise NotImplementedError()

        return after_adv_reported_ct, before_adv_reported_ct, total_ct


@click.command()
# @click.option('--identity_model_checkpoint', required=True, type=str, default="checkpoints/identity_adversarial_training2/checkpoint-658")
@click.option('--identity_model_checkpoint', required=True, type=str, default="checkpoints/identity-model")
@click.option('--attacker', required=True, type=click.Choice(['bae', 'deepwordbug', 'gpt', 'concatsent', 'textfooler'], case_sensitive=False), help="Specify the attacker type (e.g., 'bae')")
@click.option('--cls_to_attack', required=True, type=click.Choice(['identity', 'action'], case_sensitive=False), help="Attack which NER class")
@click.option('--typo_type', help="Specify the typo type, only for the DeepWordBug attacking method", type=click.Choice(['repeat', 'delete', 'replace', 'switch'], case_sensitive=False))
@click.option('--eval_only', help="Eval only or Attack+Eval", is_flag=True, show_default=True, default=False)
@click.option('--defender', type=click.Choice(['t5'], case_sensitive=False), help="Specify the defender type (e.g., 't5')")
def main(identity_model_checkpoint, attacker, cls_to_attack, typo_type, eval_only, defender):

    evaluator = MyAttackEvaluator(defender)
    model = IdentityBert(identity_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(identity_model_checkpoint)

    output_dir = f'./datasets/ner_training_adversarial/'
    if typo_type:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}_{typo_type}.json'), 'r') as json_file:
            dataset = json.load(json_file)
        # result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_{typo_type}.csv'
        result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_{typo_type}_no_advtraining.csv'
    else:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}.json'), 'r') as json_file:
            dataset = json.load(json_file)
        # result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}.csv'
        result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_no_advtraining.csv'
    if defender:
        result_csv_path = result_csv_path.replace(".csv", f"_with_defender_{defender}.csv")

    if eval_only:
        if not os.path.exists(result_csv_path):
            raise FileNotFoundError(f"The {result_csv_path} does not exist")
        after_adv_reported_ct,  before_adv_reported_ct, total_ct = \
                                                        evaluator.metrics_collector(result_csv_path=result_csv_path,
                                                                                    cls_to_check=cls_to_attack)
    else:
        evaluator.results_collector(dataset=dataset,
                                    tokenizer=tokenizer,
                                    model=model,
                                    result_csv_path=result_csv_path,
                                    defender=defender)

        after_adv_reported_ct, before_adv_reported_ct, total_ct = \
                                                        evaluator.metrics_collector(result_csv_path=result_csv_path,
                                                                                    cls_to_check=cls_to_attack)

    print(f"Baseline NER detection rate = {before_adv_reported_ct}/{total_ct} = {before_adv_reported_ct / total_ct}")
    print(f"After Attacker = {attacker} {typo_type} \t NER detection rate = {after_adv_reported_ct}/{total_ct} = {after_adv_reported_ct / total_ct}")


if __name__ == '__main__':
    main()

### with adv training ### ###

# After Attacker = deepwordbug switch 	 NER detection rate = 190/268 = 0.7089552238805971
# After Attacker = deepwordbug replace 	 NER detection rate = 181/260 = 0.6961538461538461
# After Attacker = deepwordbug delete 	 NER detection rate = 191/269 = 0.7100371747211895
# After Attacker = deepwordbug repeat 	 NER detection rate = 189/269 = 0.7026022304832714

# ### w/o adv training ### ###
# Baseline NER detection rate = 233/278 = 0.8381294964028777
# After Attacker = bae None 	 NER detection rate = 231/278 = 0.8309352517985612

# Baseline NER detection rate = 155/173 = 0.8959537572254336
# After Attacker = deepwordbug replace 	 NER detection rate = 130/173 = 0.7514450867052023

# Baseline NER detection rate = 182/212 = 0.8584905660377359
# After Attacker = deepwordbug repeat 	 NER detection rate = 172/212 = 0.8113207547169812

# Baseline NER detection rate = 183/208 = 0.8798076923076923
# After Attacker = deepwordbug delete 	 NER detection rate = 172/208 = 0.8269230769230769

# Baseline NER detection rate = 175/203 = 0.8620689655172413
# After Attacker = deepwordbug switch 	 NER detection rate = 162/203 = 0.7980295566502463

# Baseline NER detection rate = 171/221 = 0.7737556561085973
# After Attacker = gpt None 	 NER detection rate = 174/221 = 0.7873303167420814

# Baseline NER detection rate = 147/175 = 0.84
# After Attacker = concatsent None 	 NER detection rate = 133/175 = 0.76

# Baseline NER detection rate = 169/221 = 0.7647058823529411
# After Attacker = textfooler None 	 NER detection rate = 165/221 = 0.746606334841629
