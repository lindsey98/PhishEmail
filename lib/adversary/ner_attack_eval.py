import copy
import csv
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer
from ..encoder import IdentityBert
import pandas as pd
import unicodedata
import difflib
import click
import json
from ..utilities.data_utils import remove_trailing_digits
import os


class MyAttackEvaluator():
    def __init__(self):
        pass

    @staticmethod
    def check_is_match(prediction_set: List, expected_set: List, cutoff:float):
        '''
        Use SequenceMatcher to find a closest match
        :param prediction_set:  query
        :param expected_set:  reference set
        :param cutoff:
        :return:
        '''
        count = 0
        predicted_identities = [x.lower() for x in prediction_set]
        for pred_identity in predicted_identities:
            closest_match = difflib.get_close_matches(pred_identity, expected_set, n=1, cutoff=cutoff)
            if closest_match:
                count = 1
                break
        return count

    def results_collector(self, dataset: List[Dict], tokenizer: AutoTokenizer, model: IdentityBert, result_csv_path: str):
        '''
        Log results
        :param dataset:
        :param tokenizer:
        :param model:
        :param result_csv_path:
        :return:
        '''
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
        '''
        Compute metrics after attack (NER recognition rate)
        :param result_csv_path:
        :param cls_to_check:
        :return:
        '''
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
                    expected_actions_all = [x.lower() for x in list(eval(rephrased_dict).values())] + [x.lower() for x in list(eval(rephrased_dict).keys())]

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
@click.option('--identity_model_checkpoint', required=True, type=str, default="checkpoints/identity_adversarial_training/checkpoint-770")
# @click.option('--identity_model_checkpoint', required=True, type=str, default="checkpoints/identity-model")
@click.option('--attacker', required=True, type=click.Choice(['bae', 'deepwordbug', 'gpt', 'concatsent', 'textfooler'], case_sensitive=False), help="Specify the attacker type (e.g., 'bae')")
@click.option('--cls_to_attack', required=True, type=click.Choice(['identity', 'action'], case_sensitive=False), help="Attack which NER class")
@click.option('--typo_type', help="Specify the typo type, only for the DeepWordBug attacking method", type=click.Choice(['repeat', 'delete', 'replace', 'switch'], case_sensitive=False))
@click.option('--eval_only', help="Eval only or Attack+Eval", is_flag=True, show_default=True, default=False)
def main(identity_model_checkpoint, attacker, cls_to_attack, typo_type, eval_only):

    evaluator = MyAttackEvaluator()
    model = IdentityBert(identity_model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(identity_model_checkpoint)

    output_dir = f'./datasets/ner_training_adversarial/'
    if typo_type:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}_{typo_type}.json'), 'r') as json_file:
            dataset = json.load(json_file)
        result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_{typo_type}.csv'
        # result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_{typo_type}_no_advtraining.csv'
    else:
        with open(os.path.join(output_dir, f'adversarial_rephrase_{attacker}.json'), 'r') as json_file:
            dataset = json.load(json_file)
        result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}.csv'
        # result_csv_path = f'./datasets/nazario_results_adversarial_{attacker}_no_advtraining.csv'

    if eval_only:
        if not os.path.exists(result_csv_path):
            raise FileNotFoundError(f"The {result_csv_path} does not exist")
        after_adv_reported_ct,  before_adv_reported_ct, total_ct = evaluator.metrics_collector(result_csv_path=result_csv_path,
                                                                                               cls_to_check=cls_to_attack)
    else:
        evaluator.results_collector(dataset=dataset, tokenizer=tokenizer, model=model, result_csv_path=result_csv_path)

        after_adv_reported_ct, before_adv_reported_ct, total_ct = evaluator.metrics_collector(result_csv_path=result_csv_path,
                                                                                              cls_to_check=cls_to_attack)

    print(f"Clean detection rate = {before_adv_reported_ct}/{total_ct} = {before_adv_reported_ct / total_ct}")
    print(f"After Attack = {attacker} {typo_type} \t detection rate = {after_adv_reported_ct}/{total_ct} = {after_adv_reported_ct / total_ct}")


if __name__ == '__main__':
    main()

### with adv training ### ###

# Clean detection rate = 247/278 = 0.8884892086330936
# After Attack = bae None 	 detection rate = 241/278 = 0.8669064748201439

# Clean detection rate = 162/173 = 0.9364161849710982
# After Attack = deepwordbug replace 	 detection rate = 162/173 = 0.9364161849710982

# Clean detection rate = 196/212 = 0.9245283018867925
# After Attack = deepwordbug repeat 	 detection rate = 194/212 = 0.9150943396226415

# Clean detection rate = 183/203 = 0.9014778325123153
# After Attack = deepwordbug switch 	 detection rate = 185/203 = 0.9113300492610837

# Clean detection rate = 189/208 = 0.9086538461538461
# After Attack = deepwordbug delete 	 detection rate = 191/208 = 0.9182692307692307

# Clean detection rate = 199/221 = 0.9004524886877828
# After Attack = gpt None 	 detection rate = 195/221 = 0.8823529411764706

# Clean detection rate = 158/175 = 0.9028571428571428
# After Attack = concatsent None 	 detection rate = 156/175 = 0.8914285714285715

# Clean detection rate = 198/221 = 0.8959276018099548
# After Attack = textfooler None 	 detection rate = 197/221 = 0.8914027149321267