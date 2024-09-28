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
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

class MyAttackEvaluator():
    def __init__(self, correction_method=Optional[str]):
        if correction_method is not None and correction_method.lower() == 't5':
            self.corrector = T5SpellFixer()

    def model_pred_with_explicit_tokenization(self, email: str, tokenizer: AutoTokenizer, model: IdentityBert):
        tokens = tokenizer.tokenize(email, return_offsets_mapping=True, truncation=True)
        tokenized_email = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
        tokenized_email_str = tokenizer.convert_tokens_to_string(tokenized_email)
        identities, actions, relations, urls_after_actions, identity_recog_runtime = model(tokenized_email_str)

        return identities, actions, identity_recog_runtime

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
                                 'rephrased_dict',
                                 'pred_time'])

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
                identities_after_adv_with_defence, actions_after_adv_with_defence, _ = \
                    self.model_pred_with_explicit_tokenization(corrected_attacked_email, tokenizer, model)

            identities_after_adv, actions_after_adv, identity_recog_runtime_after_adv = \
                self.model_pred_with_explicit_tokenization(attacked_email, tokenizer, model)
            identities_before_adv, actions_before_adv, _ = \
                self.model_pred_with_explicit_tokenization(unattacked_email, tokenizer, model)

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
                                 identity_recog_runtime_after_adv
                                 ])

    def metrics_collector(self, result_csv_path: str, matcher_cls: IdentityMatcher, embed_model: CharacterBERT, cls_to_check: str):
        result_df = pd.read_csv(result_csv_path)

        before_adv_reported_ct = 0
        after_adv_reported_ct = 0
        after_adv_with_defence_reported_ct = 0
        matching_ct = 0
        total_ct = 0
        seen_email_files = set()

        for it, row in tqdm(result_df.iterrows()):
            email_file_path = row['email_file_path']
            parsed_email = row["attacked_email"]
            rephrased_dict = row["rephrased_dict"]
            identities_after_adv = row["identities_after_adv"]
            actions_after_adv = row["actions_after_adv"]
            identities_after_adv_with_defence = row.get("identities_after_adv_with_defence", None)
            actions_after_adv_with_defence = row.get("actions_after_adv_with_defence", None)
            identities_before_adv = row["identities_before_adv"]
            actions_before_adv = row["actions_before_adv"]

            if email_file_path in seen_email_files:
                continue
            else:
                seen_email_files.add(email_file_path)

            if not check_lang(parsed_email):
                continue

            if len(eval(rephrased_dict).values()):
                total_ct += 1

                if cls_to_check == "identity":
                    # expected identities
                    expected_identities_orig = [remove_trailing_digits(unicodedata.normalize("NFKC", x.lower())) for x in list(eval(rephrased_dict).keys())]
                    expected_identities_rephrased = [unicodedata.normalize("NFKC", x.lower()) for x in list(eval(rephrased_dict).values())]
                    expected_identities_all = expected_identities_rephrased + expected_identities_orig

                    # # build an index base for those expected identities
                    # orig_index_db = BaseFaissIPRetriever(tags=expected_identities_orig, embed_model=embed_model)
                    #
                    # ## matching acc
                    # for id in expected_identities_rephrased:
                    #     _, closest_match = matcher_cls.find_closest_match(query=id, value_index_db=orig_index_db)
                    #     if closest_match:
                    #         matching_ct += 1
                    #         break
                    # orig_index_db.add(None, expected_identities_rephrased)

                    ### after adv
                    after_adv_reported_ct += self.check_is_match(prediction_set=list(eval(identities_after_adv)),
                                                                 expected_set=expected_identities_all,
                                                                 cutoff=0.5)


                    ### after adv with defence
                    if identities_after_adv_with_defence is not None:
                        after_adv_with_defence_reported_ct += self.check_is_match(prediction_set=list(eval(identities_after_adv_with_defence)),
                                                                 expected_set=expected_identities_all,
                                                                 cutoff=0.5)

                    ### baseline
                    before_adv_reported_ct += self.check_is_match(prediction_set=list(eval(identities_before_adv)),
                                                                 expected_set=expected_identities_all,
                                                                 cutoff=0.5)


                elif cls_to_check == "action":
                    # matching_ct += 1
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

        return after_adv_reported_ct, after_adv_with_defence_reported_ct, before_adv_reported_ct, matching_ct, total_ct


@click.command()
# @click.option('--checkpoint', required=True, type=str, default="checkpoints/identity_adversarial_training/checkpoint-435")
@click.option('--identity_model_checkpoint', required=True, type=str, default="checkpoints/identity-model")
@click.option('--embed_model_checkpoint', required=True, type=str, default="./checkpoints/characterbert-typos-st/")
@click.option('--matching_thre', required=True, type=float, default=0.85)
@click.option('--attacker', required=True, type=click.Choice(['bae', 'deepwordbug', 'scpn', 'gpt',
                                                              'viper', 'bart', 't5', 'concatsent', 'textfooler'], case_sensitive=False), help="Specify the attacker type (e.g., 'bae')")
@click.option('--cls_to_attack', required=True, type=click.Choice(['identity', 'action'], case_sensitive=False), help="Attack which NER class")
@click.option('--typo_type', help="Specify the typo type, only for the DeepWordBug attacking method", type=click.Choice(['repeat', 'delete', 'replace', 'switch'], case_sensitive=False))
@click.option('--eval_only', help="Eval only or Attack+Eval", is_flag=True, show_default=True, default=False)
@click.option('--defender', type=click.Choice(['t5'], case_sensitive=False), help="Specify the defender type (e.g., 't5')")
def main(identity_model_checkpoint, embed_model_checkpoint, matching_thre,
         attacker, cls_to_attack, typo_type, eval_only, defender):

    evaluator = MyAttackEvaluator(defender)
    embed_model = CharacterBERT(embed_model_checkpoint)
    matcher_cls = IdentityMatcher(brand_index_db=None,
                                  internal_relation_index_db=None,
                                  embed_model=embed_model,
                                  brand_domain_map_path=None,
                                  knowledge_base_expansion=False,
                                  gpt_client=None, gpt_assistant=None,
                                  check_action=True,
                                  threshold=matching_thre)

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

    model = IdentityBert(identity_model_checkpoint, aggregation_strategy="simple")
    tokenizer = AutoTokenizer.from_pretrained(identity_model_checkpoint)

    if eval_only:
        if not os.path.exists(result_csv_path):
            raise FileNotFoundError(f"The {result_csv_path} does not exist")
        after_adv_reported_ct, after_adv_with_defence_reported_ct, before_adv_reported_ct, matching_ct, total_ct = \
                                                        evaluator.metrics_collector(result_csv_path=result_csv_path,
                                                                                    matcher_cls=matcher_cls,
                                                                                    embed_model=embed_model,
                                                                                    cls_to_check=cls_to_attack)

    else:
        evaluator.results_collector(dataset=dataset,
                                    tokenizer=tokenizer,
                                    model=model,
                                    result_csv_path=result_csv_path,
                                    defender=defender)

        after_adv_reported_ct, after_adv_with_defence_reported_ct, before_adv_reported_ct, matching_ct, total_ct = \
                                                        evaluator.metrics_collector(result_csv_path=result_csv_path,
                                                                                    matcher_cls=matcher_cls,
                                                                                    embed_model=embed_model,
                                                                                    cls_to_check=cls_to_attack)

    print(f"Baseline NER detection rate = {before_adv_reported_ct}/{total_ct} = {before_adv_reported_ct / total_ct}")
    if defender:
        print(
            f"Attacker = {attacker} {typo_type} \t With Defence \t NER detection rate = {after_adv_with_defence_reported_ct}/{total_ct} = {after_adv_with_defence_reported_ct / total_ct}")
    else:
        print(
            f"Attacker = {attacker} {typo_type} \t NER detection rate = {after_adv_reported_ct}/{total_ct} = {after_adv_reported_ct / total_ct}")


if __name__ == '__main__':
    main()

###### With Adversairal training #############################################################################################################
# Baseline NER detection rate = 191/215 = 0.8883720930232558
# Attacker = deepwordbug replace 	 NER detection rate = 175/215 = 0.813953488372093

# Baseline NER detection rate = 201/229 = 0.8777292576419214
# Attacker = deepwordbug repeat 	 NER detection rate = 189/229 = 0.8253275109170306

# Baseline NER detection rate = 203/231 = 0.8787878787878788
# Attacker = deepwordbug delete 	 NER detection rate = 190/231 = 0.8225108225108225

# Baseline NER detection rate = 196/220 = 0.8909090909090909
# Attacker = deepwordbug switch 	 NER detection rate = 178/220 = 0.8090909090909091

# Baseline NER detection rate = 215/246 = 0.8739837398373984
# Attacker = bae None 	 NER detection rate = 209/246 = 0.8495934959349594

# Baseline NER detection rate = 154/198 = 0.7777777777777778
# Attacker = textfooler None 	 NER detection rate = 138/198 = 0.696969696969697

# Baseline NER detection rate = 154/199 = 0.7738693467336684
# Attacker = t5 None 	 NER detection rate = 156/199 = 0.7839195979899497

# Baseline NER detection rate = 157/201 = 0.7810945273631841
# Attacker = gpt None 	 NER detection rate = 148/201 = 0.736318407960199

# Baseline NER detection rate = 127/162 = 0.7839506172839507
# Attacker = concatsent None 	 NER detection rate = 116/162 = 0.7160493827160493


###### Without Adversairal training #############################################################################################################

# Baseline NER detection rate = 192/215 = 0.8930232558139535
# Attacker = deepwordbug replace 	 NER detection rate = 171/215 = 0.7953488372093023

# Baseline NER detection rate = 203/229 = 0.8864628820960698
# Attacker = deepwordbug repeat 	 NER detection rate = 185/229 = 0.8078602620087336

# Baseline NER detection rate = 203/231 = 0.8787878787878788
# Attacker = deepwordbug delete 	 NER detection rate = 188/231 = 0.8138528138528138

# Baseline NER detection rate = 194/220 = 0.8818181818181818
# Attacker = deepwordbug switch 	 NER detection rate = 170/220 = 0.7727272727272727

# Baseline NER detection rate = 218/246 = 0.8861788617886179
# Attacker = bae None 	 NER detection rate = 209/246 = 0.8495934959349594

# Baseline NER detection rate = 152/198 = 0.7676767676767676
# Attacker = textfooler None 	 NER detection rate = 146/198 = 0.7373737373737373

# Baseline NER detection rate = 153/199 = 0.7688442211055276
# Attacker = t5 None 	 NER detection rate = 154/199 = 0.7738693467336684

# Baseline NER detection rate = 156/201 = 0.7761194029850746
# Attacker = gpt None 	 NER detection rate = 155/201 = 0.7711442786069652

# Baseline NER detection rate = 133/162 = 0.8209876543209876
# Attacker = concatsent None 	 NER detection rate = 125/162 = 0.7716049382716049

########### With T5 defender ####################################################################################################
# Baseline NER detection rate = 203/229 = 0.8864628820960698
# Attacker = deepwordbug repeat    With Defence    NER detection rate = 197/229 = 0.8602620087336245

# Baseline NER detection rate = 192/215 = 0.8930232558139535
# Attacker = deepwordbug replace   With Defence    NER detection rate = 187/215 = 0.8697674418604651

# Baseline NER detection rate = 203/231 = 0.8787878787878788
# Attacker = deepwordbug delete    With Defence    NER detection rate = 189/231 = 0.8181818181818182

# Baseline NER detection rate = 194/220 = 0.8818181818181818
# Attacker = deepwordbug switch    With Defence    NER detection rate = 185/220 = 0.8409090909090909