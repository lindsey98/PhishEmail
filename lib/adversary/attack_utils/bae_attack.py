from OpenAttack.attackers import BAEAttacker
import torch
import copy
from transformers import AutoTokenizer
from typing import List, Dict
from tqdm import tqdm
import os
import json
from lib.encoder.IdentityBert import IdentityBert
from .base_attack import SuperAttacker

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error

class MyBAEAttacker(BAEAttacker, SuperAttacker):

    def explicit_truncate_ground_truth(self, ground_truth_labels):
        if len(ground_truth_labels) > self.max_length:
            return ground_truth_labels[:self.max_length-1] + ['[SEP]'] # truncate again
        else:
            return ground_truth_labels

    def explicit_truncate_tokens(self, long_tokens):
        if len(long_tokens) > self.max_length:
            return long_tokens[:self.max_length-1] + ['[SEP]'] # truncate again
        else:
            return long_tokens

    ##### TODO: make this one of the substitute unit under ./substitures #####
    def get_substitues(self, masked_index, tokens, tokenizer, model, sub_mode, k, threshold=3.0):
        masked_tokens = copy.deepcopy(tokens)

        if sub_mode == "r":
            masked_tokens[masked_index] = '[MASK]'

        elif sub_mode == "i":
            masked_tokens.insert(masked_index, '[MASK]')
        else:
            raise NotImplementedError()

        masked_tokens = self.explicit_truncate_tokens(masked_tokens) # truncate again

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(masked_tokens)

        # truncate again:
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)

        model.eval()

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        predicted_indices = torch.topk(predictions[0, masked_index], self.k)[1]
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)
        return predicted_tokens

    def get_transformations(self, model, tokens, ground_truth, tokenizer, cls_to_modify: List[int]=[1, 3, 5]):

        # with CLS and SEP tokens added
        tokens = ['[CLS]'] + tokens[:self.max_length - 2] + ['[SEP]']
        ground_truth = [-100] + ground_truth[:self.max_length - 2] + [-100]

        sent = " ".join(tokens)

        # get the original prob
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
        orig_probs = torch.Tensor(model.get_prob(input_ids))

        # Those are the interested entities to be inserted
        interested_indices = [i for i in range(len(ground_truth)) if (ground_truth[i] in cls_to_modify)]

        offset = 0
        final_words = copy.deepcopy(tokens)
        final_ground_truth = copy.deepcopy(ground_truth)
        inserted_new_tokens = {}

        for top_index in interested_indices:
            tgt_word = tokens[top_index]
            orig_prob = orig_probs[top_index][ground_truth[top_index]]

            if tgt_word in self.filter_words:
                continue
            if top_index + offset + 1 >= self.max_length: # if after the insertion the original entity would be pushed out of boundary => skip
                continue

            substitutes = self.get_substitues(top_index,
                                              tokens,
                                              tokenizer,
                                              self.mlm_model,
                                              'i',
                                              self.k,
                                              self.threshold_pred_score)
            most_gap = 0.0
            candidate = None

            for i, substitute in enumerate(substitutes):
                if substitute == tgt_word or '##' in substitute or substitute in self.filter_words:
                    continue

                temp_replace = final_words
                temp_replace.insert(top_index + offset, substitute) # insert to the left of the entity

                temp_replace = self.explicit_truncate_tokens(temp_replace)
                if len(temp_replace) > self.max_length:  # Safety check
                    continue

                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                use_score = self.encoder.calc_score(temp_text, sent)

                # From TextAttack's implementation: Finally, since the BAE code is based on the TextFooler code, we need to
                # adjust the threshold to account for the missing / pi in the cosine
                # similarity comparison. So the final threshold is 1 - (1 - 0.8) / pi
                # = 1 - (0.2 / pi) = 0.936338023.
                if use_score < 0.936:
                    continue

                input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(temp_replace)])
                temp_prob = torch.Tensor(model.get_prob(input_ids))

                label_prob = temp_prob[top_index + offset + 1][ground_truth[top_index]] # now the original interested indice becomes top_index + offset + 1
                gap = orig_prob - label_prob # decrease in confidence
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

                final_words.pop(top_index + offset) # pop out

            # keep the subsitute that most significantly decrease the confidence
            if most_gap > 0:
                inserted_new_tokens[tgt_word + '_' + str(top_index)] = candidate
                final_words.insert(top_index + offset, candidate)
                final_ground_truth.insert(top_index + offset, 0)

                final_words = self.explicit_truncate_tokens(final_words)
                final_ground_truth = self.explicit_truncate_tokens(final_ground_truth)

                if len(final_words) >= self.max_length:
                    break  # Stop processing further to prevent exceeding max_length

                offset += 1

        return {
            "perturbed_tokens": final_words,
            "perturbed_ground_truth": final_ground_truth,
            "rephrased_dict": inserted_new_tokens
        }


    def process_entries(self, data, model, tokenizer):
        seen_texts = set()
        processed_data = []

        for entry in tqdm(data):
            tokens = entry['tokens']
            tags = entry['ner_tags']
            text = ' '.join(tokens)
            if text in seen_texts:
                continue  # Skip duplicates
            seen_texts.add(text)

            rephrased_result = self.get_transformations(model=model,
                                                        tokens=tokens,
                                                        tokenizer=tokenizer,
                                                        ground_truth=tags,
                                                        cls_to_modify=[1, 3])
            rephrased_tokens = rephrased_result["perturbed_tokens"]
            rephrased_tags = rephrased_result["perturbed_ground_truth"]

            processed_data.append({
                "id": entry['id'],
                "tokens": rephrased_tokens,
                "ner_tags": rephrased_tags,
                'rephrased_dict': rephrased_result["rephrased_dict"],
                "metadata": entry["metadata"]  # Add metadata field
            })

        return processed_data

# if __name__ == '__main__':
    # label_list = [
    #     "O",
    #     "B-identity",
    #     "I-identity",
    #     "B-relation",
    #     "I-relation",
    #     "B-action",
    #     "I-action",
    # ]
    # label_to_id = {label: idx for idx, label in enumerate(label_list)}
    #
    # with open('./datasets/ner_training_augmented/test.json', 'r') as json_file:
    #     data = json.load(json_file)
    #
    # transformation = MyBAEAttacker()
    # model = IdentityBert("./checkpoints/identity-model", aggregation_strategy="none")
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
    #
    # processed_data = []
    # seen_texts = set()
    # transformation.process_entries(data, model, tokenizer)
    #
    # output_dir = f'./datasets/ner_training_adversarial/'
    # os.makedirs(output_dir, exist_ok=True)
    # with open(os.path.join(output_dir, f'adversarial_rephrase_bae.json'), 'w') as outfile:
    #     json.dump(processed_data, outfile, indent=4)
