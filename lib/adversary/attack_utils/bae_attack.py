from OpenAttack.attackers import BAEAttacker
import torch
import copy
from typing import List, Dict, Optional
from tqdm import tqdm
from .base_attack import SuperAttacker
from transformers import AutoTokenizer, BertForMaskedLM
from ...encoder.IdentityBert import IdentityBert
import re
import ssl
from ...utilities.data_utils import match_case, tokenize_and_map
ssl._create_default_https_context = ssl._create_unverified_context # fix the ssl certificate expiration error

class MyBAEAttacker(BAEAttacker, SuperAttacker):
    _CallerPrefix = "BAE Attacker"
    _Help = "Insert a token before the entity"

    def explicit_truncate_ground_truth(self, ground_truth_labels: List[int]):
        '''
        truncate the labels within max_length
        :param ground_truth_labels:
        :return:
        '''
        if len(ground_truth_labels) > self.max_length:
            return ground_truth_labels[:self.max_length-1] + [-100] # truncate again
        else:
            return ground_truth_labels

    def explicit_truncate_tokens(self, long_tokens: List[str]):
        '''
        truncate the tokens within max_length
        :param ground_truth_labels:
        :return:
        :param long_tokens:
        :return:
        '''
        if len(long_tokens) > self.max_length:
            return long_tokens[:self.max_length-1] + ['[SEP]'] # truncate again
        else:
            return long_tokens

    @torch.inference_mode()
    def get_substitues(self, masked_index: int, tokens: List[str],
                       tokenizer: AutoTokenizer, model: BertForMaskedLM,
                       sub_mode: str, k: int, threshold: float=3.0) -> List[str]:
        '''
        get top-k most probable candidate tokens to insert here
        :param masked_index:
        :param tokens:
        :param tokenizer:
        :param model:
        :param sub_mode:
        :param k:
        :param threshold:
        :return: top-k candidate tokens
        '''

        masked_tokens = copy.deepcopy(tokens)
        if sub_mode == "r":
            masked_tokens[masked_index] = '[MASK]'
        elif sub_mode == "i":
            masked_tokens.insert(masked_index, '[MASK]')
        else:
            raise NotImplementedError()

        masked_tokens = self.explicit_truncate_tokens(masked_tokens) # truncate again in case the insertion cause length overflow

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(masked_tokens)

        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)

        # Predict all tokens
        model.eval()
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

        # Get the top-k most probable tokens at the masked index
        predicted_indices = torch.topk(predictions[0, masked_index], k)[1]
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)
        return predicted_tokens

    def get_transformations(self, model: IdentityBert, tokens: List[str],
                            ground_truth: List[int], tokenizer: AutoTokenizer,
                            cls_to_modify: List[int]=[1]) -> Dict:
        '''
        before the entity's starting token, predict top-k most probable candidate tokens to insert here, select the one that decreases the model confidence the most
        :param model:
        :param tokens:
        :param ground_truth:
        :param tokenizer:
        :param cls_to_modify:
        :return: tokens to be inserted
        '''
        # with CLS and SEP tokens added
        tokens = ['[CLS]'] + tokens[:self.max_length - 2] + ['[SEP]']
        ground_truth = [-100] + ground_truth[:self.max_length - 2] + [-100]

        sent = tokenizer.convert_tokens_to_string(tokens)

        # get the original prob
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
        orig_probs = torch.Tensor(model.get_prob(input_ids))

        # Those are the interested entities' indices to be inserted
        interested_indices = [i for i in range(len(ground_truth)) if (ground_truth[i] in cls_to_modify)]

        offset = 0
        final_words = copy.deepcopy(tokens)
        inserted_new_tokens = {}

        for top_index in interested_indices:
            tgt_word = tokens[top_index]
            ground_truth_cls_idx = ground_truth[top_index]
            orig_prob = orig_probs[top_index][ground_truth_cls_idx]

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
                if substitute == tgt_word or \
                        '##' in substitute or \
                        substitute in self.filter_words:
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

                label_prob = temp_prob[top_index + offset + 1][ground_truth_cls_idx] # now the original interested indice becomes top_index + offset + 1
                gap = orig_prob - label_prob # decrease in confidence in the most significant way
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

                final_words.pop(top_index + offset) # pop out

            # keep the subsitute that most significantly decrease the confidence
            if most_gap > 0:
                inserted_new_tokens[tgt_word] = candidate
                final_words.insert(top_index + offset, candidate)
                final_words = self.explicit_truncate_tokens(final_words)

                if len(final_words) >= self.max_length:
                    break  # Stop processing further to prevent exceeding max_length

                offset += 1

        return inserted_new_tokens


    def process_entries(self, data: List[Dict], model: Optional[IdentityBert], tokenizer: Optional[AutoTokenizer]) -> List[Dict]:
        '''
        Conduct attack on a list of data, each data is a dict with 'text' and 'annotations'
        :param data:
        :param model:
        :param tokenizer:
        :return: the list of data after attack
        '''
        seen_texts = set()
        processed_data = []

        for entry in tqdm(data):
            orig_text = entry['text']
            if orig_text in seen_texts:
                continue  # Skip duplicates
            seen_texts.add(orig_text)

            annotations = entry.get('annotations', [])
            if len(annotations) == 0:
                continue

            for annot in annotations:
                annot['text'] = re.sub(r"From:\s*", "", annot['text'], re.IGNORECASE) # very ad-hoc fix

            rephrased = {}
            tokens, tags = tokenize_and_map(orig_text, annotations, tokenizer, SuperAttacker.label_to_id)
            rephrased_dict = self.get_transformations(model, tokens, tags, tokenizer, cls_to_modify=[1])

            have_attacked = False
            for annot in annotations:
                entity = annot['text']
                entity_cls = annot['labels'][0]
                entity_tokens = tokenizer.tokenize(entity)
                old_entity_starting_token = entity_tokens[0]

                if have_attacked == False and entity_cls == "identity":
                    if old_entity_starting_token in rephrased_dict.keys():
                        new_entity_starting_token = rephrased_dict[old_entity_starting_token]
                        new_entity = tokenizer.convert_tokens_to_string([new_entity_starting_token] + entity_tokens)

                        # Use a case-insensitive replacement while preserving the original case
                        def replace_with_case(match):
                            matched_case_new_entity = match_case(new_entity, match.group(0))
                            # Update annot['text'] with the properly cased version
                            rephrased[entity] = matched_case_new_entity
                            annot['text'] = matched_case_new_entity
                            return matched_case_new_entity

                        orig_text = re.sub(re.escape(entity), replace_with_case, orig_text, flags=re.IGNORECASE)
                        have_attacked = True

            processed_data.append({
                "id": entry['Id'],
                "text": orig_text,
                "annotations": annotations,
                "metadata": entry.get("Path", ""),  # Add metadata field
                "rephrased_text": rephrased
            })

        return processed_data

