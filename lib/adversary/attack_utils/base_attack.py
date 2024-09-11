from lib.encoder.IdentityBert import IdentityBert
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Set, Tuple

class SuperAttacker():
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

    @staticmethod
    def get_entity_from_context(contexts: List[str], entities_dict: Dict, used_entities: Set[str]) -> Tuple[Optional[str], Optional[int]]:
        """ Helper function to find the matching entity in the given contexts. """
        for context in contexts:
            if context in entities_dict and context not in used_entities:
                used_entities.add(context)
                return entities_dict[context]["token"], entities_dict[context]["truth"]
        return None, None

    def sentence_segmentation(self, tokens: List[str], ground_truth: List[int], cls_to_modify: List[int]) -> Tuple[List[str], List[str], List[str], List[int]]:
        '''
        Segment the paragraphs based on the entities
        :param tokens:
        :param ground_truth:
        :param cls_to_modify:
        :return:
        '''
        # Precompute cls_to_modify + 1 for faster lookup
        cls_to_modify_plus_one = set([x + 1 for x in cls_to_modify]) # I-xxx

        sentences_to_be_rephrased = []
        sentences_to_be_rephrased_entity_id = []
        other_sentences = []
        all_sentences = []

        current_sent = []
        current_sent_id = 0
        current_rest = []

        for token, truth in zip(tokens, ground_truth):
            token = token.lower()
            if truth in cls_to_modify:  # Start of entity
                if current_rest:
                    rest_sentence = ' '.join(current_rest).strip()
                    other_sentences.append(rest_sentence)
                    all_sentences.append(rest_sentence)
                    current_rest = []
                current_sent.append(token)
                current_sent_id = truth

            elif truth in cls_to_modify_plus_one and current_sent:  # Continue collecting the entity
                current_sent.append(token)

            else:  # Collect non-entity tokens
                if current_sent:  # End of entity, append the collected sentence
                    rephrased_sentence = ' '.join(current_sent).strip()
                    sentences_to_be_rephrased.append(rephrased_sentence)
                    all_sentences.append(rephrased_sentence)
                    sentences_to_be_rephrased_entity_id.append(current_sent_id)
                    current_sent = []
                    current_sent_id = 0
                current_rest.append(token)  # Add to non-entity part

        # Append any remaining sentences
        if current_sent:
            rephrased_sentence = ' '.join(current_sent).strip()
            sentences_to_be_rephrased.append(rephrased_sentence)
            all_sentences.append(rephrased_sentence)
            sentences_to_be_rephrased_entity_id.append(current_sent_id)

        if current_rest:
            rest_sentence = ' '.join(current_rest).strip()
            other_sentences.append(rest_sentence)
            all_sentences.append(rest_sentence)

        return sentences_to_be_rephrased, other_sentences, all_sentences, sentences_to_be_rephrased_entity_id


    def get_transformations(self) -> Dict:
        raise NotImplementedError()

    def process_entries(self) -> List[Dict]:
        raise NotImplementedError()