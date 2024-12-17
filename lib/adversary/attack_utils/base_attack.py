from ...encoder.IdentityBert import IdentityBert
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

    def get_transformations(self, model: IdentityBert, tokens: List[str], ground_truth: List[int], tokenizer: AutoTokenizer, cls_to_modify: List[int]=[1]) -> Dict:
        '''
        Attack one sample
        :param model:
        :param tokens:
        :param ground_truth: ground truth token labels
        :param tokenizer:
        :param cls_to_modify: 1 stands for identity, 5 stands for action
        :return:
        '''
        raise NotImplementedError()

    def process_entries(self, data: List[Dict], model: Optional[IdentityBert], tokenizer: Optional[AutoTokenizer]) -> List[Dict]:
        '''
        Attack all samples
        :param data:
        :param model:
        :param tokenizer:
        :return:
        '''
        raise NotImplementedError()