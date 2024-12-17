from .attack_utils import MyDeepWordBugAttacker
import json
from ..reference_db import CharacterBERT
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

@torch.inference_mode()
def get_embedding(text, model, tokenizer):
    '''
    Get the embedding
    :param text:
    :param model:
    :param tokenizer:
    :return:
    '''
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"]
    # Get the model's output (hidden states)
    outputs = model(**tokens)
    # Extract the embeddings (last hidden state)
    last_hidden_state = outputs.last_hidden_state
    # Map tokens to words for clarity
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Aggregate embeddings for the word "payppal" (subwords: 'pay', '##pp', '##al')
    subword_embeddings = last_hidden_state[0][1:4]  # Indices of 'pay', '##pp', '##al'
    word_embedding = subword_embeddings.mean(dim=0)  # Average embeddings of subwords
    word_embedding = F.normalize(word_embedding, p=2, dim=-1)
    return word_embedding


if __name__ == '__main__':
    REF_IDENTITY_MAP = './checkpoints/company_database_knowphish.json'
    with open(REF_IDENTITY_MAP, 'r') as file:
        brand_domain_map = json.load(file)
    brand_name_variants = list(brand_domain_map.keys())

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    # charbert_encoder = CharacterBERT("./checkpoints/characterbert-typos-st")
    # charbert_encoder = CharacterBERT("../CharacterBERT-DR/model_msmarco_characterbert_st/checkpoint-80000")
    charbert_encoder = CharacterBERT("./checkpoints/characterbert-typos-st-adv")
    bert_alignment_list = []
    charbert_alignment_list = []

    # for typo_type in ['repeat', 'delete', 'switch', 'replace']:
    typo_type = 'replace'
    transformation = MyDeepWordBugAttacker(transform=typo_type, power=1)

    for brand_name in tqdm(brand_name_variants):
        typo_brand_name = transformation.transform(transformation.transformer, brand_name)
        if typo_brand_name:
            typo_brand_name = typo_brand_name[0]
        else:
            typo_brand_name = brand_name

        charbert_embeddings, _ = charbert_encoder([brand_name, typo_brand_name])
        charbert_embedding = charbert_embeddings[0, :]
        charbert_embedding_typo = charbert_embeddings[1, :]
        charbert_alignment = torch.dot(charbert_embedding, charbert_embedding_typo).item()

        bert_embedding = get_embedding(brand_name, bert_model, bert_tokenizer)
        bert_embedding_typo = get_embedding(typo_brand_name, bert_model, bert_tokenizer)
        bert_alignment = torch.dot(bert_embedding, bert_embedding_typo).item()

        bert_alignment_list.append(bert_alignment)
        charbert_alignment_list.append(charbert_alignment)

    print(f'Typo type = {typo_type}, CharBERT matching rate = {np.sum(np.asarray(charbert_alignment_list) > 0.78) / len(charbert_alignment_list)} \n '
          f'BERT matching rate = {np.sum(np.asarray(bert_alignment_list) > 0.78) / len(bert_alignment_list)}')

    # x = np.asarray(bert_alignment_list)
    # y = np.asarray(charbert_alignment_list)
    # xweights = 100 * np.ones_like(x) / x.size
    # yweights = 100 * np.ones_like(y) / y.size
    #
    # fig, ax = plt.subplots()
    # ax.hist(x, weights=xweights, color='lightblue', alpha=0.5)
    # ax.hist(y, weights=yweights, color='salmon', alpha=0.5)
    #
    # ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
    # ax.margins(0.05)
    # ax.set_ylim(bottom=0)
    # plt.savefig('debug.png')