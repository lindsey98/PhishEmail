# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

测试训练好的模型。

Author: pankeyu
Date: 2023/01/11
"""
from typing import List
import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_jsonl, get_body_text
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

def inference(
    model, 
    tokenizer, 
    sentences: List[str],
    device: str,
    batch_size=16,
    max_seq_len=128
    ) -> List[int]:
    """
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        sentences (List[str]): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        max_seq_len (int, optional): _description_. Defaults to 128.

    Returns:
        List[int]: [laebl1, label2, label3, ...]
    """
    res = []
    for i in range(0, len(sentences), batch_size):
        batch_sentence = sentences[i:i+batch_size]
        ipnuts = tokenizer(
            batch_sentence,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        output = model(
            input_ids=ipnuts['input_ids'].to(device),
            token_type_ids=ipnuts['token_type_ids'].to(device),
            attention_mask=ipnuts['attention_mask'].to(device),
        )
        logits = output.logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        res.extend(preds)
    return res


if __name__ == '__main__':
    device = 'cuda:0'  # 指定GPU设备
    saved_model_path = './checkpoints/model_best'       # 训练模型存放地址
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path) 
    model.to(device).eval()

    jsonl_data = load_jsonl('./datasets/nazario_annot.jsonl')

    for item in jsonl_data:
        sentences = get_body_text(item['text'])
        sentences = sent_tokenize(sentences)  # Using NLTK for sentence tokenization

        processed_text = ""
        res = inference(model, tokenizer, sentences, device)
        print('detected sentence: ', np.asarray(sentences)[np.asarray(res)==1])
        print()