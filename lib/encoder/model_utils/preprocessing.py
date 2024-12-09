
import re


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)


    word_ids = tokenized_inputs.word_ids()  # This function is applied directly to the tokenized_inputs.
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Iterate through all word_ids for the tokens
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(examples['ner_tags'][word_idx])  # Access the label using word_idx directly from ner_tags
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids  # Assign labels directly, not in a nested list
    return tokenized_inputs

