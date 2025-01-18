
import re
from tqdm import tqdm
from lib.model_utils.modeling import CharacterIndexer

def remove_extra_spaces(text):
    # remove text surrounded by <>, since they are likely be comments that are invisible
    text_content = re.sub(r'<[^>]*>', '', text)
    # replace multiple newline characters with a single \n
    text_content = re.sub(r'\n+', '\n', text_content)
    # replace multiple consecutive periods with a single period
    text_content = re.sub(r'\.{2,}', '', text_content)
    # replace multiple spaces with a single space
    text_content = re.sub(r'\s+', ' ', text_content)
    return text_content

def remove_urls(text):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(pattern, '', text)

def prepare_prompt_no_output_unstucture(raw_input,
                                        instruction,
                                        tokenizer,
                                        max_seq_len=4096):
    input = remove_extra_spaces(remove_urls(raw_input))
    length_predefined_prompt = len(tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

    # Tokenize the input and truncate to the max_seq_len
    tokens = tokenizer.tokenize(input)[:max_seq_len-length_predefined_prompt-50]  # Reserving space for an EOS token, if necessary
    cleaned_input = tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

    return f"### Instruction:\n{instruction}\n\n### Input:\n{cleaned_input}\n\n### Response:\n"

def prepare_prompt_no_output(raw_input, tokenizer, max_seq_len=4096):
    input = remove_extra_spaces(remove_urls(raw_input['input']))
    instruction = raw_input['instruction']
    length_predefined_prompt = len(tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

    # Tokenize the input and truncate to the max_seq_len
    tokens = tokenizer.tokenize(input)[:max_seq_len-length_predefined_prompt-50]  # Reserving space for an EOS token, if necessary
    cleaned_input = tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

    return f"[INST]### Instruction:\n{instruction}\n\n### Input:\n{cleaned_input}[/INST]\n\n### Response:\n<s>"

def prepare_prompt(raw_input, tokenizer, max_seq_len=4096):
    input = remove_extra_spaces(remove_urls(raw_input['input']))
    instruction = raw_input['instruction']
    response = raw_input['output']
    length_predefined_prompt = len(tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

    # Tokenize the input and truncate to the max_seq_len
    tokens = tokenizer.tokenize(input)[:max_seq_len-length_predefined_prompt-50]  # Reserving space for an EOS token, if necessary
    cleaned_input = tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

    return f"[INST]### Instruction:\n{instruction}\n\n### Input:\n{cleaned_input}[/INST]\n\n### Response:\n<s>{response}</s>\n"

def create_prompt_no_answer(row, tokenizer, max_seq_len=4096):
    return {"text": prepare_prompt_no_output(row, tokenizer, max_seq_len=max_seq_len)}

def prepare_prompt_batch(examples, tokenizer, max_seq_len=500):

    instruction = examples["instruction"][0]  # shared instruciton
    length_predefined_prompt = len(tokenizer.tokenize(f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"))

    # return output_text
    output_text = []
    for i in range(len(examples["instruction"])):
        input_text = examples["input"][i]
        response = examples["output"][i]

        input_text = remove_extra_spaces(remove_urls(input_text))  # Assuming 'input' is a key in the row dictionary
        tokens = tokenizer.tokenize(input_text)[:max_seq_len - length_predefined_prompt - 50]  # Reserving space for an EOS token, if necessary
        cleaned_input = tokenizer.convert_tokens_to_string(tokens)  # Convert tokens back to a string

        text = f'''[INST]### Instruction:{instruction}\n\n### Input:{cleaned_input}[/INST]\n\n### Response:\n<s>{response}</s>\n'''

        output_text.append(text)

    return output_text

def pad_eos(ds):
    EOS_TOKEN = "</s>"
    return [f"{row['output']}{EOS_TOKEN}" for row in ds]



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

