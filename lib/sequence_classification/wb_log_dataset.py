
import json
import random
import re
import ast
import wandb

def create_ner_labels(sentence, phrase, label_type):
    sentence = sentence.strip('.').strip()
    phrase = phrase.strip('.').strip()
    words4sentence = sentence.split()
    phrase4sentence = phrase.split()
    labels = []

    for word in words4sentence:

        if len(phrase4sentence)>0 and word == phrase4sentence[0]:
            if len(labels) == 0 or labels[-1][1] == 'O':
                labels.append((word, f'B-{label_type}'))
            else:
                labels.append((word, f'I-{label_type}'))
            phrase4sentence.pop(0)

        else:
            labels.append((word, 'O'))

    return labels



def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def remove_urls(text):
    pattern = r'\(?(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)\)?'
    return re.sub(pattern, '', text)

def have_overlap_over_half(sentence1, sentence2):
    # Split sentences into words and create sets
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())

    # Calculate the overlap
    overlap = words1.intersection(words2)

    # Check if overlap is more than half of the length of the shorter sentence
    min_length = min(len(words1), len(words2))
    return len(overlap) > min_length / 2


if __name__ == '__main__':

    data_inst = []
    dataset_file_inst = "./datasets/spamarchieve-instruction-annot-2023.jsonl"

    with open(dataset_file_inst, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data_inst.append(entry)


    random.seed(1234)
    random.shuffle(data_inst)  # shuffle inplace
    train_dataset_inst = data_inst[:-int(len(data_inst) * 0.2)]
    eval_dataset_inst = data_inst[-int(len(data_inst) * 0.2):]

    annot_json = './datasets/spamarchieve-instruction-train.jsonl'
    data = []
    existed_set = set()

    for i, d_inst in enumerate(train_dataset_inst):
        body = d_inst["input"]
        output = d_inst["output"].replace('\\"', "'").replace("['", '["').replace("']", '"]').replace("', '", '", "')
        try:
            instruction_sentences = ast.literal_eval(output)
        except:
            continue

        all_sentences = re.split(r'(?<=[,.])\s', body)
        all_sentences = [s for s in all_sentences \
                             if not any([have_overlap_over_half(out, s) \
                                         for out in instruction_sentences])]

        # Randomly selecting another irrelevant sentence
        random_sentences = []
        if len(instruction_sentences) > 0:
            random_sentences = random.sample(all_sentences, k=min(5, len(all_sentences)))

        for sent in instruction_sentences:
            sent = remove_urls(sent)
            if sent in existed_set or len(sent)<3:
                continue
            new_entry =  {
                "text": sent,
                "label": 1,
            }
            # Append new entry to data
            data.append(new_entry)
            existed_set.add(sent)

        for sent in random_sentences:
            sent = remove_urls(sent)
            if sent in existed_set or len(sent)<3:
                continue
            new_entry =  {
                "text": sent,
                "label": 0,
            }
            # Append new entry to data
            data.append(new_entry)
            existed_set.add(sent)

        # Save data periodically
        if i % 50 == 0:
            with open(annot_json, 'w', encoding='utf-8') as file:
                for item in data:
                    file.write(json.dumps(item, ensure_ascii=False) + '\n')


    annot_json = './datasets/spamarchieve-instruction-eval.jsonl'
    data = []
    existed_set = set()

    for i, d_inst in enumerate(eval_dataset_inst):
        body = d_inst["input"]
        output = d_inst["output"].replace('\\"', "'").replace("['", '["').replace("']", '"]').replace("', '", '", "')
        try:
            instruction_sentences = ast.literal_eval(output)
        except:
            continue
        # output_ident = d_ident["output"]
        #
        # step_1_text = re.search(r'Step 1: .*?\.?Explanation:\s(.*?)\.?\s*\n?\s*Step 2', output_ident)
        # step_2_text = re.search(r'Step 2: .*?\.?Explanation:\s(.*?)\.?\s*\n?\s*Step 3', output_ident)
        #
        # capabilities = step_1_text.group(1) if step_1_text else "Not found"
        # organization = step_2_text.group(1) if step_2_text else "Not found"
        #
        # capabilities = capabilities.split('; ')
        # capabilities = [item.strip("'").strip().strip('.') for item in capabilities]
        # organization = organization.split('; ')
        # organization = [item.strip("'").replace('From:', '').replace('From', '').strip().strip('.') for item in organization]


        # Splitting the text body into sentences
        all_sentences = re.split(r'(?<=[,.])\s', body)
        all_sentences = [s for s in all_sentences \
                             if not any([have_overlap_over_half(out, s) \
                                         for out in instruction_sentences])]

        # Randomly selecting another irrelevant sentence
        random_sentences = []
        if len(instruction_sentences) > 0:
            random_sentences = random.sample(all_sentences, k=min(5, len(all_sentences)))

        for sent in instruction_sentences:
            sent = remove_urls(sent)
            if sent in existed_set or len(sent)<3:
                continue
            new_entry =  {
                "text": sent,
                "label": 1,
            }
            # Append new entry to data
            data.append(new_entry)
            existed_set.add(sent)

        for sent in random_sentences:
            sent = remove_urls(sent)
            if sent in existed_set or len(sent)<3:
                continue
            new_entry =  {
                "text": sent,
                "label": 0,
            }
            # Append new entry to data
            data.append(new_entry)
            existed_set.add(sent)


        # Save data periodically
        if i % 50 == 0:
            with open(annot_json, 'w', encoding='utf-8') as file:
                for item in data:
                    file.write(json.dumps(item, ensure_ascii=False) + '\n')


    # log to wandb
    with wandb.init(project="spamarchieve_instruction"):

        at2 = wandb.Artifact(
            name="spamarchieve_instruction_splitted_spamarchieve",
            type="dataset",
            description="A GPT generated Alpaca like dataset for instruction finetunning",
        )
        at2.add_file("./datasets/spamarchieve-instruction-train.jsonl")
        at2.add_file("./datasets/spamarchieve-instruction-eval.jsonl")
        wandb.log_artifact(at2)