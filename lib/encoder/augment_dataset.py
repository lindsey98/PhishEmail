from openai import OpenAI
import os
import json
import random

AUGMENTATION_PROB = 0.25
MAX_TOKENS = 500
GPT_PROMPT = '''
Paraphrase the list of phrases, with special emphasis on the action verbs used. The input will consist of phrases separated by a newline, and you will output the paraphrased phrase in JSON format. Ensure that your paraphrased output is different from the original!
Example input: \n <phrase_1> \n <phrase_2> \n<phrase_3>
Example output: \n {{"phrases": [{{"original": <phrase_1>, "paraphrased": "<paraphrased_phrase_1>"}}, {{"original": <phrase_2>, "paraphrased": "<paraphrased_phrase_2>"}}, {{"original": <phrase_3>, "paraphrased": "<paraphrased_phrase_3>"}}]}}

Input:
{phrase_list}

Output:
'''
GPT_PROMPT_TOKENS = len(GPT_PROMPT) / 4
GPT_MODEL = 'gpt-3.5-turbo'

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()
gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def gpt_request(phrase_list):
    prompt = GPT_PROMPT.format(phrase_list = phrase_list)

    completion = gpt_client.chat.completions.create(
        model=GPT_MODEL,
        response_format={ "type": "json_object" },
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

def send_request(phrase_list):
    while True:
        try:
            results = json.loads(gpt_request('\n'.join(phrase_list)))['phrases']
            if (len(results) != len(phrase_list)):
                raise Exception(f"Missing phrases in output (Expected: {len(phrase_list)}, Got: {len(results)})")
            for entry in results:
                if 'original' not in entry or 'paraphrased' not in entry:
                    raise Exception(f"Wrong formatting for {entry}")
            return results
        except Exception as e:
            print(e)

def paraphrase_entries(data):
    running_tokens = 0
    phrase_list = []
    paraphrased_results = []
    token_budget = MAX_TOKENS - GPT_PROMPT_TOKENS
    for _, _, phrase in data:
        phrase_list.append(phrase)
        running_tokens += len(phrase) / 4
        if token_budget < running_tokens:
            paraphrased_results.extend(send_request(phrase_list))
            phrase_list = []
            running_tokens = 0
    
    if phrase_list:
        paraphrased_results.extend(send_request(phrase_list))
    return paraphrased_results

def process_entries(data):
    global seen_texts
    processed_phrases = []

    for idx, entry in enumerate(data):
        text = entry['text']
        if text in seen_texts:
            continue  # Skip duplicates
        seen_texts.add(text)

        annotations = entry.get('annotations', [])
        for annot_idx, annot in enumerate(annotations):
            if annot['labels'][0] == 'action' and random.random() < AUGMENTATION_PROB:
                processed_phrases.append((idx, annot_idx, annot['text']))
        
    return processed_phrases

def align_entries(phrases, paraphrases, data):
    for (idx, annot_idx, phrase), paraphrase in zip(phrases, paraphrases):
        if phrase != paraphrase['original']:
            print(f"discrepency: {phrase}, {paraphrase['original']}")
        data[idx]['annotations'][annot_idx]['text'] = paraphrase['paraphrased']
        data[idx]['text'] = data[idx]['text'].replace(phrase, paraphrase['paraphrased'])

seen_texts = set()
random.seed(42)

dataset_1_name = "spam_archive_2023"
# need to encode to utf-8 in windows
with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_all.json', 'r', encoding="utf-8") as json_file:
    data_1 = json.load(json_file)
    processed_phrases = process_entries(data_1)
    paraphrased_phrases = paraphrase_entries(processed_phrases)
    align_entries(processed_phrases, paraphrased_phrases, data_1)
    with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_all_augmented_{AUGMENTATION_PROB * 100}.json', 'w', encoding="utf-8") as outfile:
        json.dump(data_1, outfile, indent=4)

dataset_2_name = "Nazario_2005"
with open(f'./datasets/{dataset_2_name}_unique_annotation/annotated_all.json', 'r', encoding="utf-8") as json_file:
    data_2 = json.load(json_file)
    processed_phrases = process_entries(data_2)
    paraphrased_phrases = paraphrase_entries(processed_phrases)
    align_entries(processed_phrases, paraphrased_phrases, data_2)
    with open(f'./datasets/{dataset_2_name}_unique_annotation/annotated_all_augmented_{AUGMENTATION_PROB * 100}.json', 'w', encoding="utf-8") as outfile:
        json.dump(data_2, outfile, indent=4)

dataset_3_name = "annotated_datasets_from_paul"
with open(f'./datasets/{dataset_3_name}/annotated_all.json', 'r', encoding="utf-8") as json_file:
    data_3 = json.load(json_file)
    processed_phrases = process_entries(data_3)
    paraphrased_phrases = paraphrase_entries(processed_phrases)
    align_entries(processed_phrases, paraphrased_phrases, data_3)
    with open(f'./datasets/{dataset_3_name}/annotated_all_augmented_{AUGMENTATION_PROB * 100}.json', 'w', encoding="utf-8") as outfile:
        json.dump(data_3, outfile, indent=4)

dataset_4_name = "spam_archive_2023"
with open(f'./datasets/{dataset_4_name}_unique_annotation/annotated_jiafan.json', 'r', encoding="utf-8") as json_file:
    data_4 = json.load(json_file)
    processed_phrases = process_entries(data_4)
    paraphrased_phrases = paraphrase_entries(processed_phrases)
    align_entries(processed_phrases, paraphrased_phrases, data_4)
    with open(f'./datasets/{dataset_4_name}_unique_annotation/annotated_jiafan_augmented_{AUGMENTATION_PROB * 100}.json', 'w', encoding="utf-8") as outfile:
        json.dump(data_4, outfile, indent=4)


    