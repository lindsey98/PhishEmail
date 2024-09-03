from openai import OpenAI
import os
import json
import random
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

AUGMENTATION_PROB = 0.5
MAX_TOKENS = 500
GPT_PROMPT = '''
Paraphrase the following list of phrases with a special focus on diversifying the sentence patterns and action verbs. The input will consist of phrases separated by a newline, and your output should be in JSON format. Ensure that the paraphrased output significantly differs from the original!
For example, transform phrases like "click the link to do something" into variations such as "to learn more about" "if you are interested" "to discover more" "for more information" "for more details" or "find out how to".
The number of output phrases should be equal to the number of input phrases. 

Example input: \n <phrase_1>
Example output: \n {{"phrases": [{{"original": <phrase_1>, "paraphrased": "<paraphrased_phrase_1>"}}]}}

Example input: \n <phrase_1> \n <phrase_2> \n <phrase_3>
Example output: \n {{"phrases": [{{"original": <phrase_1>, "paraphrased": "<paraphrased_phrase_1>"}}, {{"original": <phrase_2>, "paraphrased": "<paraphrased_phrase_2>"}}, {{"original": <phrase_3>, "paraphrased": "<paraphrased_phrase_3>"}}]}}

Input:
{phrase_list}

Output:
'''
GPT_PROMPT_TOKENS = len(GPT_PROMPT) / 4
GPT_MODEL = 'gpt-3.5-turbo'
CHUNK_SIZE = 10  # Define your desired chunk size here
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()
gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def gpt_request(phrase_list_chunk):
    prompt = GPT_PROMPT.format(phrase_list='\n'.join(phrase_list_chunk))

    completion = gpt_client.chat.completions.create(
        model=GPT_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


def send_request(phrase_list):
    results = []
    max_length = len(phrase_list)
    for i in range(0, max_length, CHUNK_SIZE):
        chunk = phrase_list[i:min(i + CHUNK_SIZE, max_length)]
        while True:
            try:
                chunk_results = json.loads(gpt_request(chunk))['phrases']
                if len(chunk_results) != len(chunk):
                    raise Exception(f"Missing phrases in output (Expected: {len(chunk)}, Got: {len(chunk_results)})")
                for entry in chunk_results:
                    if 'original' not in entry or 'paraphrased' not in entry:
                        raise Exception(f"Wrong formatting for {entry}")
                results.extend(chunk_results)
                break
            except Exception as e:
                print(e)
    return results


def paraphrase_entries(data):
    phrase_list = []
    paraphrased_results = []
    for _, _, phrase in data:
        phrase_list.append(phrase)
        paraphrased_results.extend(send_request(phrase_list))
        phrase_list = []

    if phrase_list:
        paraphrased_results.extend(send_request(phrase_list))
    return paraphrased_results


def process_entries(data):
    global seen_texts
    seen_texts = set()
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
            print(f"Discrepancy: {phrase}, {paraphrase['original']}")
        if phrase in data[idx]['text']:
            data[idx]['annotations'][annot_idx]['text'] = paraphrase['paraphrased']
            data[idx]['text'] = data[idx]['text'].replace(phrase, paraphrase['paraphrased'])
        else:
            print(f"Phrase not found in text: {data[idx]}")


seen_texts = set()
# random.seed(42)

# dataset_1_name = "spam_archive_2023"
# # need to encode to utf-8 in windows
# with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_all.json', 'r', encoding="utf-8") as json_file:
#     data_1 = json.load(json_file)
# processed_phrases = process_entries(data_1)
# paraphrased_phrases = paraphrase_entries(processed_phrases)
# align_entries(processed_phrases, paraphrased_phrases, data_1)
# with open(f'./datasets/{dataset_1_name}_unique_annotation/annotated_all_augmented_{AUGMENTATION_PROB * 100}.json',
#           'w', encoding="utf-8") as outfile:
#     json.dump(data_1, outfile, indent=4)
#
# dataset_2_name = "Nazario_2005"
# with open(f'./datasets/{dataset_2_name}_unique_annotation/annotated_all.json', 'r', encoding="utf-8") as json_file:
#     data_2 = json.load(json_file)
# processed_phrases = process_entries(data_2)
# paraphrased_phrases = paraphrase_entries(processed_phrases)
# align_entries(processed_phrases, paraphrased_phrases, data_2)
# with open(f'./datasets/{dataset_2_name}_unique_annotation/annotated_all_augmented_{AUGMENTATION_PROB * 100}.json',
#           'w', encoding="utf-8") as outfile:
#     json.dump(data_2, outfile, indent=4)

# dataset_3_name = "annotated_datasets_from_paul"
# with open(f'./datasets/{dataset_3_name}/annotated_all.json', 'r', encoding="utf-8") as json_file:
#     data_3 = json.load(json_file)
# processed_phrases = process_entries(data_3)
# paraphrased_phrases = paraphrase_entries(processed_phrases)
# align_entries(processed_phrases, paraphrased_phrases, data_3)
# with open(f'./datasets/{dataset_3_name}/annotated_all_augmented_{AUGMENTATION_PROB * 100}.json', 'w',
#           encoding="utf-8") as outfile:
#     json.dump(data_3, outfile, indent=4)

dataset_4_name = "spam_archive_2023"
with open(f'./datasets/{dataset_4_name}_unique_annotation/annotated_jiafan.json', 'r', encoding="utf-8") as json_file:
    data_4 = json.load(json_file)
processed_phrases = process_entries(data_4)
paraphrased_phrases = paraphrase_entries(processed_phrases)
align_entries(processed_phrases, paraphrased_phrases, data_4)
with open(
        f'./datasets/{dataset_4_name}_unique_annotation/annotated_jiafan_augmented_{AUGMENTATION_PROB * 100}.json',
        'w', encoding="utf-8") as outfile:
    json.dump(data_4, outfile, indent=4)


