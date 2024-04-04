
import openai
import time
from typing import Callable, List, Dict, Optional, Union, Any
import json
from lib.data.download_miller import parse_json
from lib.prompt.prompt import PromptClass, FunctionCall
import os
from tqdm import tqdm
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()


def chat_completion(model_name: str,
                    filled_content: str,
                    prompt_template: Callable[[str], str],
                    functions: Callable[[None], str],
                    function_name: Union[None, str],
                    temperature: float = 0):
    """
    Generates a chat completion using OpenAI's ChatCompletion API.
    :param function_call:
    :param model_name:
    :param filled_content:
    :param prompt_template:
    :param max_tokens:
    :param temperature:
    :return:
    """

    answer = ''

    if len(filled_content):
        prompt = prompt_template(filled_content)
        inference_done = False
        while not inference_done:
            try:
                if function_name:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=prompt,
                        functions=functions,
                        function_call={"name": function_name}, # force the model to call this function
                        temperature=temperature
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=prompt,
                        temperature=temperature
                    )
                inference_done = True
            except Exception as e: # too long
                print(f"Error was: {e}")
                prompt_len = len(prompt[-1]['content'])
                prompt[-1]['content'] = prompt[-1]['content'][:prompt_len//2]
                time.sleep(30)

        if function_name:
            answer = response['choices'][0]['message']['function_call']['arguments']
        else: # normal response
            answer = ''.join([choice["message"]["content"] for choice in response['choices']])

    print(answer)
    return answer

def truncate_json_string(json_string):
    # Find the last occurrence of '},'
    last_object_end = json_string.rfind('},')

    # If '},' is found, truncate the string up to that position and close the JSON structure
    if last_object_end != -1:
        truncated_string = json_string[:last_object_end + 1]
        # Close the array and the JSON object
        truncated_string += "\n  ]\n}"
        return truncated_string
    else:
        # If '},' is not found, return the original string
        return json_string

if __name__ == '__main__':

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890"  # proxy
    model_name = "gpt-3.5-turbo-16k"

    result_json = './datasets/spamarchieve-annot.jsonl'
    # result_json = './datasets/miller-annot.jsonl'

    with open(result_json, 'r', encoding='utf-8') as file:
        data = file.read()
    # parsed_json_objects = parse_json(data)
    parsed_json_objects = parse_json(data, delimiter='{"Id":')

    # automatically identify named entity
    for i in tqdm(range(len(parsed_json_objects))): #
        if "label" in parsed_json_objects[i]:
            continue
        answer = chat_completion(model_name=model_name, filled_content=parsed_json_objects[i]['text'],
                                prompt_template=PromptClass.ask_orgization,
                                functions = FunctionCall.ner_gpt_functions,
                                 function_name = "find_ner_organization")

        if 'entities' in answer:
            try:
                annot = [[x['start'], x['end'], x['organization']] for x in eval(answer)['entities']]
            except SyntaxError:
                try:
                    answer = truncate_json_string(answer)
                    annot = [[x['start'], x['end'], x['organization']] for x in eval(answer)['entities']]
                except SyntaxError:
                    annot = []
        else:
            annot = []
        parsed_json_objects[i]["label"] = annot # list of dict

        if i % 50 == 0:
            # Save to a new JSON file
            with open(result_json, 'w', encoding='utf-8') as f:
                for entry in parsed_json_objects:
                    json.dump(entry, f)
                    f.write('\n')

    # Save to a new JSON file
    with open(result_json, 'w', encoding='utf-8') as f:
        for entry in parsed_json_objects:
            json.dump(entry, f)
            f.write('\n')
