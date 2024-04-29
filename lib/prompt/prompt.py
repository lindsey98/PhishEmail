import json
import openai
import time
from typing import Callable, Union

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


class PromptClass:

    def ask_identity(filled_content:str) -> str:
        ""
        with open('./lib/prompt/identity_recognition_prompt.json', 'rb') as handle:
            context = json.load(handle)

        fresh_content = {"role": "user", "content": f"{filled_content}"}
        context.append(fresh_content)

        return context

    def ask_relation(filled_content:str) -> str:
        ""
        with open('./lib/prompt/relationship_inference_prompt.json', 'rb') as handle:
            context = json.load(handle)

        fresh_content = {"role": "user",
                         "content": f"{filled_content}"}
        context.append(fresh_content)

        return context


class FunctionCall:
    ner_gpt_functions =  \
    [
        {
            "name": "find_ner_organization",
            "description": "Extracts named entities (in particular 'organizations') from the input text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {
                                    "type": "integer",
                                    "description": "The starting position of the recognized organization."
                                },
                                "end": {
                                    "type": "integer",
                                    "description": "The ending position of the recognized organization."
                                },
                                "organization": {
                                    "type": "string",
                                    "description": "A Named entity (organization) extracted from text."
                                }
                            }
                        }
                    }
                }
            },
        "required": ["entities"]  # 'required' inside 'parameters'
        }
    ]

    action_gpt_functions = \
        [
            {
                "name": "find_action",
                "description": "Finds the sentences that are instructions to the message recipients.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "instruction": {
                                        "type": "string",
                                        "description": "Detected instruction."
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["entities"]  # 'required' inside 'parameters'
            }
        ]

