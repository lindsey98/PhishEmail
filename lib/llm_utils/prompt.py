import json
from openai import OpenAI
import time
from typing import Callable, Union, Tuple, Optional
import os

def chat_completion(model_name: str,
                    filled_content: Union[str, Tuple[str]],
                    prompt_template: Callable[[str], str],
                    functions: Callable[[None], str] = None,
                    function_name: Union[None, str] = None,
                    temperature: float = 0,
                    top_p: float = 1
                    ):
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

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    answer = ''

    if len(filled_content):
        prompt = prompt_template(filled_content)
        inference_done = False
        while not inference_done:
            try:
                if function_name:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                        functions=functions,
                        function_call={"name": function_name}, # force the model to call this function
                        temperature=temperature,
                        top_p=top_p
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=500
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
            answer = ''.join([choice.message.content for choice in response.choices])

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
        with open('./lib/llm_utils/identity_recognition_prompt.json', 'rb') as handle:
            context = json.load(handle)

        fresh_content = {"role": "user", "content": f"{filled_content}"}
        context.append(fresh_content)

        return context

    def ask_identity_internal_only(filled_content:str) -> str:
        ""
        with open('./lib/llm_utils/identity_recognition_internal_prompt.json', 'rb') as handle:
            context = json.load(handle)

        fresh_content = {"role": "user", "content": f"{filled_content}"}
        context.append(fresh_content)

        return context

    def generate_email(filled_content: Tuple[str]) -> str:
        ""
        sender_role = filled_content[0]
        recipient_role = filled_content[1]
        industry_sector = filled_content[2]
        context = [
            {
                "role": "user",
                "content": f"Generate an email that imitates the {sender_role} to send out email to a/an {recipient_role} from the same company, both of you are working in a/an {industry_sector} company. Don't mention any company name, you should implicitly show that they work in the same company. Signature the email with your generated sender person name. Your generated email should follow the standard format of a complete .eml file, e.g. From: randomly_generated_sender_person_name <random_sender_address> \n To: randomly_generated_random_recipient_person_name <randomly_generated_recipient_address_sharing_the_same_domain_as_sender> \n Subject: randomly_generated_subject \n Date: today's_date \n\n randomly_generated_email_body. Be more creative and diverse."
            },
        ]

        return context

