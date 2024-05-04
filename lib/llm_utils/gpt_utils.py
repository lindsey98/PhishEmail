import openai
from openai import OpenAI
import time

def assistant_completion(client, query, assistant_id):
    # Step 1: create new thread
    thread = client.beta.threads.create()
    # Step 2: send user message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )
    # Step 3: specify the assistant to call
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    # Step 4: Waits for the run to be completed.
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id,
                                                       run_id=run.id)
        if run_status.status in ["completed", "failed"]:
            if run_status.status == "failed":
                print("Run failed:", run_status.last_error)
            break
        time.sleep(2)  # wait for 2 seconds before checking again

    # Step 5: Parse the Assistant's Response to Print the Results
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Print only messages from the assistant, with the latest message at the bottom
    response = ""
    for message in reversed(messages.data):
        if message.role == "assistant":
            response = ' '.join([content.text.value for content in message.content if content.type == 'text'])
    return response
