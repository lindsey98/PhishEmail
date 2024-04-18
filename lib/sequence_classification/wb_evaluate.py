import os
from wandb import Api
import torch
from datasets import load_dataset, load_metric
import wandb
from types import SimpleNamespace
import pandas as pd
import wandb
import torch
from tqdm.auto import tqdm
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from lib.data.utils import prepare_prompt_batch, prepare_prompt, create_prompt_no_answer, pad_eos, prepare_prompt_no_output
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM
import math
import numpy as np
import re
import evaluate
from functools import partial
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def _generate(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    with torch.inference_mode():
        t0 = perf_counter()
        output = model.generate(input_ids=tokenized_prompt,
                                generation_config=gen_config)
        total_time = perf_counter() - t0
        generation_ids = output[0][len(tokenized_prompt[0]):]
        num_gen_tokens = len(generation_ids)
        generation = tokenizer.decode(generation_ids, skip_special_tokens=True)
        return dict(generation=generation, generation_ids=generation_ids.tolist(), total_time=total_time,
                    num_gen_tokens=num_gen_tokens)


# def evaluate(examples, model, tokenizer, gen_config):
#     sample_out = _generate(examples[0]["prompt"], model, tokenizer, gen_config)
#     columns = ["prompt", "label"] + list(sample_out.keys()) + ["temperature", "max_new_tokens"]
#     data = []
#     for example in tqdm(examples, leave=False):
#         prompt, label = example["prompt"], example["output"]
#         output = _generate(prompt, model, tokenizer, gen_config)
#         data.append((prompt, label, *list(output.values()), gen_config.temperature, gen_config.max_new_tokens))
#     return data, columns

@torch.inference_mode()
def compute_perplexity(eval_dataloader, model):
    model.eval()
    nlls = []

    for i, batch in tqdm(enumerate(eval_dataloader)):
        if (batch["labels"] == -100).all():
            continue  # Skip this batch
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            neg_log_likelihood = out.loss

        nlls.append(neg_log_likelihood)

    # Log results at the end
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl

def split_by_step(results):
    step_1_match = re.search(r"(Step 1:.*?)(?=Step 2)", results, re.DOTALL)
    step_1_text = step_1_match.group(1).strip() if step_1_match else ''

    step_2_match = re.search(r"(Step 2:.*?)(?=Step 3)", results, re.DOTALL)
    step_2_text = step_2_match.group(1).strip() if step_2_match else ''

    step_3_match = re.search(r"(Step 3:.*?)(?=\n|(</s>))", results, re.DOTALL)
    step_3_text = step_3_match.group(1).strip() if step_3_match else ''

    return step_1_text, step_2_text, step_3_text


if __name__ == '__main__':

    TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float}
    config = SimpleNamespace(
        dataset="spamarchieve",
        wandb_project='spamarchieve_ft',
        checkpoint_path='./checkpoints/output_llama2_lora_relation/checkpoint-98fmxipa:v1',
        use_cache=False,
        device_map="auto",
        num_eval_samples=50,  # we only run on the first samples of the eval_dataset
        temperature=0.001,
        max_new_tokens=100,  # how many tokens to generate as a response
        max_seq_length=4096,
    )
    gen_config = GenerationConfig.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                  temperature=config.temperature,
                                                  max_new_tokens=config.max_new_tokens,)

    api = Api()
    artifact = api.artifact(f'lindsey98/{config.dataset}_ft/{config.dataset}_gpt_splitted:latest', type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    # artifact = api.artifact(f'lindsey98/spamarchieve_ft/checkpoint-98fmxipa:v1', type='model')
    # artifact_dir = artifact.download()
    # exit()

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(config.checkpoint_path,
                                                     use_cache=config.use_cache,
                                                     device_map=config.device_map)
    model.eval()

    # '''Compute perplexity'''
    response_template = "\n### Response:"  # Note: For llama2, the response_template with context and w/o context are encoded differently. https://huggingface.co/docs/trl/en/sft_trainer#using-tokenids-directly-for-responsetemplate
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # ignore the \n part
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir='./debug',
            per_device_eval_batch_size=1,
            dataloader_drop_last=False
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        packing=False,
        max_seq_length=config.max_seq_length,
        formatting_func=partial(prepare_prompt_batch, tokenizer=tokenizer, max_seq_len=config.max_seq_length),
    )

    eval_loader = trainer.get_eval_dataloader()
    ppl = compute_perplexity(eval_loader, model)
    print('Perplexity = ', ppl) # Perplexity = 1.1016572713851929

    train_prompts = [prepare_prompt_no_output(row, tokenizer, config.max_seq_length) for row in train_dataset]
    eval_prompts = [prepare_prompt_no_output(row, tokenizer, config.max_seq_length) for row in eval_dataset]

    train_outputs = pad_eos(train_dataset)
    eval_outputs = pad_eos(eval_dataset)

    train_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
    eval_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

    response_template = "\n### Response:" # Note: For llama2, the response_template with context and w/o context are encoded differently. https://huggingface.co/docs/trl/en/sft_trainer#using-tokenids-directly-for-responsetemplate
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # ignore the \n part
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    rouge = evaluate.load('rouge') # do pip install rouge_score first

    references = []
    predictions = []
    reference_class = []
    predicted_class = []
    class_map = {'A': 0, 'B': 1, 'Unclear': 2}

    for data in tqdm(eval_dataset, leave=False):
        prompt = data['prompt']
        reference = data['output']
        step1_ref, step2_ref, step3_ref = split_by_step(reference)
        step23_ref = step2_ref + step3_ref

        prediction = _generate(prompt, model, tokenizer, gen_config)['generation']
        step1_pred, step2_pred, step3_pred = split_by_step(prediction)
        step23_pred = step2_pred + step3_pred

        if 'A' in step1_ref:
            mapped_value = class_map['A']
        elif 'B' in step1_ref:
            mapped_value = class_map['B']
        else:
            mapped_value = class_map['Unclear']
        reference_class.append(mapped_value)

        if 'A' in step1_pred:
            mapped_value = class_map['A']
        elif 'B' in step1_pred:
            mapped_value = class_map['B']
        else:
            mapped_value = class_map['Unclear']
        predicted_class.append(mapped_value)

        references.append(step23_ref)
        predictions.append(step23_pred)

    rouge_metric = rouge.compute(predictions=predictions, references=references)
    print('Rouge = ', rouge_metric)
    correct_and_relevant = (np.asarray(reference_class) == np.asarray(predicted_class)) & (np.asarray(reference_class) != 2)
    total_relevant = np.asarray(reference_class) != 2
    classification_accuracy = np.sum(correct_and_relevant) / np.sum(total_relevant) #
    print('Classification accuracy = ', classification_accuracy)
    # Rouge =  {'rouge1': 0.7330069403527175, 'rouge2': 0.6466189876874545, 'rougeL': 0.7235050858701357, 'rougeLsum': 0.7240731321939433}
    # Classification accuracy =  0.8690549722050649
    exit()


    '''Visualize some generation results'''
    # wandb.init(project=f'{dataset}_ft',
    #            job_type="inference")
    #
    #
    # # #
    # config.gen_config = gen_config  # we save this
    #
    # print("Running inference on your model!")
    # eval_samples = eval_dataset[:config.num_eval_samples]
    # data, columns = evaluate(eval_samples, model=model, tokenizer=tokenizer, gen_config=gen_config)
    # #
    # df = pd.DataFrame(data=data, columns=columns)
    # print(df.head())
    # df.to_csv(f"{wandb.run.id}_results.csv")
    # table = wandb.Table(dataframe=df)
    # wandb.log({"eval_predictions": table})
    # exit()

    runtime = []
    response_token_len_list = []
    pbar = tqdm(eval_prompts[:500], leave=False)

    for data in pbar:
        t0 = perf_counter()
        results = _generate(data, model, tokenizer, gen_config)
        results = results['generation']
        print(results)
        total_time = perf_counter() - t0
        runtime.append(total_time)

        step_1_match = re.search(r"(Step 1:.*?)(?=Step 2)", results, re.DOTALL)
        step_1_prediction = step_1_match.group(1).strip() if step_1_match else None

        step_2_match = re.search(r"(Step 2:.*?)(?=Step 3)", results, re.DOTALL)
        step_2_text = step_2_match.group(1).strip() if step_2_match else None

        step_3_match = re.search(r"(Step 3:.*?)(?=\n)", results, re.DOTALL)
        step_3_text = step_3_match.group(1).strip() if step_3_match else None

        if step_1_prediction and step_2_text:
            response_token_len = len(tokenizer(step_1_prediction + step_2_text, return_tensors='pt')['input_ids'].to(model.device)[0])
            response_token_len_list.append(response_token_len)
            print(step_1_prediction + step_2_text)

            pbar.set_description(f"Max response length (token length):  {max(response_token_len_list)}, "
                                 f"Median Runtime: {np.median(runtime)} ", refresh=True)

    print('Max response length (token length): ', max(response_token_len_list))
    print('Median Runtime: ', np.median(runtime))

