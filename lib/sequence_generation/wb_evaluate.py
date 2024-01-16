import os
from wandb import Api
import torch
from lib.llm.wb_log_dataset import prompt_input, pad_eos, formatting_prompts_func
from datasets import load_dataset, load_metric
import wandb
from types import SimpleNamespace
import pandas as pd
import wandb
import torch
from tqdm.auto import tqdm
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
from torch.nn import CrossEntropyLoss
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import math
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


def _generate(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    with torch.inference_mode():
        t0 = perf_counter()
        output = model.generate(input_ids=tokenized_prompt,
                                generation_config=gen_config)
        generation_ids = output[0][len(tokenized_prompt[0]):]
        num_gen_tokens = len(generation_ids)
        total_time = perf_counter() - t0
        generation = tokenizer.decode(generation_ids, skip_special_tokens=True)
        return dict(generation=generation, generation_ids=generation_ids.tolist(), total_time=total_time,
                    num_gen_tokens=num_gen_tokens)


def evaluate(examples, model, tokenizer, gen_config):
    sample_out = _generate(examples[0]["prompt"], model, tokenizer, gen_config)
    columns = ["prompt", "label"] + list(sample_out.keys()) + ["temperature", "max_new_tokens"]
    data = []
    for example in tqdm(examples, leave=False):
        prompt, label = example["prompt"], example["output"]
        output = _generate(prompt, model, tokenizer, gen_config)
        data.append((prompt, label, *list(output.values()), gen_config.temperature, gen_config.max_new_tokens))
    return data, columns

@torch.inference_mode()
def metric_eval(eval_dataloader, model):
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
    ppl = torch.exp(torch.stack(nlls).mean())
    print(ppl)

    return ppl

if __name__ == '__main__':

    #
    # gen_config = GenerationConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
    #
    # # Prepare the evaluation dataset using the formatting function
    # formatted_eval_dataset = formatting_prompts_func_no_out(test_dataset)
    # # Generate predictions for the formatted evaluation dataset
    #
    # for prompt in formatted_eval_dataset:
    #     predictions = generate(prompt, tokenizer, model, gen_config)
    #     print(predictions)

    TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float}

    config = SimpleNamespace(
        wandb_project='spamarchieve_ft',
        use_cache=True,
        device_map="auto",
        ## Generation params
        num_eval_samples=50,  # we only run on the first samples of the eval_dataset
        temperature=0.001,
        max_new_tokens=256,  # how many tokens to generate as a response
    )

    api = Api()
    artifact = api.artifact('lindsey98/spamarchieve_ft/spamarchieve_gpt_splitted:latest', type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]


    # Load the tokenizer and the model
    checkpoint_path = './output/checkpoint-19900'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_path,
                                                     use_cache=config.use_cache,
                                                     device_map=config.device_map)
    model.eval()

    ''''''
    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    eval_args = TrainingArguments(
        output_dir='./debug',
        per_device_eval_batch_size=1,
        dataloader_drop_last=False
    )
    trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        packing=False,
        max_seq_length=4096,
        formatting_func=formatting_prompts_func,
    )

    eval_loader = trainer.get_eval_dataloader()
    metric_eval(eval_loader, model) # perplexity=1.1714, the lower the better, the best is 1

    '''Visualize some generation results'''
    # wandb.init(project='spamarchieve_ft',
    #            job_type="inference",
    #            config=config)
    # config = wandb.config

    train_prompts = [prompt_input(row) for row in train_dataset]
    eval_prompts = [prompt_input(row) for row in eval_dataset]

    train_outputs = pad_eos(train_dataset)
    eval_outputs = pad_eos(eval_dataset)

    train_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
    eval_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]
    #
    gen_config = GenerationConfig.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                  temperature=config.temperature,
                                                  max_new_tokens=config.max_new_tokens)
    config.gen_config = gen_config  # we save this
    # wandb.config.update(config)

    print("Running inference on your model!")
    # eval_samples = eval_dataset[:config.num_eval_samples]
    eval_samples = eval_dataset[794:795]
    data, columns = evaluate(eval_samples, model=model, tokenizer=tokenizer, gen_config=gen_config)

    df = pd.DataFrame(data=data, columns=columns)
    # print(df.head())
    # df.to_csv(f"{wandb.run.id}_results.csv")
    # table = wandb.Table(dataframe=df)
    # wandb.log({"eval_predictions": table})

