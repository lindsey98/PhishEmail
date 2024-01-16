from transformers import AutoTokenizer
import os
from wandb import Api
import torch
from transformers import AutoModelForCausalLM
from transformers import GenerationConfig
from lib.llm.wb_log_dataset import prompt_input, pad_eos
from datasets import load_dataset, load_metric
import wandb
from types import SimpleNamespace
import pandas as pd
import wandb
import torch
from tqdm.auto import tqdm
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from lib.llm.wb_evaluate import metric_eval, evaluate, formatting_prompts_func
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


if __name__ == '__main__':


    TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float}

    config = SimpleNamespace(
        wandb_project='spamarchieve_ft',
        use_cache=True,
        device_map="auto",
        torch_dtype="bf16",
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
    checkpoint_path = './output_6layer/checkpoint-15920'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
                                                 use_cache=config.use_cache,
                                                 device_map=config.device_map)
    model.eval()

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
    metric_eval(eval_loader, model) # perplexity: 1.2566

    # inference stuff

    # wandb.init(project='spamarchieve_ft',
    #            job_type="inference",
    #            config=config)
    # config = wandb.config
    # train_prompts = [prompt_input(row) for row in train_dataset]
    # eval_prompts = [prompt_input(row) for row in eval_dataset]
    #
    # train_outputs = pad_eos(train_dataset)
    # eval_outputs = pad_eos(eval_dataset)
    #
    # train_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
    # eval_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

    # gen_config = GenerationConfig.from_pretrained('meta-llama/Llama-2-7b-hf',
    #                                               temperature=config.temperature,
    #                                               max_new_tokens=config.max_new_tokens)
    # config.gen_config = gen_config  # we save this
    # wandb.config.update(config)
    #
    # print("Running inference on your model!")
    # eval_samples = eval_dataset[:config.num_eval_samples]
    # data, columns = evaluate(eval_samples, model=model, tokenizer=tokenizer, gen_config=gen_config)
    #
    # df = pd.DataFrame(data=data, columns=columns)
    # print(df.head())
    # df.to_csv(f"{wandb.run.id}_results.csv")
    # table = wandb.Table(dataframe=df)
    # wandb.log({"eval_predictions": table})

