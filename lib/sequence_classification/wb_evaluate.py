import shutil

from datasets import load_dataset
from wandb import Api
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline
)
from lib.sequence_classification.wb_finetune import preprocess_function, compute_metrics
from functools import partial
import json
import random
import re
import numpy as np
import ast
import os
from transformers.integrations import WandbCallback
import wandb
from tqdm import tqdm
import torch
from torch import device as torch_device
import torch.nn.functional as F

def get_background_color(score):
    if score > 0.8:
        return "#FFD700"  # Dark yellow for high scores
    elif score > 0.6:
        return "#FFFACD"  # Lighter yellow for medium scores
    else:
        return "#FFFFE0"  # Very light yellow for lower scores

class LLMFilter(WandbCallback):
    def __init__(self, trainer, test_dataset, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.test_dataset = test_dataset
        self.model, self.tokenizer = trainer.model, trainer.tokenizer

    @torch.inference_mode()
    def classify(self, text):
        tokenized_prompt = self.tokenizer(text, return_tensors='pt',
                                          truncation=True, max_length=self.tokenizer.model_max_length)['input_ids'].cuda()
        logits = self.model(tokenized_prompt).logits
        predicted_class_id = logits.argmax().item()
        return self.model.config.id2label[predicted_class_id]

    def wrong_table(self, examples):
        records_table = wandb.Table(columns=["text", "predicted class", "ground-truth class"])
        for example in tqdm(examples, leave=False):
            text = example["text"]
            gt_label = self.model.config.id2label[example["label"]]
            pred_label = self.classify(text=text)
            if gt_label != pred_label:
                records_table.add_data(text, pred_label, gt_label)
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.wrong_table(self.test_dataset)
        self._wandb.log({"wrong_predictions": records_table})

if __name__ == '__main__':
    api = Api()
    artifact = api.artifact('lindsey98/spamarchieve_instruction/spamarchieve_instruction_splitted_spamarchieve:latest',
                            type='dataset')
    dataset_dir = artifact.download()
    ds = load_dataset("json", data_dir=dataset_dir)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    checkpoint = "./output_instruction/checkpoint-3604"
    id2label = {0: "Non-instruction", 1: "Instruction"}
    label2id = {"Non-instruction": 0, "Instruction": 1}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = ds.map(partial(preprocess_function, tokenizer=tokenizer), batched=True)
    tokenized_datasets = tokenized_dataset.rename_column("label", "labels")


    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )
    model.eval()

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    # eval_args = TrainingArguments(
    #     output_dir='./debug',
    #     per_device_eval_batch_size=16,
    #     dataloader_drop_last=False
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=eval_args,
    #     tokenizer=tokenizer,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics
    # )

    '''Log training code'''
    # wandb.init(project="spamarchieve_instruction", job_type='test')
    # artifact = wandb.Artifact("training_code", type="code")
    # artifact.add_file("./lib/sequence_classification/wb_finetune.py")  # For a single file
    # wandb.log_artifact(artifact)
    # wandb.finish()

    '''Evaluation metrics'''
    # wandb.init(project="spamarchieve_instruction", job_type='test')
    # wandb_callback = LLMFilter(trainer, eval_dataset)
    # trainer.add_callback(wandb_callback)
    # trainer.evaluate()
    # wandb.finish()
    # exit()
    # #     eval/accuracy 0.96463
    # #        eval/f1 0.92587
    # #      eval/loss 0.1622
    # # eval/precision 0.91235
    # #    eval/recall 0.93981
    
    '''Classify examples'''
    data_inst = []
    dataset_file_inst = "./datasets/spamarchieve-instruction-annot-2023.jsonl"
    result_dir = './datasets/spamarchieve-eval-instruction-pred/'
    # dataset_file_inst = "./datasets/iwspa-data.json"
    # result_dir = './datasets/iwspa-eval-instruction-pred/'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    with open(dataset_file_inst, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data_inst.append(entry)

    random.seed(1234)
    random.shuffle(data_inst)  # shuffle inplace
    train_dataset_inst = data_inst[:-int(len(data_inst) * 0.2)]
    eval_dataset_inst = data_inst[-int(len(data_inst) * 0.2):]

    pipe = pipeline(task="text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device="cuda")

    for it, d_inst in enumerate(eval_dataset_inst):
        body = d_inst["input"]
        all_sentences = re.split(r'(?<=[,.])\s', body) # sub-sequences, split by . or ,
        output = d_inst["output"].replace('\\"', "'").replace("['", '["').replace("']", '"]').replace("', '", '", "')
        try:
            instruction_sentences = ast.literal_eval(output)
        except:
            continue

        # body = d_inst["text"]
        # all_sentences = re.split(r'(?<=[,.])\s', body) # sub-sequences, split by . or ,
        batch_size = 16
        results = []

        # Process sentences in batches
        for sent in all_sentences:
            tokenized_prompt = tokenizer(sent,
                                         return_tensors='pt',
                                         truncation=True,
                                         max_length=tokenizer.model_max_length)['input_ids'].cuda()
            logits = model(tokenized_prompt).logits
            scores = F.softmax(logits, dim=1)
            score = scores.data[:, 1].item()
            predicted_class_id = logits.argmax().item()
            predicted_class = model.config.id2label[predicted_class_id]
            results.append({'label': predicted_class, 'score': score})

        predicted_instruction_sentences = np.asarray(all_sentences)[np.asarray([x['label'] == 'Instruction' for x in results])]
        instruction_scores = [output['score'] if output['label'] == 'Instruction' else 0 for output in results]
        sorted_indices = np.argsort(instruction_scores)[::-1]  # Sort in descending order

        # Highlight the top-1 predicted instruction sentence in yellow
        html_content = ""
        for i, sentence in enumerate(all_sentences):
            if results[i]['label'] == 'Instruction':
                color = get_background_color(results[i]['score'])
                html_content += f"<span style='background-color:{color}'>{sentence}</span> "
            else:
                html_content += f"{sentence} "

        with open(f"{result_dir}/{it}.html", "w") as html_file:
            html_file.write(html_content)

        print()