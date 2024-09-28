import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BertGenerationEncoder,
    BertGenerationDecoder,
    EncoderDecoderModel,
    Trainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    BertTokenizer
)
from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def preprocess_function(examples, tokenizer):
    inputs = ["grammar: " + example['noise'] for example in examples["translation"]]
    # inputs = [example['noise'] for example in examples["translation"]]
    targets = [example['clean'] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
    return model_inputs


if __name__ == '__main__':

    model_id = "t5-efficient-tiny"
    student_model = T5ForConditionalGeneration.from_pretrained(f'google/{model_id}')
    tokenizer = T5Tokenizer.from_pretrained(f'google/{model_id}')

    # model_id = "bert-base-uncased"
    # # Initialize the encoder
    # encoder = BertGenerationEncoder.from_pretrained(
    #     f"google-bert/{model_id}",
    #     bos_token_id=101,  # [CLS]
    #     eos_token_id=102  # [SEP]
    # )
    #
    # # Initialize the decoder with cross-attention layers
    # decoder = BertGenerationDecoder.from_pretrained(
    #     f"google-bert/{model_id}",
    #     add_cross_attention=True,
    #     is_decoder=True,
    #     bos_token_id=101,
    #     eos_token_id=102
    # )
    #
    # student_model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    # tokenizer = BertTokenizer.from_pretrained(f"google-bert/{model_id}")
    #
    # # Manually set special tokens
    # tokenizer.bos_token = tokenizer.cls_token
    # tokenizer.eos_token = tokenizer.sep_token
    # tokenizer.pad_token = tokenizer.eos_token  # Use [SEP] as the pad token
    #
    # # Update model configuration
    # student_model.config.pad_token_id = tokenizer.pad_token_id
    # student_model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Load and preprocess dataset
    dataset = load_dataset('./lib/adversary/defence_utils/translation_dataset.py')
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer=tokenizer),
        batched=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=student_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./checkpoints/{model_id}",
        num_train_epochs=30, # Large epochs
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # Effective batch size of 32
        per_device_eval_batch_size=4,
        learning_rate=1e-3, # High learning rate
        warmup_ratio=0.1,  # Use warmup ratio or warmup_steps
        evaluation_strategy='epoch',
        logging_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        do_train=True,
        do_eval=True,
        fp16=True,  # Enable mixed precision training for memory efficiency
        dataloader_pin_memory=True,
        save_total_limit=2, # only save the latest and the best
        load_best_model_at_end=True, # save the best
        save_strategy ="epoch",
        predict_with_generate=True,
        overwrite_output_dir=True
    )

    trainer = Seq2SeqTrainer(
        model = student_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # Start training
    trainer.train()
