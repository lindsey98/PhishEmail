import json
import os.path
from .visualizer import Visualizer, TracInVisualizer
import pandas as pd
import shutil
from ..data import OCR, RenderDataset
import torch
import datasets
from .model_utils.preprocessing import tokenize_and_align_labels
from .model_utils.trainer import BertTrainer_FocalLoss
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
from transformers import TrainingArguments
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def align_words_and_tokens(example):
    inputs = tokenizer(
        example,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        return_offsets_mapping=True  # Return character-level offsets
    )

    # Extract inputs
    input_ids = inputs["input_ids"][0]  # Token IDs (first sequence)
    offset_mapping = inputs["offset_mapping"][0]  # Offset mapping (first sequence)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Tokens as strings

    # Original words
    words = example.split()

    # Map each word to its token index (accounting for special tokens like [CLS])
    word_to_token_indices = {}
    for word in words:
        start, end = example.find(word), example.find(word) + len(word)  # Char range of word
        aligned_indices = [
            idx for idx, (token_start, token_end) in enumerate(offset_mapping)
            if token_start >= start and token_end <= end and tokens[idx] != '[CLS]' and tokens[idx] != '[SEP]'
        ]
        word_to_token_indices[word] = aligned_indices

    return word_to_token_indices

if __name__ == '__main__':

    label_list = [
        "O",
        "B-identity",
        "I-identity",
        "B-relation",
        "I-relation",
        "B-action",
        "I-action",
    ]
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}
    action_cls = label_to_id["B-action"]
    identity_cls = label_to_id["B-identity"]

    pretrained_checkpoint = "./checkpoints/output_ner_augmented_corrected/checkpoint-1540"
    gradient_logging_file_action = "./datasets/ner_adversarial_training_grads_action.pth"
    gradient_logging_file_identity = "./datasets/ner_adversarial_training_grads_identity.pth"
    rootDir = "./datasets/phishpot"
    dumpDir = "./datasets/phishpot_imgs/"
    visDir = "./datasets/phishpot_fn_round4"
    token_alignment_file = "./datasets/phishpot_token_alignment_round4.json"

    '''Compare 2 preprocessing solutions'''
    # ocr = OCR()
    # dataset = RenderDataset(rootDir, ocr_model=ocr, dumpDir=dumpDir)
    # for i in range(len(dataset)):
    #     if dataset.file_list[i] not in ["./datasets/phishpot/sample-69.eml"]:
    #         continue
    #     email_file_path, (sender_name, sender_address), \
    #                 (to_names, to_addresses), reply_to_address, \
    #             subject, email_body_text, header = dataset[i]
    #     #
    # exit()

    '''Log training samples gradients'''
    dataset_dir = f'./datasets/ner_adversarial_training/'
    ds = datasets.load_dataset("json", data_files={"train": dataset_dir + "train.json", "test": dataset_dir + "test.json"})
    train_dataset, test_dataset = ds["train"], ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    tokenized_wnut = ds.map(lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_checkpoint, num_labels=7, id2label=id_to_label, label2id=label_to_id
    )
    training_args = TrainingArguments( output_dir="debug",  per_device_train_batch_size=1 )
    trainer = BertTrainer_FocalLoss(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer, data_collator=data_collator,
    )
    #
    # exact_file_paths = trainer.train_dataset["metadata"]
    # train_data_loader = trainer.get_train_dataloader_no_shuffle()
    # training_gradients_dict = trainer.batch_gradient(train_data_loader, cls_of_interest=action_cls, metadata=exact_file_paths)
    # torch.save(training_gradients_dict, gradient_logging_file_action)
    #
    # training_gradients_dict = trainer.batch_gradient(train_data_loader, cls_of_interest=identity_cls, metadata=exact_file_paths)
    # torch.save(training_gradients_dict, gradient_logging_file_identity)

    '''Prepare the token alignment files to find the interesting token indices'''
    ocr = OCR()
    dataset = RenderDataset(rootDir, ocr_model=ocr, dumpDir=dumpDir)
    # # Load the list of email paths to process
    # dataset_name = "CSDMC2010_benign"
    dataset_name = "nazario"
    # df = pd.read_csv("./datasets/CSDMC2010_benign_results.csv")
    # df = pd.read_csv("./datasets/phishpot_results.csv")
    df = pd.read_csv("./datasets/nazario_results.csv")
    df = df.drop_duplicates(subset='email_file_path')
    print(f'Original # Emails = {len(df)}') # 2584

    if os.path.exists(f'./datasets/{dataset_name}_noisy_list.txt'):
        noisy_email_files = [x.strip() for x in open(f'./datasets/{dataset_name}_noisy_list.txt').readlines()]
        df = df[~df['email_file_path'].isin(noisy_email_files)]
    print(f'Clean # Emails = {len(df)}') #

    filtered_df = df.loc[(df["our_pred"] != True)]

    no_instruction = filtered_df.loc[(~filtered_df["matched_identity"].isin(["No Matched Brand", "Consistent", "No Prediction"]))]
    no_instruction = no_instruction["email_file_path"].tolist()

    consistent = filtered_df.loc[filtered_df["matched_identity"].isin(["Consistent"])]
    no_matched_brand = filtered_df.loc[filtered_df["matched_identity"].isin(["No Matched Brand"])]
    no_pred_identity = filtered_df.loc[filtered_df["matched_identity"].isin(["No Prediction"])]

    no_identity = no_pred_identity["email_file_path"].tolist() + no_matched_brand["email_file_path"].tolist()

    ## breakdown
    # Recall =  0.929471032745592
    # FN no instruction =  0.25
    # FN no matched brand =  0.30357142857142855
    # FN no predicted identity =  0.125
    # FN consistent =  0.32142857142857145

    wo_instruction_pos = df[
        (df['our_pred'] == True) | (~df['matched_identity'].isin(['Sent from an whitelisted address',
                                                                  'Non-original email (Reply/Forward)',
                                                                  'Consistent',
                                                                  'No Prediction',
                                                                  'No Matched Brand']))]
    wo_internal = df[(df['our_pred'] == True) & (df['matched_identity'] != 'Internal')]
    print("Without call-to-action, recall = ", len(wo_instruction_pos) / len(df))
    print("Without internal, recall = ", len(wo_internal) / len(df))
    print("Complete Recall = ", len(df.loc[(df["our_pred"] == True)]) / len(df))
    print("FN no instruction = ", len(no_instruction) / len(filtered_df) )
    print("FN no matched brand = ", len(no_matched_brand) / len(filtered_df) )
    print("FN no predicted identity = ", len(no_pred_identity) / len(filtered_df) )
    print("FN consistent = ", len(consistent) / len(filtered_df) )
    exit()

    #
    # # no recognized instruction

    # alignment_tokens = []
    # for i in range(len(dataset)):
    #     if dataset.file_list[i] not in no_instruction:
    #         continue
    #
    #     if dataset.file_list[i] not in ["./datasets/phishpot/sample-115.eml",
    #                                     "./datasets/phishpot/sample-1121.eml",
    #                                     "./datasets/phishpot/sample-3309.eml",
    #                                     "./datasets/phishpot/sample-3108.eml"]:
    #         continue
    #
    #     email_file_path, (sender_name, sender_address), \
    #         (to_names, to_addresses), reply_to_address, \
    #         subject, email_body_text, header = dataset[i]
    #
    #     example = f'Subject: {subject} \n From: {sender_name} \n Body: {email_body_text}'
    #
    #     inputs = tokenizer(
    #         example,  # Input text
    #         return_tensors="pt",  # Return PyTorch tensors
    #         truncation=True,  # Truncate if longer than the model's max length
    #         add_special_tokens=True,  # Add [CLS] and [SEP]
    #     )
    #     word_to_token_indices = align_words_and_tokens(example)
    #     alignment_tokens.append(
    #         {
    #             "Id": i,
    #             "text": example,
    #             "cls_of_interest": action_cls,
    #             "email_file_path": email_file_path,
    #             "word_to_token_indices": word_to_token_indices
    #         }
    #     )
    # # #
    # # # # no recognized identity
    #
    # for i in range(len(dataset)):
    #     if dataset.file_list[i] not in no_identity:
    #         continue
    #
    #     if dataset.file_list[i] not in ["./datasets/phishpot/sample-115.eml",
    #                                     "./datasets/phishpot/sample-1121.eml",
    #                                     "./datasets/phishpot/sample-3309.eml",
    #                                     "./datasets/phishpot/sample-3108.eml"]:
    #         continue
    #
    #
    #     email_file_path, (sender_name, sender_address), \
    #         (to_names, to_addresses), reply_to_address, \
    #         subject, email_body_text, header = dataset[i]
    #
    #     example = f'Subject: {subject} \n From: {sender_name} \n Body: {email_body_text}'
    #
    #     inputs = tokenizer(
    #         example,  # Input text
    #         return_tensors="pt",  # Return PyTorch tensors
    #         truncation=True,  # Truncate if longer than the model's max length
    #         add_special_tokens=True,  # Add [CLS] and [SEP]
    #     )
    #     word_to_token_indices = align_words_and_tokens(example)
    #     alignment_tokens.append(
    #         {
    #             "Id": i,
    #             "text": example,
    #             "cls_of_interest": identity_cls,
    #             "email_file_path": email_file_path,
    #             "word_to_token_indices": word_to_token_indices
    #         }
    #     )
    #
    # with open(token_alignment_file, "w") as f:
    #     json.dump(alignment_tokens, f)
    # exit()

    '''Trace IN'''
    # Run the main function
    # Initialize required objects
    training_gradients_dict_action = torch.load(gradient_logging_file_action)
    training_gradients_dict_identity = torch.load(gradient_logging_file_identity)
    tvisualizer = TracInVisualizer(pretrained_checkpoint)
    ocr = OCR()

    with open(token_alignment_file, "r") as f:
        token_alignment_data = json.load(f)
    # Clear and recreate output directory
    if os.path.exists(visDir):
        shutil.rmtree(visDir)
    os.makedirs(visDir)

    for sample in tqdm(token_alignment_data):

        example = sample["text"]
        example_file_path = sample['email_file_path']
        token_ind_of_interest = sample.get("ind", [])
        token_ind = token_ind_of_interest[0] if len(token_ind_of_interest) else None
        cls_of_interest = sample["cls_of_interest"]
        harmful_samples = []

        inputs = tokenizer(
            example,  # Input text
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,  # Truncate if longer than the model's max length
            add_special_tokens=True,  # Add [CLS] and [SEP]
        )

        if cls_of_interest == 5:
            training_gradients_dict = training_gradients_dict_action
        elif cls_of_interest == 1:
            training_gradients_dict = training_gradients_dict_identity
        else:
            raise NotImplementedError

        token_gradient_for_cls = trainer.token_gradient(
            inputs=inputs,
            cls_of_interest = cls_of_interest,
            token_index_of_interest=token_ind # subject to ths example
        )

        training_gradients = torch.stack([x["grads"] for x in training_gradients_dict.values()])
        training_email_file_paths = [x["metadata"] for x in training_gradients_dict.values()]
        influence_scores = torch.einsum('i,ji->j', token_gradient_for_cls, training_gradients)
        sorted_indices = influence_scores.argsort().tolist()
        topk = 5
        harmful = sorted_indices[-topk:][::-1] # positive is harmful

        for ind in harmful:
            input_ids, tags = training_gradients_dict[ind]["input_ids"], training_gradients_dict[ind]["labels"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            tags = [x for x in tags[0].tolist()]
            clean_tags = []
            for tag in tags:
                if tag in [0, -100]:
                    clean_tags.append("O")
                else:
                    clean_tags.append(model.config.id2label[tag])
            harmful_samples.append((tokens, clean_tags, influence_scores[ind], training_email_file_paths[ind]))

        html = tvisualizer(raw_text=example,
                           token_idx=token_ind,
                           metadata=example_file_path,
                           helpful_examples=[],
                           harmful_examples=harmful_samples)

        dataset = RenderDataset(example_file_path, ocr_model=ocr, dumpDir=visDir)

        with open(example_file_path, "rb") as f:
            msg = f.read()
        dataset.render_eml(msg, dumpName=f"{os.path.basename(example_file_path)}") # save the image as well

        with open(f"{visDir}/{os.path.basename(example_file_path)}.html", "w") as f:
            f.write(html)

    # ## Use OCR to render => improve from 0.44 to 0.66
    # ## 1st round of action debugging => from 0.66 to 0.78
    # ## 2nd round of action debugging + identity debugging => from 0.78 to 0.91




