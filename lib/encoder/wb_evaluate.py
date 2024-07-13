
from lib.encoder.wb_finetune_bert import ner_clean_predictions, ner_create_spacy_doc
from lib.model_utils.postprocessing import visualize_predictions
from transformers import pipeline
import spacy
from spacy import displacy
from tqdm import tqdm
from lib.data.dataloader import EmailDataset
import os
from pathlib import Path

if __name__ == '__main__':
    classifier = pipeline("ner", model="./checkpoints/output_ner/checkpoint-1755")
    nlp = spacy.blank("en")
    entity_colors = {"organization": "#ADD8E6",  # Light Blue
                     "action": "#FFA07A",  # Light Salmon
                     "relation": "#98FB98"}  # Pale Green

    dataset_name = "GPT_Dataset_V2"
    dataset = EmailDataset(f"./datasets/{dataset_name}")
    print(f'Length of dataset {len(dataset)}')

    output_dir = Path(f"./datasets/ner_visualization_{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for it in tqdm(range(len(dataset))):
        email_file_path, (sender_name, sender_address), (to_names, to_addresses), subject, email_body_text, headers = dataset[it]
        parsed_email = f'Subject: {subject}. From: {sender_name}. Body: {email_body_text}'

        output = classifier(parsed_email)
        # Clean the output
        cleaned_output = ner_clean_predictions(output, parsed_email)
        # Load SpaCy model
        nlp = spacy.blank("en")
        # Create SpaCy doc
        pred_doc = ner_create_spacy_doc(cleaned_output, nlp)
        # Render the visualization
        html = visualize_predictions(pred_doc, metadata=email_file_path, options={"colors": entity_colors})

        # 提取文件路径的二级目录名和基本文件名
        email_file_path_obj = Path(email_file_path)
        email_file_path_2nd_level_basename = email_file_path_obj.parts[-2]  # 假设文件路径至少有两级目录
        email_file_path_basename = email_file_path_obj.stem  # 不包括扩展名的文件名

        html_file_path = output_dir / f"{email_file_path_2nd_level_basename}_{email_file_path_basename}.html"
        with open(html_file_path, "w", encoding="utf-8") as file:
            file.write(html)

