
from transformers.integrations import WandbCallback
from transformers import GenerationConfig
import wandb
import torch
from lib.model_utils.postprocessing import ner_prediction_postprocess, ner_clean_predictions, ner_create_spacy_doc
import spacy
from tqdm import tqdm

class CausalLMCallback(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
            generation_ids = output[0][len(tokenized_prompt[0]):]
        return self.tokenizer.decode(generation_ids, skip_special_tokens=True)

    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})


class NERCallback(WandbCallback):

    def __init__(self, trainer, test_dataset, num_samples=30):
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model = trainer.model
        self.tokenizer = trainer.tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        nlp = spacy.blank("en")
        plots = []
        entity_colors = {"organization": "#ADD8E6",  # Light Blue
                         "action": "#FFA07A",  # Light Salmon
                         "relation": "#98FB98"}  # Pale Green

        for example in tqdm(self.sample_dataset, leave=False):

            text = ' '.join(example['tokens'])
            tokens = self.tokenizer([text], return_tensors='pt',
                                    truncation=True, padding=True,
                                    is_split_into_words=False, return_offsets_mapping=True)
            inputs = {
                "input_ids": tokens["input_ids"].to(self.model.device),
                "attention_mask": tokens["attention_mask"].to(self.model.device)
            }
            with torch.inference_mode():
                outputs = self.model(**inputs)  # Assume model is on the appropriate device

            tokenized_email = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"])
            tokenized_email_str = self.tokenizer.convert_tokens_to_string(tokenized_email)

            entities = ner_prediction_postprocess(self.model, self.tokenizer, outputs, tokens["input_ids"], tokens["offset_mapping"])
            doc = ner_create_spacy_doc(tokenized_email_str, entities, nlp)
            # Apply entity colors
            html = spacy.displacy.render(doc, style="ent", options={"colors": entity_colors})
            plots.append([wandb.Html(html)])

        table = wandb.Table(data=plots, columns=['predictions'])
        wandb.log({'NER predictions': table})

        # Restore the model to the original device
        return control