from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import os
import re
import torch
from ...utilities.data_utils import preserve_case
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

class T5SpellFixer:

    def __init__(self, model_id="ai-forever/T5-large-spell"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_id, device_map='balanced')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prefix = "grammar: "
        self.max_length = 512
        self.max_new_token = 512


    def __call__(self, paragraph: str):
        # Prepare the input text with the prefix
        input_text = paragraph.strip()

        # Tokenize the input text to get the total number of tokens
        tokens = self.tokenizer.tokenize(input_text)

        # Determine the maximum number of tokens per chunk
        max_tokens = self.max_length - 4  # Adjust for special tokens if necessary

        # Split tokens into chunks that fit within the maximum length
        chunks = [
            tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)
        ]

        corrected_chunks = []
        for chunk_tokens in chunks:
            # Convert tokens back to text
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunk_text = self.prefix + chunk_text.strip()  # Add prefix to each chunk

            # Tokenize and encode the chunk
            encodings = self.tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            encodings = encodings.to(self.model.device)

            # Generate corrections
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encodings,
                    max_new_tokens=self.max_new_token
                )

            # Decode the output
            corrected_chunk = self.tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True
            )

            corrected_chunks.append(corrected_chunk)

        # Join corrected chunks to form the full corrected paragraph
        corrected_paragraph = " ".join(corrected_chunks)

        # Optionally, preserve the original casing
        corrected_paragraph = preserve_case(paragraph, corrected_paragraph)

        return corrected_paragraph

if __name__ == '__main__':
    fixer = T5SpellFixer("./checkpoints/t5-efficient-tiny/checkpoint-15426")
    print(fixer("Amazon is a global company"))

## T5-tiny 1.516秒