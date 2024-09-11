from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import re
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

class SpellFixer:

    def __init__(self, model_id="ai-forever/T5-large-spell"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prefix = "grammar: "
        self.max_length = 512

    def preserve_case(self, original_sentence, corrected_sentence):
        """Preserve the original case for the first letter of each sentence."""
        if original_sentence and corrected_sentence:
            if original_sentence[0].isupper():
                corrected_sentence = corrected_sentence[0].upper() + corrected_sentence[1:]
            else:
                corrected_sentence = corrected_sentence[0].lower() + corrected_sentence[1:]
        return corrected_sentence

    def __call__(self, paragraph: str):
        # Find sentences and keep delimiters
        sentences_with_delimiters = re.findall(r'([^.!?]+)([.!?]*)', paragraph)

        corrected_sentences = []
        i = 0
        while i < len(sentences_with_delimiters):
            batch = []
            token_count = 0
            original_batch = []  # Track the original batch for case and delimiter preservation
            # Dynamically determine how many sentences to include in the batch
            for j in range(i, len(sentences_with_delimiters)):
                sentence, delimiter = sentences_with_delimiters[j]
                sent = self.prefix + sentence.strip()
                # Calculate the number of tokens for the current sentence
                encodings = self.tokenizer(sent, return_tensors="pt", truncation=True)
                sent_token_count = encodings['input_ids'].shape[1]

                if token_count + sent_token_count > self.max_length:
                    # If adding this sentence exceeds max_length, process the batch
                    break
                else:
                    batch.append(sent)
                    original_batch.append((sentence, delimiter))  # Preserve original case and delimiter
                    token_count += sent_token_count

            if batch:  # If there's anything to process
                # Tokenize the batch
                encodings = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

                # Generate corrections in a batch
                generated_tokens = self.model.generate(**encodings)

                # Decode the batch and collect corrected sentences
                answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                # Apply case preservation and append delimiters back
                for corrected_sentence, (original_sentence, delimiter) in zip(answers, original_batch):
                    corrected_sentence = self.preserve_case(original_sentence, corrected_sentence)
                    corrected_sentences.append(corrected_sentence + delimiter)

            # Move the index forward by the number of sentences in the current batch
            i += len(batch)

        # Join corrected sentences with a space to form the paragraph
        corrected_paragraph = " ".join(corrected_sentences)
        return corrected_paragraph

if __name__ == '__main__':
    fixer = SpellFixer()
    print(fixer("lets do a comparsion. Amazzon is a global company! Are you kiding?"))