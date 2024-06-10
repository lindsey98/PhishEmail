from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from time import perf_counter
from spacy import displacy

class StopAtMultipleTokensCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences):
        # Encode each sequence to token IDs using the tokenizer
        self.stop_token_ids_list = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]

    def __call__(self, input_ids, scores):
        input_len = input_ids.shape[1]
        # Check each set of stop token IDs
        for stop_token_ids in self.stop_token_ids_list:
            seq_len = len(stop_token_ids)
            if input_len >= seq_len:
                # Compare the last generated tokens against the current stop token IDs set
                if all(input_ids[0, -seq_len + i] == stop_token_ids[i] for i in range(seq_len)):
                    return True
        return False


def _generate_identity(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    stop_sequences = ['###', '</s>']
    stopping_criterion = StoppingCriteriaList([StopAtMultipleTokensCriteria(tokenizer, stop_sequences)])

    with torch.inference_mode():
        t0 = perf_counter()
        output = model.generate(input_ids=tokenized_prompt,
                                stopping_criteria=stopping_criterion,
                                generation_config=gen_config)
        total_time = perf_counter() - t0
        generation_ids = output[0][len(tokenized_prompt[0]):]
        num_gen_tokens = len(generation_ids)
        generation = tokenizer.decode(generation_ids, skip_special_tokens=True)
        return dict(generation=generation,
                    generation_ids=generation_ids.tolist(),
                    total_time=total_time,
                    num_gen_tokens=num_gen_tokens)


def ner_prediction_postprocess(model, tokenizer, model_outputs, input_ids, offset_mapping):

    entities = []
    for i in range(len(model_outputs.logits)):
        logits = model_outputs.logits[i]
        token_scores = logits.softmax(dim=-1)  # 计算每个 token 的 softmax 得分
        token_labels = token_scores.argmax(dim=-1)  # 选择得分最高的标签

        for token_index, token_label in enumerate(token_labels):
            # 跳过特殊 token（例如 [CLS], [SEP] 等）
            if input_ids[i][token_index] in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                continue

            # 获取原始文本中 token 的起始和结束位置
            start, end = offset_mapping[i][token_index].tolist()
            entity = {
                "word": tokenizer.convert_ids_to_tokens([input_ids[i][token_index]])[0],
                "entity": model.config.id2label[token_label.item()],
                "score": token_scores[token_index][token_label].item(),
                "start": start,
                "end": end
            }
            entities.append(entity)

    return entities

def adjust_for_multiline(text, start, end):
    """Adjust entity start and end positions for multi-line text."""
    new_start = start
    new_end = end
    lines = text.split('\n')

    # Adjust start position
    for line in lines:
        if new_start > len(line):
            new_start -= (len(line) + 1)  # +1 for the newline character
        else:
            break

    # Adjust end position
    for line in lines:
        if new_end > len(line):
            new_end -= (len(line) + 1)  # +1 for the newline character
        else:
            break

    return new_start, new_end

# Define a function to clean the output
def ner_clean_predictions(predictions, text):
    entities = []
    current_entity = None
    current_label = None

    for item in predictions:
        if item['entity'].startswith('B-') or (item['entity'].startswith('I-') and current_entity is None):
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "label": current_label
                })
            current_entity = {"start": item['start'], "end": item['end']}
            current_label = item['entity'][2:]
        elif item['entity'].startswith('I-') and current_label == item['entity'][2:]:
            current_entity['end'] = item['end']
        else:
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "label": current_label
                })
            current_entity = None
            current_label = None

    if current_entity is not None:
        entities.append({
            "start": current_entity['start'],
            "end": current_entity['end'],
            "label": current_label
        })

    # Handle multi-line text
    for entity in entities:
        entity['start'], entity['end'] = adjust_for_multiline(text, entity['start'], entity['end'])

    return {"text": text, "ents": entities, "title": None}

def ner_clean_ground_truth(tokens, ner_tags, id_to_label):
    text = ' '.join(tokens)
    entities = []
    current_entity = None
    current_label = None

    start_char = 0
    for idx, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
        end_char = start_char + len(token)
        if id_to_label[tag_id].startswith('B-') or (id_to_label[tag_id].startswith('I-') and current_entity is None):
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "label": current_label
                })
            current_entity = {"start": start_char, "end": end_char}
            current_label = id_to_label[tag_id][2:]
        elif id_to_label[tag_id].startswith('I-') and current_label == id_to_label[tag_id][2:]:
            current_entity['end'] = end_char
        else:
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "label": current_label
                })
            current_entity = None
            current_label = None

        start_char = end_char + 1  # +1 for the space between tokens

    if current_entity is not None:
        entities.append({
            "start": current_entity['start'],
            "end": current_entity['end'],
            "label": current_label
        })

    # Handle multi-line text
    for entity in entities:
        entity['start'], entity['end'] = adjust_for_multiline(text, entity['start'], entity['end'])

    return {"text": text, "ents": entities, "title": None}


def ner_create_spacy_doc(cleaned_output, nlp):
    doc = nlp.make_doc(cleaned_output['text'])
    ents = []

    for ent in cleaned_output['ents']:
        span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
        if span is not None:
            # Check for overlapping entities
            overlap = False
            for existing_ent in ents:
                if span.start < existing_ent.end and span.end > existing_ent.start:
                    overlap = True
                    if span.label_ == "action":
                        # Overwrite the existing entity if the new label is 'action'
                        ents.remove(existing_ent)
                        ents.append(span)
                        break
                    elif span.label_ == existing_ent.label_:
                        # Merge entities with the same label
                        merged_span = doc.char_span(
                            min(span.start, existing_ent.start),
                            max(span.end, existing_ent.end),
                            label=span.label_
                        )
                        if merged_span is not None:
                            ents.remove(existing_ent)
                            ents.append(merged_span)
                        break
                    else:
                        print(f"Overlapping entity found: {span.text} overlaps with {existing_ent.text}")
                        break
            if not overlap:
                ents.append(span)
    doc.ents = ents
    return doc

def visualize_predictions(pred_doc, metadata="", options=None):
    pred_html = displacy.render(pred_doc, style="ent", page=True, options=options)
    rendered_html = f"""
        <div style="display: flex; justify-content: space-around; position: relative;">
            <div style="width: 100%; border: 1px solid black;">
                <h3>Predicted</h3>
                {pred_html}
            </div>
            <div style="position: absolute; top: 0; right: 0; background-color: #fff; padding: 5px; border: 1px solid black;">
                <strong>Metadata:</strong> {metadata}
            </div>
        </div>
        """
    return rendered_html

def visualize_predictions_and_ground_truth(pred_doc, gt_doc, metadata="", options=None):
    pred_html = displacy.render(pred_doc, style="ent", page=True, options=options)
    gt_html = displacy.render(gt_doc, style="ent", page=True, options=options)
    combined_html = f"""
    <div style="display: flex; justify-content: space-around; position: relative;">
        <div style="width: 45%; border: 1px solid black;">
            <h3>Predicted</h3>
            {pred_html}
        </div>
        <div style="width: 45%; border: 1px solid black;">
            <h3>Ground Truth</h3>
            {gt_html}
        </div>
        <div style="position: absolute; top: 0; right: 0; background-color: #fff; padding: 5px; border: 1px solid black;">
            <strong>Metadata:</strong> {metadata}
        </div>
    </div>
    """
    return combined_html