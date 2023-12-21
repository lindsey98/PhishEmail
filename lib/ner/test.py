from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from utils import load_jsonl
import Levenshtein
from tqdm import tqdm
from lib.data.dataloader import sanitize_text
from nltk.tokenize import sent_tokenize


def adjust_gt_labels_for_sentence(sentence, gt_labels, offset):
    """
    Adjust GT labels to be relative to the sentence.
    """
    adjusted_labels = []
    for start, end, label in gt_labels:
        if start >= offset and end <= offset + len(sentence):
            # Adjust the start and end positions relative to the sentence
            adjusted_start = start - offset
            adjusted_end = end - offset
            adjusted_labels.append([adjusted_start, adjusted_end, label])
    return adjusted_labels

def apply_ner_pipeline(line, pipeline):
    if len(line.strip()) > 0:
        result = pipeline(line)
        return result['output']
    return []

def wrap_entities_in_html(sentence, predicted_entities, gt_entities=None):
    """Wrap identified entities in HTML with bounding boxes and annotations."""
    last_idx = 0
    html_sentence = ""

    # Combine and sort entities by their start position
    if gt_entities is not None:
        combined_entities = [(e, 'gt') for e in gt_entities] + [(e, 'pred') for e in predicted_entities]
    else:
        combined_entities = [(e, 'pred') for e in predicted_entities]
    combined_entities.sort(key=lambda x: x[0]['start'])

    for result, entity_type in combined_entities:
        start, end = result['start'], result['end']
        entity = sentence[start:end]
        entity_label = result['type']
        prob = result.get('prob', None)  # Probability might not be present for GT

        # Adding the text before the entity
        html_sentence += sentence[last_idx:start]

        # Define background color based on entity type (GT or predicted)
        bg_color = 'lightblue' if entity_type == 'gt' else 'lightcoral'

        # Adding the entity with background color and annotation
        annotation_text = f"{entity_label}, {prob:.2f}" if prob is not None else entity_label
        html_sentence += f"""
            <span class='entity' style='background-color: {bg_color}; padding: 0 2px;'>
                {entity}
                <span class='annotation' style='font-size: smaller;'>
                    {annotation_text}
                </span>
            </span>
        """
        last_idx = end

    html_sentence += sentence[last_idx:]  # Add the remaining part of the sentence
    return html_sentence

if __name__ == '__main__':

    ner_pipeline = pipeline(Tasks.named_entity_recognition,
                            'damo/nlp_raner_named-entity-recognition_english-large-news')
    #

    # jsonl_data = load_jsonl('./datasets/nazario_annot.jsonl')
    # os.makedirs('./datasets/ner_results_nazario/', exist_ok=True)
    #
    # for item in jsonl_data:
    #     body_text = get_body_text(item['text'])
    #     sentences = sent_tokenize(body_text)  # Using NLTK for sentence tokenization
    #
    #     processed_text = ""
    #     for sentence in sentences:
    #         if sentence.strip():  # Skip empty sentences
    #             ner_results = apply_ner_pipeline(sentence)  # Your NER function
    #             processed_sentence = wrap_entities_in_html(sentence, ner_results)
    #             processed_text += processed_sentence + " "
    #
    #     # Display the processed text
    #     with open(f"./datasets/ner_results_nazario/{item['id']}.html", 'w') as file:
    #         file.write(f"<div style='line-height: 1.5;'><style>.entity {{ background-color: yellow; }}</style>{processed_text}</div>")

    jsonl_data = load_jsonl('./datasets/iwspa-annot.jsonl')
    total = 0
    predict_correct = 0
    os.makedirs('./datasets/iwspa-ner-failure/', exist_ok=True)

    with tqdm(total=len(jsonl_data)) as pbar:
        for count, item in enumerate(jsonl_data):
            body_text = item['text']
            gt_labels = [x for x in item['label'] if x[-1] == 'sender identity']
            gt_sender_identity = [item['text'][gt_labels[i][0]:gt_labels[i][1]] for i in range(len(gt_labels))]

            sentences = sent_tokenize(body_text)
            predicted_sender_identity = []
            processed_text = ""
            offset = 0  # Keeps track of the character position in the original text

            for sentence in sentences:
                if sentence.strip():
                    sentence = sanitize_text(sentence)

                    # Adjust GT labels for the current sentence
                    adjusted_gt_labels = adjust_gt_labels_for_sentence(sentence, gt_labels, offset)
                    # Convert adjusted GT labels to the format expected by wrap_entities_in_html
                    gt_entities = [{'start': start, 'end': end, 'type': label} for start, end, label in
                                   adjusted_gt_labels]

                    ner_results = apply_ner_pipeline(sentence, ner_pipeline)
                    predicted_sender_identity.extend([x['span'] for x in ner_results if x['type'] == 'ORG'])
                    processed_sentence = wrap_entities_in_html(sentence, ner_results, gt_entities)
                    processed_text += processed_sentence + " "

                # Update the offset for the next sentence
                offset += len(sentence) + 1  # +1 for the space or punctuation after the sentence
                # Account for newline characters and other separators
                next_offset = body_text.find(sentence, offset) + len(sentence)
                separators_length = next_offset - offset
                offset += separators_length

            predicted_sender_identity = [s.lower() for s in predicted_sender_identity]
            gt_sender_identity = [s.lower() for s in gt_sender_identity]

            correct = False
            for s1 in predicted_sender_identity:
                for s2 in gt_sender_identity:
                    distance = Levenshtein.distance(s1, s2)
                    if distance <= 1:
                        correct = True
                        break
                    elif s2 in s1:
                        correct = True
                        break
                    elif s1 in s2:
                        correct = True
                        break
                if correct:
                    break
            if len(gt_sender_identity) == 0:
                correct = True

            total += 1
            if correct:
                predict_correct += 1
            else:
                print('Ground-truth ', gt_sender_identity)
                print('Predicted ', predicted_sender_identity)
                # Write the HTML content to the file with additional styling
                with open(f'./datasets/iwspa-ner-failure/{count}.html', 'w', encoding='utf-8') as file:
                    file.write(f"<div style='line-height: 1.5;'>{processed_text}</div>")

            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"Recall: {predict_correct / total:.2f}", refresh=True)


    final_recall = predict_correct / total if total > 0 else 0
    print(f"Final Recall: {final_recall:.2f}") # iwspa Recall = 0.98


