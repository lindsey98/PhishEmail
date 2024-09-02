import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from collections import defaultdict

@torch.inference_mode()
def compute_perplexity(eval_dataloader, model):
    model.eval()
    nlls = []

    for i, batch in tqdm(enumerate(eval_dataloader)):
        if (batch["labels"] == -100).all():
            continue  # Skip this batch
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            neg_log_likelihood = out.loss

        nlls.append(neg_log_likelihood)

    # Log results at the end
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


def compute_token_classification_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert numeric predictions and labels to their corresponding labels
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate overall precision, recall, and F1 score
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    # Generate a classification report that includes class-wise precision, recall, and F1 score
    class_report = classification_report(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_report": class_report
    }


def calculate_iou(pred, truth):
    # Calculate the overlap between predicted and truth
    start = max(pred['start'], truth['start'])
    end = min(pred['end'], truth['end'])
    if start < end:  # There is an overlap
        intersection = end - start
        union = (pred['end'] - pred['start']) + (truth['end'] - truth['start']) - intersection
        return intersection / union
    return 0

def compute_entity_overlap_metrics(cleaned_output, cleaned_label):
    metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    predicted_entities = cleaned_output['ents']
    true_entities = cleaned_label['ents']

    # Track matched ground-truth entities to avoid double counting
    matched_truths = set()

    # Calculate true positives and false positives
    for pred in predicted_entities:
        match_found = False
        for truth in true_entities:
            if truth not in matched_truths and pred['label'] == truth['label']:
                if calculate_iou(pred, truth) > 0.5:
                    metrics[pred['label']]['tp'] += 1
                    matched_truths.add(truth)
                    match_found = True
                    break
        if not match_found:
            metrics[pred['label']]['fp'] += 1

    # Calculate false negatives
    for truth in true_entities:
        if truth not in matched_truths:
            metrics[truth['label']]['fn'] += 1

    # Calculate class-wise and overall metrics
    class_wise_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    for label, counts in metrics.items():
        precision = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
        recall = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_wise_metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1}
        total_tp += counts['tp']
        total_fp += counts['fp']
        total_fn += counts['fn']

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    return {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "class_wise_metrics": class_wise_metrics
    }