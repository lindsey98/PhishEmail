import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from collections import defaultdict


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

