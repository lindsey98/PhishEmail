from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from lib.model_utils.losses import FocalLoss

class BertTrainer_FocalLoss(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=2)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute focal loss
        loss = self.focal_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss