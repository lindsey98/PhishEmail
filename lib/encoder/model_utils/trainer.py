from transformers import Trainer
from .losses import FocalLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers.trainer_utils import (
    has_length
)
from typing import Dict, List

class BertTrainer_FocalLoss(Trainer):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=2, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.get("labels")
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute focal loss
        loss = self.focal_loss(
            logits.view(-1, logits.size(-1))[labels.view(-1) != -100],
            labels.view(-1)[labels.view(-1) != -100]
        )
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader_no_shuffle(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": False,
            "drop_last": False
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    def batch_gradient(
        self,
        dataloader: DataLoader,
        metadata: List[str],
        cls_of_interest: int
    ) :
        '''
        Get gradients
        :param dataloader:
        :return:
        '''
        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        model = self._wrap_model(self.model, training=True, dataloader=dataloader)
        model.train()

        num_examples = self.num_examples(dataloader)
        assert num_examples == len(metadata)
        print(f"  Num examples = {num_examples}")

        gradient_dict = {}

        for step, (inputs, file_path) in enumerate(zip(dataloader, metadata)):

            model.zero_grad()

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
            loss = loss.mean()
            loss.backward()

            proj_layer_grad = {
                n: p.grad.data.cpu() for n, p in model.named_parameters() if
                (p.grad is not None) and ("classifier.weight" in n)
            }
            proj_layer_grad = next(iter(proj_layer_grad.values()), None)
            if proj_layer_grad is not None:
                proj_layer_grad = F.normalize(proj_layer_grad, dim=-1)
                proj_layer_grad = proj_layer_grad[cls_of_interest, :]
            loss.detach()

            gradient_dict[step] = inputs.to("cpu").data
            gradient_dict[step]["grads"] = proj_layer_grad
            gradient_dict[step]["metadata"] = file_path

        model.zero_grad()
        return gradient_dict

    def token_gradient(
        self,
        inputs,
        token_index_of_interest: int,
        cls_of_interest: int
    ):
        model = self._wrap_model(self.model, training=True, dataloader=None)
        model.train()
        inputs = inputs.to(model.device)

        model.zero_grad()
        outputs = model(**inputs)

        logits = outputs.get("logits")
        logits_of_cls = logits[:, token_index_of_interest, cls_of_interest] # (B, Seq_len, C) -> (1, 1, 1)
        logits_of_cls = logits_of_cls.mean() # collapse into 1 metric

        logits_of_cls.backward()

        proj_layer_grad = {
            n: p.grad.data.cpu() for n, p in model.named_parameters() if
            (p.grad is not None) and ("classifier.weight" in n)
        }
        proj_layer_grad = next(iter(proj_layer_grad.values()), None) # (C, d), where C = num_classes
        if proj_layer_grad is not None:
            proj_layer_grad = F.normalize(proj_layer_grad, dim=-1)
            proj_layer_grad = proj_layer_grad[cls_of_interest, :]
        logits_of_cls.detach()

        model.zero_grad()
        return proj_layer_grad





