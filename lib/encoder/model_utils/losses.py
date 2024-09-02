import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss for Multiclass Classification

        :param alpha: (tensor, optional) A tensor of shape (C,) where C is the number of classes. If not provided, it defaults to 1 for all classes.
        :param gamma: (float, optional) The focusing parameter. Default is 2.
        :param reduction: (str, optional) Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.alpha is not None:
            assert isinstance(self.alpha, torch.Tensor), "alpha must be a tensor"
            assert len(self.alpha) > 1, "alpha must be a tensor of shape (C,), where C is the number of classes"

    def forward(self, inputs, targets):
        """
        Forward pass for the Focal Loss.

        :param inputs: (tensor) Predictions of shape (N, C), where N is the batch size and C is the number of classes.
        :param targets: (tensor) True labels of shape (N,) where each value is 0 <= targets[i] <= C-1.
        :return: (tensor) The calculated Focal Loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss