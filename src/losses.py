import torch
import torch.nn as nn
import torch.nn.functional as F

SMOOTH = 1.0


def dice_loss(logits, targets, smooth=SMOOTH):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(1)
    dice = (2 * intersection + smooth) / (probs.sum(1) + targets.sum(1) + smooth)
    return 1 - dice.mean()


def iou_loss(logits, targets, smooth=SMOOTH):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(1)
    union = probs.sum(1) + targets.sum(1) - intersection
    return 1 - ((intersection + smooth) / (union + smooth)).mean()


class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        return dice_loss(logits, targets)


class DiceBCELoss(nn.Module):
    """Dice + BCE. bce_weight trades one against the other."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss(logits, targets)


class IoUBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * iou_loss(logits, targets)


LOSSES = {
    "bce": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "dice_bce": DiceBCELoss,
    "iou_bce": IoUBCELoss,
    "mse": lambda: (lambda logits, t: F.mse_loss(torch.sigmoid(logits), t)),
}
