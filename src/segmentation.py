import numpy as np
import torch

from .training import EarlyStopping


@torch.no_grad()
def confusion_counts(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    return np.array([tp, fp, fn, tn])


def metrics_from_counts(counts, eps=1e-8):
    """Table 2 of the exercise, computed over all pixels of all images at once.

    Pixel counts are pooled before dividing, not averaged per image. A per-image mean
    would let a tiny image with 3 foreground pixels weigh as much as a full one.
    """
    tp, fp, fn, tn = counts
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "jaccard": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "recall": recall,
        "precision": precision,
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, n = 0.0, 0
    counts = np.zeros(4)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        n += images.size(0)
        counts += confusion_counts(logits.detach(), masks)

    return metrics_from_counts(counts), loss_sum / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    loss_sum, n = 0.0, 0
    counts = np.zeros(4)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)

        loss_sum += criterion(logits, masks).item() * images.size(0)
        n += images.size(0)
        counts += confusion_counts(logits, masks, threshold)

    return metrics_from_counts(counts), loss_sum / n


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        n_epochs, patience=5, scheduler=None, checkpoint_path=None):
    stopper = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "train_jaccard": [], "val_jaccard": []}

    for epoch in range(1, n_epochs + 1):
        train_m, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m, val_loss = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for key in ("f1", "jaccard"):
            history[f"train_{key}"].append(train_m[key])
            history[f"val_{key}"].append(val_m[key])

        print(f"Epoch [{epoch}/{n_epochs}]  "
              f"train_loss {train_loss:.4f}  train_dice {train_m['f1']:.4f}  "
              f"val_loss {val_loss:.4f}  val_dice {val_m['f1']:.4f}  "
              f"val_iou {val_m['jaccard']:.4f}")

        if stopper.step(val_loss, model, epoch):
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch} "
                  f"(val_loss {stopper.best_loss:.4f})")
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch
    return history
