import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true, y_prob, threshold=0.5):
    """Table 1 of the exercise: accuracy, sensitivity, specificity, F1, AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "sensitivity": tp / (tp + fn) * 100 if tp + fn else np.nan,
        "specificity": tn / (tn + fp) * 100 if tn + fp else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "auc": roc_auc_score(y_true, y_prob),
    }


def best_threshold(y_true, y_prob):
    """Youden's J on the ROC. Pick this on validation, never on test."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def multiclass_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(1)
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "auc_macro": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
    }


def plot_roc(y_true, y_prob, label="model", ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    ax = ax or plt.subplots(figsize=(5, 5))[1]
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return ax


def plot_confusion(y_true, y_pred, class_names=None, ax=None, normalize=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d"
    )
    ax.set_title("Confusion matrix")
    return ax


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    if history.get("best_epoch"):
        for ax in (ax_loss, ax_acc):
            ax.axvline(history["best_epoch"], color="grey", linestyle=":", linewidth=1)

    fig.tight_layout()
    return fig
