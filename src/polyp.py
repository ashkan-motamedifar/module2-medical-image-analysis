"""Polyp characterization on the CCE dataset (Task 1).

Splits are by **polyp**, not by image and not by patient id alone.

Each line of `train_data.txt` / `valid_data.txt` / `test_data.txt` is
`<patient_id>,<polyp_number>,<class>`, and image files are named
`<patient_id>_P<polyp_number>_<frame>.jpg`. So the split key is the pair, e.g.
`12485_P1`.

Keying on the patient id alone — as the lecturer's original notebook does with
`filename.split('_')[0]` — is wrong here: one patient can have several polyps with
different labels sent to different splits. Patient 12485 has P1 (Neoplasia) in train
and P2 (Non-neoplasia) in test, so an id-only match copies all ten of that patient's
images into *both* splits. Across this dataset that puts 355 of 569 test images into
training as well, which makes the reported test scores meaningless.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .training import EarlyStopping

MEAN_STD = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class PolypDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), os.path.basename(path)


def _polyp_keys(split_file):
    """`12485,1,Neoplasia` -> `12485_P1`, matching the first two fields of a filename."""
    keys = set()
    with open(split_file) as f:
        for line in f:
            if line.strip():
                patient, polyp = line.strip().split(",")[:2]
                keys.add(f"{patient}_P{polyp}")
    return keys


def _polyp_key_of(filename):
    return "_".join(os.path.basename(filename).split("_")[:2])


def load_polyp_splits(root=".", data_dir="PCData", image_size=224, batch_size=32,
                      augment=True, num_workers=0):
    """Returns (train_loader, val_loader, test_loader)."""
    all_samples = []
    for folder, label in (("Non-neoplasia", 0), ("Neoplasia", 1)):
        directory = os.path.join(root, data_dir, folder)
        all_samples += [
            (os.path.join(directory, f), label)
            for f in sorted(os.listdir(directory))
            if not f.startswith(".")
        ]

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD),
    ])
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD),
    ]) if augment else eval_tf

    loaders, chosen = [], {}
    for name, transform in (("train", train_tf), ("valid", eval_tf), ("test", eval_tf)):
        keys = _polyp_keys(os.path.join(root, f"{name}_data.txt"))
        samples = [(p, y) for p, y in all_samples if _polyp_key_of(p) in keys]
        if not samples:
            raise ValueError(f"{name} split is empty — check the keys in {name}_data.txt")
        chosen[name] = {p for p, _ in samples}
        loaders.append(DataLoader(
            PolypDataset(samples, transform),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        ))
        n_pos = sum(y for _, y in samples)
        print(f"{name:>5}: {len(samples):5d} images  ({n_pos} neoplasia, "
              f"{len(samples) - n_pos} non-neoplasia)  {len(keys)} polyps")

    _check_splits(chosen, len(all_samples))
    return tuple(loaders)


def _check_splits(chosen, n_total):
    """Loud failure beats a silently leaking split."""
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = chosen[a] & chosen[b]
        if shared:
            raise ValueError(f"{len(shared)} images appear in both {a} and {b} — "
                             f"the splits leak, e.g. {sorted(shared)[:3]}")

    covered = set().union(*chosen.values())
    if len(covered) != n_total:
        print(f"  note: {n_total - len(covered)} of {n_total} images are in no split file")

    patients = {name: {os.path.basename(p).split("_")[0] for p in paths}
                for name, paths in chosen.items()}
    straddling = patients["train"] & patients["test"]
    if straddling:
        print(f"  note: {len(straddling)} patients have polyps in both train and test. "
              f"The splits are polyp-level, not patient-level — mention this in the report.")


def build_model(name="mobilenet_v3_large", pretrained=True):
    """One binary logit out. `pretrained=False` trains from scratch."""
    import torchvision.models as models

    weights = "DEFAULT" if pretrained else None
    model = getattr(models, name)(weights=weights)

    if hasattr(model, "fc"):                                  # resnet, resnext
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):       # mobilenet, efficientnet
            last = model.classifier[-1]
            model.classifier[-1] = nn.Linear(last.in_features, 1)
        else:                                                 # densenet
            model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        raise ValueError(f"don't know how to reshape the head of {name}")
    return model


def pos_weight_from(loader, device):
    """For BCEWithLogitsLoss when the classes are unbalanced."""
    labels = np.concatenate([y.numpy() for _, y, _ in loader])
    n_pos = float(labels.sum())
    return torch.tensor([(len(labels) - n_pos) / max(n_pos, 1.0)],
                        dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, n = 0.0, 0
    y_true, y_prob = [], []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * targets.size(0)
        n += targets.size(0)
        y_true.append(targets.detach().cpu().numpy().ravel())
        y_prob.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())

    return loss_sum / n, np.concatenate(y_true), np.concatenate(y_prob)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (mean loss, true labels, predicted probabilities, file names)."""
    model.eval()
    loss_sum, n = 0.0, 0
    y_true, y_prob, names = [], [], []

    for images, targets, batch_names in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)
        logits = model(images)

        loss_sum += criterion(logits, targets).item() * targets.size(0)
        n += targets.size(0)
        y_true.append(targets.cpu().numpy().ravel())
        y_prob.append(torch.sigmoid(logits).cpu().numpy().ravel())
        names += list(batch_names)

    return loss_sum / n, np.concatenate(y_true), np.concatenate(y_prob), names


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        n_epochs, patience=5, scheduler=None, checkpoint_path=None):
    from .metrics import binary_metrics

    stopper = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        train_loss, ytr, ptr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, yva, pva, _ = evaluate(model, val_loader, criterion, device)
        train_m, val_m = binary_metrics(ytr, ptr), binary_metrics(yva, pva)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_m["auc"])
        history["val_auc"].append(val_m["auc"])
        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])

        print(f"Epoch [{epoch}/{n_epochs}]  "
              f"train_loss {train_loss:.4f}  train_auc {train_m['auc']:.4f}  "
              f"val_loss {val_loss:.4f}  val_auc {val_m['auc']:.4f}  "
              f"val_acc {val_m['accuracy']:.2f}%")

        if stopper.step(val_loss, model, epoch):
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch} "
                  f"(val_loss {stopper.best_loss:.4f})")
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch
    return history
