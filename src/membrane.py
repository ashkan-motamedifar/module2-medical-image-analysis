"""Neuronal membrane dataset (ISBI 2012) loading for Task 2.

Two things about this dataset drive every choice here:

1. It has **30 labelled images total**. That is tiny, so training uses random 256x256
   crops plus flips and 90-degree rotations. Evaluation uses the full image.
2. The official `test/` folder has **no ground truth** — labels were withheld for the
   challenge. So the metrics table cannot come from it. We split the 30 labelled
   images into train/val/test instead, and keep the official test images only for
   qualitative figures.
"""

import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _numeric_key(path):
    parts = re.split(r"(\d+)", os.path.basename(path))
    return [int(p) if p.isdigit() else p for p in parts]


def _load_gray(path):
    from PIL import Image
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _find(directory):
    files = []
    for ext in ("tif", "tiff", "png", "jpg"):
        files += glob.glob(os.path.join(directory, f"*.{ext}"))
    if not files:
        raise FileNotFoundError(f"no images in {directory}")
    return sorted(files, key=_numeric_key)


class MembraneDataset(Dataset):
    """`crop_size=None` returns the full image (use for validation and test)."""

    def __init__(self, images, masks, crop_size=256, augment=True, seed=27, repeats=1):
        self.images = images
        self.masks = masks
        self.crop_size = crop_size
        self.augment = augment
        self.repeats = repeats
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, idx):
        idx = idx % len(self.images)
        image, mask = self.images[idx], self.masks[idx]

        if self.crop_size:
            h, w = image.shape
            top = self.rng.integers(0, h - self.crop_size + 1)
            left = self.rng.integers(0, w - self.crop_size + 1)
            image = image[top:top + self.crop_size, left:left + self.crop_size]
            mask = mask[top:top + self.crop_size, left:left + self.crop_size]

        if self.augment:
            k = self.rng.integers(0, 4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
            if self.rng.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if self.rng.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)

        return (torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0))


def load_membrane(root="data", crop_size=256, batch_size=2, splits=(20, 5, 5),
                  invert_masks=None, seed=27, repeats=8, include_train_eval=False):
    """Returns (train_loader, val_loader, test_loader), or a 4th loader if
    `include_train_eval` — the same training images at full resolution with no crops and
    no augmentation. The exercise asks for metrics on the training set, because the
    official membrane test folder ships no masks.

    `repeats` draws that many random crops per image per epoch. With 20 images and
    batch 2 an epoch would otherwise be 10 gradient steps, and early stopping with
    patience 5 would trigger on noise long before the model converged.

    `invert_masks=None` auto-detects polarity: in the ISBI files membranes are black
    (0) on white (255) background, but we want the membrane to be the positive class.
    """
    image_dir = os.path.join(root, "train", "images")
    mask_dir = os.path.join(root, "train", "labels")

    images = [_load_gray(p) for p in _find(image_dir)]
    masks = [_load_gray(p) for p in _find(mask_dir)]
    if len(images) != len(masks):
        raise ValueError(f"{len(images)} images but {len(masks)} masks")

    if invert_masks is None:
        invert_masks = np.mean([m.mean() for m in masks]) > 0.5
    masks = [(1.0 - m if invert_masks else m) for m in masks]
    masks = [(m > 0.5).astype(np.float32) for m in masks]

    n_train, n_val, n_test = splits
    if n_train + n_val + n_test != len(images):
        raise ValueError(f"splits {splits} do not sum to {len(images)} images")

    order = np.random.default_rng(seed).permutation(len(images))
    bounds = [0, n_train, n_train + n_val, len(images)]
    parts = [order[bounds[i]:bounds[i + 1]] for i in range(3)]

    print(f"membrane: {len(images)} labelled images, "
          f"foreground = {np.mean([m.mean() for m in masks]) * 100:.1f}% of pixels")
    print(f"  train {len(parts[0])} | val {len(parts[1])} | test {len(parts[2])}"
          f"  (masks {'inverted' if invert_masks else 'as-is'})")

    loaders = []
    for i, (name, crop, augment) in enumerate(
            [("train", crop_size, True), ("val", None, False), ("test", None, False)]):
        dataset = MembraneDataset([images[j] for j in parts[i]],
                                  [masks[j] for j in parts[i]],
                                  crop_size=crop, augment=augment, seed=seed + i,
                                  repeats=repeats if name == "train" else 1)
        loaders.append(DataLoader(dataset, batch_size=batch_size if name == "train" else 1,
                                  shuffle=(name == "train")))

    if include_train_eval:
        train_eval = MembraneDataset([images[j] for j in parts[0]],
                                     [masks[j] for j in parts[0]],
                                     crop_size=None, augment=False, seed=seed)
        loaders.append(DataLoader(train_eval, batch_size=1, shuffle=False))

    return tuple(loaders)


def load_unlabelled_test(root="data"):
    """The official test images, for qualitative figures only. Returns a [N,1,H,W] tensor."""
    images = [_load_gray(p) for p in _find(os.path.join(root, "test"))]
    return torch.from_numpy(np.stack(images)).unsqueeze(1)
