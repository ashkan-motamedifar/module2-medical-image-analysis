"""Recompute the Table 2 numbers from saved segmentation weights.

The exercise asks for segmentation metrics on the **training** images, because the
official membrane test folder ships no ground-truth masks. Training is done in Colab
(`colab_task2_rerun.ipynb`), which reports the held-out numbers; this script reloads
the weights that run saved and evaluates them on the training images at full
resolution, with no crops and no augmentation.

    python src/eval_train_set.py rerun_results/results

Needs the `model.pt` files written by the training run. They are ~120 MB each and are
not included with the submitted code; re-run the Colab notebook to regenerate them.
"""

import json
import sys
from pathlib import Path

import torch

from .attention_unet import AttentionUNet, UNet
from .losses import LOSSES
from .membrane import load_membrane
from .results import results_table
from .seed import get_device, set_seed
from .segmentation import evaluate
from .unetpp import UNetPlusPlus

SEED = 27
METRICS = ["jaccard", "f1", "recall", "precision", "accuracy"]

BUILD = {
    "attention_unet": lambda: AttentionUNet(base=64),
    "unet": lambda: UNet(base=64),
    "unetpp": lambda: UNetPlusPlus(base=64),
}
LABEL = {"attention_unet": "Attention U-Net", "unet": "U-Net", "unetpp": "U-Net++"}


def main(results_dir):
    set_seed(SEED)
    device = get_device()
    *_, train_eval = load_membrane(root="data", crop_size=256, batch_size=4,
                                   repeats=8, seed=SEED, include_train_eval=True)
    criterion = LOSSES["dice_bce"]()

    train_rows, held_rows = {}, {}
    for run in sorted(Path(results_dir).glob("*_rerun")):
        # Match the longest name first: plain "unet" is a substring of "unetpp",
        # so a naive search builds the wrong architecture for the U-Net++ weights.
        key = next((k for k in sorted(BUILD, key=len, reverse=True) if k in run.name), None)
        if key is None:
            continue
        weights = run / "model.pt"
        if not weights.exists():
            print(f"  skipping {run.name}: no model.pt")
            continue

        model = BUILD[key]().to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        train_rows[LABEL[key]], _ = evaluate(model, train_eval, criterion, device)
        held_rows[LABEL[key]] = json.load(open(run / "metrics.json"))
        print(f"  evaluated {LABEL[key]}")

    if not train_rows:
        print("no weights found — nothing to do")
        return

    print("\nTABLE 2 — training set (this is what the report quotes)")
    print(results_table(train_rows, METRICS, "{:.4f}"))
    print("\nHeld-out 5 images, for the discussion")
    print(results_table(held_rows, METRICS, "{:.4f}"))

    out = Path(results_dir).parent / "train_set_metrics.json"
    out.write_text(json.dumps(train_rows, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rerun_results/results")
