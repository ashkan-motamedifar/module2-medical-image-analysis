# Module 2 — AI-driven Medical Image Analysis

Exercise for Module 2 of a CS summer school (Odense): two medical imaging tasks
solved with PyTorch from scratch — no pre-built training frameworks.

- **Task 1 — Polyp classification.** ResNet18 / EfficientNet-B0 / MobileNetV3-Large,
  transfer-learned on confidential capsule-endoscopy (CCE) images, binary
  neoplasia vs. non-neoplasia.
- **Task 2 — Membrane segmentation.** U-Net / Attention U-Net / U-Net++ trained from
  scratch on the ISBI 2012 EM segmentation challenge, with a loss-function ablation
  (BCE vs. Dice vs. Dice+BCE vs. IoU+BCE).

The lecturer's starter notebooks had real bugs (see below) — finding and fixing
those, and fixing a patient-level data leak in the polyp split, was most of the
actual work.

## Results

Single run, seed 27 (see [Known bugs](#known-bugs-in-the-lecturers-originals) and
the [CCE polyp](#cce-polyp-task-1--present) section for why these numbers need the
caveats listed there; multi-seed runs are in `notebooks/colab_task2_3seeds.ipynb`).

**Task 1 — polyp classification** (test set, 319 images)

| Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| ResNet18 | 81.8% | 81.0% | 83.3% | 85.1% | 0.902 |
| EfficientNet-B0 | 81.8% | 80.5% | 84.2% | 85.1% | 0.894 |
| MobileNetV3-Large | 80.3% | 83.4% | 74.6% | 84.4% | 0.890 |

**Task 2 — membrane segmentation** (5-image held-out split, Dice+BCE loss)

| Model | Jaccard (IoU) | F1 / Dice | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| U-Net | 0.711 | 0.831 | 0.848 | 0.815 | 0.924 |
| Attention U-Net | 0.711 | 0.831 | 0.850 | 0.813 | 0.924 |
| U-Net++ | 0.700 | 0.824 | 0.826 | 0.822 | 0.922 |

**Task 2 — loss ablation** (U-Net, same split)

| Loss | Jaccard (IoU) | F1 / Dice | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| BCE | 0.703 | 0.825 | 0.814 | 0.837 | 0.924 |
| Dice | 0.716 | 0.834 | 0.869 | 0.803 | 0.924 |
| Dice + BCE | 0.704 | 0.826 | 0.829 | 0.824 | 0.924 |
| IoU + BCE | 0.701 | 0.824 | 0.847 | 0.803 | 0.921 |

Figures (ROC, confusion matrix, qualitative segmentation masks) are in
[`colab_results/`](colab_results) and [`rerun_results/`](rerun_results); the raw
per-run `config.json` / `history.json` / `metrics.json` are there too (model
checkpoints are gitignored — too big for git).

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

`torch` is pinned to 2.2.2, which was compiled against NumPy 1.x — hence `numpy<2` and
the matching `scipy==1.13.1` / `scikit-learn==1.5.2`. Bumping any one of those alone
breaks the import.

Open notebooks from anywhere — each one's first code cell walks up to the repo root
so `src/` and `data/`/`PCData/` resolve regardless of launch directory.

## Notebooks

All in [`notebooks/`](notebooks); the lecturer's untouched originals are in
[`notebooks/originals/`](notebooks/originals).

| Notebook | Task | Status |
|---|---|---|
| `notebooks/colab_module2.ipynb` | **both tasks, all 6 models** | run this on a GPU |
| `notebooks/01_mlp_organamnist.ipynb` | warm-up | runs |
| `notebooks/02_alexnet_organamnist.ipynb` | warm-up | runs |
| `notebooks/03_polyp_mobilenet.ipynb` | Task 1, single model | runs on real data |
| `notebooks/04_segmentation_attention_unet.ipynb` | Task 2, single model | runs on real data |

### Running on Colab

Rebuild the code bundle after any change to `src/`:

```bash
zip -r module2_code.zip src -x "*__pycache__*"
```

Put in your Drive root: `module2_code.zip`, and `module2/` containing `PCData/` plus the
three `*_data.txt` files. Open `notebooks/colab_module2.ipynb`, pick a T4 GPU, run all.
The membrane data downloads itself inside the notebook.

Set `SEEDS = [27, 1, 2]` for mean ± std instead of a single run (~2.5 h vs ~50 min).

Keep the Drive folder holding `PCData` private — no link sharing. It is patient data.

The files in [`notebooks/originals/`](notebooks/originals)
(`classification_*_medmnist.ipynb`, `polypcharacterization_cnns.ipynb`,
`segmentation_unet_membrane.ipynb`, `unet.py`) are the lecturer's originals, kept
untouched for reference.

## `src/`

```
seed.py            seeds (random, numpy, torch) + device pick (cuda > mps > cpu)
data.py            MedMNIST loaders
models.py          MLP, AlexNet
training.py        multiclass train/eval loops, EarlyStopping, fit()
metrics.py         binary + multiclass metrics, ROC, confusion matrix, threshold picking
losses.py          BCE, Dice, Dice+BCE, IoU+BCE, MSE
attention_unet.py  padded U-Net and Attention U-Net (shared code, one flag)
membrane.py        ISBI 2012 loading, random crops, flips/rotations
segmentation.py    segmentation train/eval loops + Table 2 metrics
polyp.py           CCE dataset, model heads, binary train/eval loops
results.py         save_run() -> results/<timestamp>_<tag>/, markdown table helper
```

## Data

### Membrane (Task 2) — present

`data/train/images/` (30), `data/train/labels/` (30), `data/test/` (30).

Source: ISBI 2012 EM segmentation challenge, public copy from
`github.com/zhixuhao/unet` (`data/membrane`), downloaded 2026-08-06. The repo's
`*_predict.png` files were excluded — those are that author's model outputs, not data.

Two things to state in the report:

- The official `test/` folder has **no ground truth** (labels withheld for the
  challenge). Metrics therefore come from a 20/5/5 split of the 30 *labelled* images;
  the official test images are used only for qualitative figures.
- In the source files membranes are black (0) on white (255). `load_membrane`
  auto-detects and inverts so membrane = positive class. Measured foreground: 22.0%.

If the instructor sends a different copy, replace `data/` and re-run — nothing else changes.

### CCE polyp (Task 1) — present

2082 jpgs: `PCData/Neoplasia/` (1398) and `PCData/Non-neoplasia/` (684), plus
`train_data.txt` / `valid_data.txt` / `test_data.txt`. Confidential patient data,
`.gitignore`d. Never commit it or upload it anywhere public.

**The split key is the polyp, not the patient.** Files are named
`<patient>_P<polyp>_<frame>.jpg` and split lines are `<patient>,<polyp>,<class>`, so the
key is e.g. `12485_P1`. The lecturer's original notebook matches on
`filename.split('_')[0]` — the patient id alone — which is wrong: patient 12485 has P1
(Neoplasia) in train and P2 (Non-neoplasia) in test, so all ten of that patient's images
land in both splits. Across the dataset that leaks **355 of 569 test images into
training**. `load_polyp_splits` keys on the pair and raises if any split overlaps.

Resulting split (0 overlap, all 2082 images used):

| Split | Images | Neoplasia | Non-neoplasia | % positive |
|---|---|---|---|---|
| train | 1446 | 987 | 459 | 68.3 |
| valid | 317 | 206 | 111 | 65.0 |
| test | 319 | 205 | 114 | 64.3 |

Caveat to state in the report: 22 patients have polyps in both train and test. The
instructor's split files are polyp-level, not patient-level.

### MedMNIST — downloads itself to `~/.medmnist/`.

## Measured runtimes (M1, MPS)

| Job | Per epoch |
|---|---|
| MLP, 28x28 | 7.6 s |
| AlexNet, 224x224, 34k images | 3.5 min |
| Attention U-Net, 256 crops, repeats=8 | ~55 s |

## Known bugs in the lecturer's originals

`notebooks/originals/segmentation_unet_membrane.ipynb`:

- `for e in range(epoch)` — `epoch` is undefined, should be `n_epochs`.
- `train_iter = next(iter(train_dl))` inside the loop rebuilds the iterator every step,
  so it trains on the **first batch only**, forever.
- Uses `F.mse_loss`; the exercise wants Dice / BCE / Dice+BCE with a justification.
- Images and masks are loaded into two *separately shuffled-free* loaders and zipped by
  position, which only works because `shuffle=False`.

`notebooks/originals/polypcharacterization_cnns.ipynb`:

- `device = 'cuda' if ... else 'mps'` — no CPU fallback.
- `resnet18(pretrained=True)` is deprecated; use `weights=`.
- Decision threshold hardcoded to 0.65 with no justification.

Both classification originals report `loss.item()` of the last batch instead of the
epoch average, and none of the four implement the early stopping the exercise requires.

## Not in this repo

- `PCData/` and the `*_data.txt` split files — confidential patient data, gitignored.
- `Module2-Part1.pdf`, `Module2-Part2.pdf`, the clinical-validation `.pptx` — the
  instructor's lecture material, not this author's work.
- Model checkpoints (`*.pt`) — large, regenerable by rerunning the notebooks.

## Author

Ashkan Motamedifar — code, bug fixes, and analysis in this repo are the author's
own work for the course exercise. The exercise brief and lecture slides belong to
the course instructors.
