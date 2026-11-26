# miniplaces-scene-classification

Scene classification on MiniPlaces (100 scene categories, 128x128 RGB images) for
COSC 50523 Deep Learning. Input: one RGB image. Output: one of 100 scene classes.
Metrics: top-1 and top-5 accuracy.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # tested with Python 3.12
```

`train.py` picks CUDA, then Apple MPS, then CPU automatically.

## Data

Get MiniPlaces (the `data.tar.gz` plus the image folders) and extract it so this layout exists
(**TODO: add the dataset download link here**). The dataset is not stored in this repo.

```
<data_root>/
    data/train.txt        # "train/a/abbey/00000001.jpg 0"
    data/val.txt          # "val/00000001.jpg 71"
    data/categories.txt   # "/a/abbey 0"
<image_root>/             # defaults to <data_root>/images if it exists, else <data_root>
    train/<letter>/<category>/<image>.jpg
    val/<image>.jpg
```

If the label files and the images live in different folders, pass `--data_root` (labels) and
`--image_root` (images) separately.

**Splits (no leakage).** MiniPlaces' official test set has no public labels, so it is not used.
The official `val/` (10,000 images) is our held-out **test** set. Our own **validation** set is a
stratified 10% of the 100,000 training images (90,000 train / 10,000 val), seeded for
reproducibility. `data_pipeline.py` asserts that no image appears in more than one split.
Model selection uses validation only; the test set is evaluated once, with the best-validation checkpoint.

## Models

`train.py --model <name>` takes either a built-in (`logreg`, `mlp`, `cnn`, defined in `train.py`)
or any `models/<name>.py` that defines `build_model()` returning an `nn.Module`.

`models/mlp_baseline.py` is the MLP baseline: flatten (3x128x128 = 49,152) -> Linear(512) -> ReLU
-> Linear(100), 25,217,636 parameters. (The built-in `mlp` in `train.py` is a different, deeper
MLP with dropout.)

## Reproduce the MLP baseline result

```
python train.py --data_root /path/to/miniplaces --model mlp_baseline \
    --lr 1e-4 --epochs 5 --out_dir results/lr1e-4
```

Everything else uses the `train.py` defaults: batch size 128, AdamW with weight decay 1e-4,
seed 42, 10% validation fraction. The run writes `results/lr1e-4/mlp_baseline_results.json`
(arguments, per-epoch history, final val/test metrics) and the best checkpoint
`mlp_baseline_best.pt` (git-ignored). Note that `train.py`'s default learning rate is 1e-3,
which is unstable for this model (below), so always pass `--lr 1e-4`. Exact numbers may differ
slightly across hardware.

### Figures and error analysis

```
python plot_results.py                        # curves for the official run
python plot_results.py --name mlp_lr_comparison \
    --results results/lr1e-3/mlp_baseline_results.json \
              results/lr1e-4/mlp_baseline_results.json \
    --labels "lr = 1e-3" "lr = 1e-4"          # overlay the two learning rates

python error_analysis.py --data_root /path/to/miniplaces --model mlp_baseline \
    --checkpoint results/lr1e-4/mlp_baseline_best.pt
```

`error_analysis.py` re-evaluates the checkpoint on the test set (and checks that it matches the
saved result), then writes a confusion matrix, per-class accuracy, the most confused class pairs,
and example success and failure images. It is post-hoc analysis and not used for tuning.

## Results: MLP baseline (5 epochs, seed 42)

| Learning rate | Val top-1 | Val top-5 | Test top-1 | Test top-5 | Test loss |
|---|---|---|---|---|---|
| 1e-3 (`results/lr1e-3/`) | 5.7% | 18.1% | 6.0% | 18.7% | 6.61 |
| **1e-4** (`results/lr1e-4/`) | **8.9%** | **26.5%** | **9.4%** | **26.4%** | **4.45** |

Chance is 1% top-1 and 5% top-5. At lr 1e-3 the training loss stays above the untrained value
(about 4.6) and rises after epoch 3; at lr 1e-4 it decreases steadily. The lr 1e-4 run was chosen
on validation accuracy. Curves: `figures/mlp_lr_comparison_loss.png`,
`figures/mlp_lr_comparison_val_accuracy.png`, `figures/mlp_baseline_loss.png`,
`figures/mlp_baseline_val_accuracy.png`.

Error analysis of the lr 1e-4 model on the test set (`results/lr1e-4/mlp_baseline_error_analysis.json`,
`figures/mlp_baseline_{confusion,successes,failures}.png`):

- Best classes: `golf_course` and `bamboo_forest` (52% each). Several classes get about 0%
  (`beauty_salon`, `coffee_shop`, `hot_spring`, `valley`).
- Many indoor scenes are predicted as `stage/indoor` (for example `auditorium`, `bar`,
  `boxing_ring`, `bowling_alley`, `ballroom`), which shows up as a vertical stripe in the
  confusion matrix.

## Repository layout

```
train.py            train / evaluate any model, writes results/<out_dir>/
data_pipeline.py    MiniPlaces dataset, splits, dataloaders
models/             custom models (one file per model, must define build_model())
plot_results.py     training-curve figures
error_analysis.py   confusion matrix, per-class accuracy, example images
results/            results JSON per run (checkpoints are git-ignored)
figures/            generated figures
```
