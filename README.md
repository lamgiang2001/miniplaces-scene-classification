# MiniPlaces scene classification

COSC 50523 Deep Learning course project: classify 128x128 images into the 100 scene
categories of MiniPlaces. This branch holds the MLP baseline.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # tested with Python 3.12
```

## Data

Extract MiniPlaces so these files exist (see `data_pipeline.py` for details):

```
<data_root>/data/train.txt, val.txt, categories.txt
<image_root>/train/<letter>/<category>/<image>.jpg, val/<image>.jpg
```

`<image_root>` defaults to `<data_root>/images` if that folder exists, else `<data_root>`.
Pass `--image_root` if the images live somewhere else.

**Splits.** The official `val/` set (10,000 images) is our held-out **test** set. Our
**validation** set is a seeded, class-stratified 10% of the training images (90,000 train /
10,000 val). `data_pipeline.py` asserts that no image appears in two splits.

## Train the MLP baseline

```
python train.py --data_root /path/to/miniplaces --model mlp_baseline \
    --lr 1e-4 --epochs 5 --out_dir results/lr1e-4
```

The model is `models/mlp_baseline.py`: flatten -> Linear(512) -> ReLU -> Linear(100), 25.2M
parameters. Use `--lr 1e-4`; `train.py`'s default of 1e-3 does not train this model stably.
Other settings are the defaults (batch size 128, AdamW, weight decay 1e-4, seed 42). The run
writes `results/lr1e-4/mlp_baseline_results.json` and a checkpoint (git-ignored).

## Figures and error analysis

```
python plot_results.py      # figures/val_accuracy.png, loss.png (official run), lr_comparison.png

python error_analysis.py --data_root /path/to/miniplaces --model mlp_baseline \
    --checkpoint results/lr1e-4/mlp_baseline_best.pt
```

`error_analysis.py` evaluates on the test set after training; do not use it to tune.

## Results (test set, 5 epochs)

| Learning rate | Top-1 | Top-5 |
|---|---|---|
| 1e-3 (`results/lr1e-3/`) | 6.0% | 18.7% |
| 1e-4 (`results/lr1e-4/`) | 9.4% | 26.4% |

## Files

```
train.py            train and evaluate a model
data_pipeline.py    dataset, splits, dataloaders
models/             custom models (define build_model())
plot_results.py     training-curve figures
error_analysis.py   per-class accuracy, confusions, example images
plot_style.py       shared figure style
results/            results JSON per run
figures/            generated figures
```
