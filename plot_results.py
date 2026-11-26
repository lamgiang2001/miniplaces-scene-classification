"""
Plot training curves from one or more train.py results files.

For the given runs it makes two line plots from the "history" in each results JSON:
  1. Validation accuracy (top-1 solid, top-5 dashed, in %) per epoch
  2. Loss (validation solid, train dashed) per epoch

Run from the repo root. Examples:
  python plot_results.py                  # official MLP baseline (lr 1e-4)
  python plot_results.py --name mlp_lr_comparison \
      --results results/lr1e-3/mlp_baseline_results.json \
                results/lr1e-4/mlp_baseline_results.json   # overlay several runs
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DEFAULT_RESULTS = "results/lr1e-4/mlp_baseline_results.json"


def load_run(path, label=None):
    with open(path) as f:
        result = json.load(f)
    a = result["args"]
    return {
        "model": a["model"],
        "label": label or f"{a['model']} (lr={a['lr']:g})",
        "history": result["history"],
    }


def save(fig, ax, path, ylabel, title):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from train.py results.")
    parser.add_argument("--results", nargs="+", default=[DEFAULT_RESULTS],
                        help="One or more <model>_results.json files written by train.py.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend label for each results file (default: model and lr).")
    parser.add_argument("--name", default=None,
                        help="Output file prefix (default: the model name for one run, "
                             "'comparison' for several).")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.results):
        parser.error("--labels needs exactly one label per --results file")

    labels = args.labels or [None] * len(args.results)
    runs = [load_run(path, label) for path, label in zip(args.results, labels)]
    single = len(runs) == 1
    name = args.name or (runs[0]["model"] if single else "comparison")
    prefix = f"{runs[0]['model']}: " if single else ""
    os.makedirs(args.out_dir, exist_ok=True)

    # Plot 1: validation accuracy per epoch
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, run in enumerate(runs):
        epochs = [row["epoch"] for row in run["history"]]
        for key, style, metric in (("val_top1", "-", "top-1"), ("val_top5", "--", "top-5")):
            label = f"Validation {metric}" if single else f"{run['label']}: {metric}"
            ax.plot(epochs, [100 * row[key] for row in run["history"]],
                    style, color=f"C{i}", marker="o", label=label)
    save(fig, ax, os.path.join(args.out_dir, f"{name}_val_accuracy.png"),
         "Accuracy (%)", f"{prefix}Validation accuracy per epoch")

    # Plot 2: loss per epoch
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, run in enumerate(runs):
        epochs = [row["epoch"] for row in run["history"]]
        for key, style, split in (("val_loss", "-", "validation"), ("train_loss", "--", "train")):
            label = f"{split.capitalize()} loss" if single else f"{run['label']}: {split}"
            ax.plot(epochs, [row[key] for row in run["history"]],
                    style, color=f"C{i}", marker="o", label=label)
    save(fig, ax, os.path.join(args.out_dir, f"{name}_loss.png"),
         "Cross-entropy loss", f"{prefix}Loss per epoch")


if __name__ == "__main__":
    main()
