"""
Plot MLP training results (15 epochs).

Makes two line plots from results/lr1e-4/mlp_baseline_results.json:
  1. Validation accuracy (top-1 and top-5) per epoch
  2. Loss (train and validation) per epoch
and a third figure comparing the two learning rates we tested
(results/lr1e-3 and results/lr1e-4):
  3. Validation loss and validation top-1 accuracy per epoch, lr 1e-3 vs lr 1e-4
and a fourth figure for the qualitative analysis (needs the saved model and the images):
  4. 4 correct and 4 wrong predictions on random test images (figures/examples.png)

Run from the repo root:  python plot_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data_pipeline import MiniPlacesDataset, build_transforms, read_label_file
from train import build_model

with open("results/lr1e-4/mlp_baseline_results.json") as f:
    results = json.load(f)
history = results["history"]

epochs = [row["epoch"] for row in history]
os.makedirs("figures", exist_ok=True)

# Plot 1: validation accuracy per epoch
plt.figure(figsize=(6, 4))
plt.plot(epochs, [100 * row["val_top1"] for row in history], marker="o", label="Validation top-1")
plt.plot(epochs, [100 * row["val_top5"] for row in history], marker="o", label="Validation top-5")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("MLP baseline: Validation accuracy per epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/val_accuracy.png", dpi=200)
plt.close()

# Plot 2: loss per epoch
plt.figure(figsize=(6, 4))
plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("MLP baseline: Loss per epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/loss.png", dpi=200)
plt.close()

# Plot 3: learning-rate comparison (validation loss and top-1 accuracy per epoch)
runs = {
    "lr = 1e-3": "results/lr1e-3/mlp_baseline_results.json",
    "lr = 1e-4": "results/lr1e-4/mlp_baseline_results.json",
}
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
for label, path in runs.items():
    with open(path) as f:
        run_history = json.load(f)["history"]
    run_epochs = [row["epoch"] for row in run_history]
    ax_loss.plot(run_epochs, [row["val_loss"] for row in run_history], marker="o", label=label)
    ax_acc.plot(run_epochs, [100 * row["val_top1"] for row in run_history], marker="o", label=label)
for ax, ylabel, title in ((ax_loss, "Validation loss", "Validation loss per epoch"),
                          (ax_acc, "Validation top-1 accuracy (%)", "Validation top-1 accuracy per epoch")):
    ax.set_xticks(run_epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("MLP baseline: learning rate comparison")
fig.tight_layout()
fig.savefig("figures/lr_comparison.png", dpi=200)
plt.close(fig)

print("Saved figures/val_accuracy.png, figures/loss.png and figures/lr_comparison.png")

# Plot 4: qualitative examples. Run the best lr 1e-4 model on 1,000 random test images
# (the official val/ set, seed 42) and show the first 4 it got right and the first 4 it got wrong.
data_root = results["args"]["data_root"]
image_root = results["args"]["image_root"] or (
    os.path.join(data_root, "images") if os.path.isdir(os.path.join(data_root, "images")) else data_root)
checkpoint = "results/lr1e-4/mlp_baseline_best.pt"

# The checkpoint and the images are not in the repo, so skip this figure if either is missing.
if not (os.path.isfile(checkpoint) and os.path.isdir(image_root)):
    print(f"Skipped figures/examples.png: needs {checkpoint} and the MiniPlaces images at {image_root}")
else:
    # Class names: "/c/church/outdoor 12" -> "church/outdoor"
    class_names = {}
    with open(os.path.join(data_root, "data", "categories.txt")) as f:
        for line in f:
            path, idx = line.strip().rsplit(" ", 1)
            class_names[int(idx)] = path.lstrip("/").split("/", 1)[-1]

    _, eval_transform = build_transforms()
    test_pairs = read_label_file(os.path.join(data_root, "data", "val.txt"))
    test_set = MiniPlacesDataset(image_root, test_pairs, transform=eval_transform)

    sample = np.random.default_rng(42).permutation(len(test_set))[:1000]
    images = torch.stack([test_set[i][0] for i in sample])
    labels = np.array([test_set[i][1] for i in sample])

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model("mlp_baseline").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        probs = model(images.to(device)).softmax(dim=1).cpu()
    conf, preds = probs.max(dim=1)
    conf, preds = conf.numpy(), preds.numpy()

    correct = np.where(preds == labels)[0][:4]
    wrong = np.where(preds != labels)[0][:4]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.8))
    for ax, k in zip(axes.flat, list(correct) + list(wrong)):
        rel_path, label = test_pairs[sample[k]]
        ax.imshow(Image.open(os.path.join(image_root, rel_path)).convert("RGB"))
        ax.set_title(f"true: {class_names[label]}\npred: {class_names[preds[k]]} ({conf[k]:.2f})", fontsize=9)
        ax.axis("off")
    axes[0, 0].text(-0.06, 0.5, "Correct", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11)
    axes[1, 0].text(-0.06, 0.5, "Wrong", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle("MLP baseline: example predictions on random test images")
    fig.tight_layout(h_pad=2.5)
    fig.savefig("figures/examples.png", dpi=200)
    plt.close(fig)
    print("Saved figures/examples.png")
