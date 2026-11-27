"""
Plot MLP training results (15 epochs)

Makes 4 different figures: 
  1. Validation accuracy (top-1 and top-5) per epoch
  2. Loss (training and validation) per epoch
  3. Learning rate comparison: Validation loss and validation top-1 accuracy per epoch, comparing 0.001 and 0.0001 learning rates
  4. Qualitative analysis: 4 correct and 4 wrong predictions on random test images

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

# Plot 1:  Validation accuracy (top-1 and top-5) per epoch
plt.figure(figsize=(6, 4))
plt.plot(epochs, [100 * row["val_top1"] for row in history], marker="o", label="Validation top-1")
plt.plot(epochs, [100 * row["val_top5"] for row in history], marker="o", label="Validation top-5")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("MLP Baseline: Validation Accuracy Per Epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/mlp_val_accuracy.png", dpi=200)
plt.close()

# Plot 2: Loss (training and validation) per epoch
plt.figure(figsize=(6, 4))
plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("MLP Baseline: Loss per epoch")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/mlp_loss.png", dpi=200)
plt.close()

# Plot 3: Learning rate comparison: Validation loss and validation top-1 accuracy per epoch, comparing 0.001 and 0.0001 learning rates
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
fig.suptitle("MLP baseline: Learning Rate Comparison")
fig.tight_layout()
fig.savefig("figures/lr_comparison.png", dpi=200)
plt.close(fig)

# Plot 4: Qualtitative analysis showing 4 correct predictions and 4 incorrect predictions on random test images 
data_root = results["args"]["data_root"]
image_root = results["args"]["image_root"]
checkpoint = "results/lr1e-4/mlp_baseline_best.pt"
if os.path.isfile(checkpoint) and os.path.isdir(image_root):

    # Get class names
    class_names = {}
    with open(os.path.join(data_root, "data", "categories.txt")) as f:
        for line in f:
            path, idx = line.strip().rsplit(" ", 1)
            class_names[int(idx)] = path.lstrip("/").split("/", 1)[-1]

    # Test dataset
    _, eval_transform = build_transforms()
    test_pairs = read_label_file(os.path.join(data_root, "data", "val.txt"))
    test_set = MiniPlacesDataset(image_root, test_pairs, transform=eval_transform)

    # Select 1,000 random test images
    sample = np.random.default_rng(42).permutation(len(test_set))[:1000]
    images = torch.stack([test_set[i][0] for i in sample])
    labels = np.array([test_set[i][1] for i in sample])

    # Load trained model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model("mlp_baseline").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Make predictions
    with torch.no_grad():
        probs = model(images.to(device)).softmax(dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    # Find and display 4 correct and 4 incorrect predictions
    correct = np.where(preds == labels)[0][:4]
    wrong = np.where(preds != labels)[0][:4]

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))

    for ax, k in zip(axes.flat, np.concatenate([correct, wrong])):
        path, label = test_pairs[sample[k]]
        image = Image.open(os.path.join(image_root, path)).convert("RGB")

        ax.imshow(image)
        ax.set_title(
            f"True: {class_names[label]}\nPred: {class_names[preds[k]]} ({conf[k]:.2f})",
            fontsize=9
        )
        ax.axis("off")

    axes[0, 0].set_ylabel("Correct Predictions", fontsize=11)
    axes[1, 0].set_ylabel("Incorrect Predictions", fontsize=11)

    fig.suptitle("MLP baseline: Example Successes and Failures")
    fig.tight_layout()
    fig.savefig("figures/mlp_qualitative.png", dpi=200)
    plt.close(fig)

else:
    print("Images not found")