"""
Plot MLP training results (5 epochs).

Makes two line plots from results in mlp_model.py:
  1. Validation accuracy (top-1 and top-5) per epoch
  2. Loss (train and validation) per epoch
"""

import json
import os
import matplotlib.pyplot as plt

with open("results/mlp_baseline_results.json") as f:
    history = json.load(f)["history"]

epochs = [row["epoch"] for row in history]
os.makedirs("figures", exist_ok=True)

# Plot 1: validation accuracy per epoch
plt.figure(figsize=(6, 4))
plt.plot(epochs, [row["val_top1"] for row in history], marker="o", label="Validation top-1")
plt.plot(epochs, [row["val_top5"] for row in history], marker="o", label="Validation top-5")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
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

print("Saved figures/val_accuracy.png and figures/loss.png")