"""
Logistic regression baseline for MiniPlaces scene classification.

A single linear layer on the flattened pixel vector. The softmax is applied
inside nn.CrossEntropyLoss during training, so the model returns raw logits.

Input : batch of images, shape (N, 3, 128, 128)
Output: logits, shape (N, 100)
"""

import torch.nn as nn
import json

IMG_SIZE = 128
NUM_CLASSES = 100


class LogisticRegression(nn.Module):
    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        in_features = 3 * img_size * img_size  # 49,152
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))


def build_model():
    """Entry point used by train.py (--model logreg)."""
    return LogisticRegression()

def plot_history(path="results/logreg_results.json", tag="logreg"):
    import matplotlib.pyplot as plt
    h = json.load(open(path))["history"]
    e = [r["epoch"] for r in h]
    plots = [("loss", "Cross-entropy loss", [("Train", "train_loss", 1), ("Validation", "val_loss", 1)]),
             ("accuracy", "Validation accuracy (%)", [("Top-1", "val_top1", 100), ("Top-5", "val_top5", 100)])]
    for name, ylabel, series in plots:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for label, key, k in series:
            ax.plot(e, [k * r[key] for r in h], marker="o", label=label)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{tag}_{name}.png", dpi=300)

if __name__ == "__main__":
    plot_history()