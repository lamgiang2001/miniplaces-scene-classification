"""
Logistic regression baseline for MiniPlaces scene classification.

A single linear layer on the flattened pixel vector. The softmax is applied
inside nn.CrossEntropyLoss during training, so the model returns raw logits.

Input : batch of images, shape (N, 3, 128, 128)
Output: logits, shape (N, 100)
"""

import torch
import torch.nn as nn

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


if __name__ == "__main__":
    # Sanity check on fake data: no dataset needed.
    model = LogisticRegression()
    fake_images = torch.randn(8, 3, IMG_SIZE, IMG_SIZE)
    logits = model(fake_images)
    n_params = sum(p.numel() for p in model.parameters())
    print("output shape:", tuple(logits.shape))   # expect (8, 100)
    print("parameters:", f"{n_params:,}")          # expect 4,915,300
    assert logits.shape == (8, NUM_CLASSES)
    print("OK")