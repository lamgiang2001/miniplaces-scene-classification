"""
MLP baseline model for MiniPlaces scene classification.

A hidden layer activated with ReLU, followed by an output layer.

Input : batch of images, shape (N, 3, 128, 128)
Output: logits, shape (N, 100)
"""

import torch
import torch.nn as nn

IMG_SIZE = 128
NUM_CLASSES = 100
HIDDEN_SIZE = 512

class MLPBaseline(nn.Module):
    def __init__(self, 
                 img_size=IMG_SIZE, 
                 num_classes=NUM_CLASSES, 
                 hidden_size=HIDDEN_SIZE):
        super().__init__()

        in_features = 3 * img_size * img_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def build_model():
    return MLPBaseline()

if __name__ == "__main__":
    # Test the model with fake images
    model = MLPBaseline()
    fake_images = torch.randn(8, 3, IMG_SIZE, IMG_SIZE)
    logits = model(fake_images)
    n_params = sum(p.numel() for p in model.parameters())

    print("output shape:", tuple(logits.shape))
    print("parameters:", f"{n_params:,}")
    assert logits.shape == (8, NUM_CLASSES)
    print("OK")
