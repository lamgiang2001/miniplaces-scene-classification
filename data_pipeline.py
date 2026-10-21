"""
MiniPlaces data pipeline.

Leak-free protocol (matches the project proposal):
  - MiniPlaces' official test/ set has no public labels, so it is NOT used here.
  - The official val/ set (10,000 images) becomes our held-out TEST set.
  - Our own VALIDATION split is carved out of the 100,000 training images,
    stratified by class, and used for model selection / hyperparameter tuning.

Expected directory layout (after extracting data.tar.gz next to this script,
or pass --data_root to point elsewhere):

    <data_root>/
        train/<letter>/<category>/<image>.jpg
        val/<image>.jpg
        data/train.txt      # "train/a/abbey/00000001.jpg 0"
        data/val.txt         # "val/00000001.jpg 3"
        data/categories.txt  # "/abbey 0"
"""

import argparse
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def read_label_file(label_file_path):
    """Read a MiniPlaces label file (e.g. train.txt) into a list of (relative_path, label) pairs."""
    pairs = []
    with open(label_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label_str = line.rsplit(" ", 1)
            pairs.append((rel_path, int(label_str)))
    return pairs


def stratified_split(pairs, val_fraction=0.1, seed=42):
    """
    Split (path, label) pairs into train/val, stratified by label, so every
    class is represented proportionally in both splits. This is our own
    validation set, separate from MiniPlaces' official val/ (which we use
    as the test set instead).
    """
    by_class = defaultdict(list)
    for path, label in pairs:
        by_class[label].append((path, label))

    train_pairs, val_pairs = [], []
    rng_state = seed
    for label, items in by_class.items():
        # simple deterministic shuffle per class so the split is reproducible
        items_sorted = sorted(items)  # stable base ordering
        import random
        rng = random.Random(rng_state + label)
        rng.shuffle(items_sorted)

        n_val = max(1, int(len(items_sorted) * val_fraction))
        val_pairs.extend(items_sorted[:n_val])
        train_pairs.extend(items_sorted[n_val:])

    return train_pairs, val_pairs


class MiniPlacesDataset(Dataset):
    """A simple image dataset over (relative_path, label) pairs."""

    def __init__(self, image_root, pairs, transform=None):
        self.image_root = image_root
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rel_path, label = self.pairs[idx]
        full_path = os.path.join(self.image_root, rel_path)
        image = Image.open(full_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms():
    # Normalization stats are ImageNet's standard values; fine as a default
    # starting point even for from-scratch models, and necessary if you
    # later fine-tune any ImageNet-pretrained backbone.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def build_dataloaders(data_root, batch_size=128, val_fraction=0.1, num_workers=4, seed=42,
                      image_root=None):
    if image_root is None:
        images_dir = os.path.join(data_root, "images")
        image_root = images_dir if os.path.isdir(images_dir) else data_root

    train_label_file = os.path.join(data_root, "data", "train.txt")
    official_val_label_file = os.path.join(data_root, "data", "val.txt")

    all_train_pairs = read_label_file(train_label_file)
    official_val_pairs = read_label_file(official_val_label_file)  # -> our TEST set

    train_pairs, val_pairs = stratified_split(all_train_pairs, val_fraction=val_fraction, seed=seed)

    # Leakage check: confirm no image path appears in more than one split.
    train_set = {p for p, _ in train_pairs}
    val_set = {p for p, _ in val_pairs}
    test_set = {p for p, _ in official_val_pairs}
    assert train_set.isdisjoint(val_set), "Leakage between train and val!"
    assert train_set.isdisjoint(test_set), "Leakage between train and test!"
    assert val_set.isdisjoint(test_set), "Leakage between val and test!"

    train_transform, eval_transform = build_transforms()

    train_dataset = MiniPlacesDataset(image_root, train_pairs, transform=train_transform)
    val_dataset = MiniPlacesDataset(image_root, val_pairs, transform=eval_transform)
    test_dataset = MiniPlacesDataset(image_root, official_val_pairs, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} images | "
          f"Val (ours, for tuning): {len(val_dataset)} images | "
          f"Test (official val/, held out): {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MiniPlaces dataloaders.")
    parser.add_argument("--data_root", type=str, default=".",
                         help="Path containing the data/ label files and image folders.")
    parser.add_argument("--image_root", type=str, default=None,
                         help="Root for paths listed in the label files. "
                              "Defaults to <data_root>/images if it exists, else data_root.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="Fraction of the 100k training images held out as our own validation split.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        image_root=args.image_root,
    )

    # Quick sanity check: pull one batch and print its shape.
    images, labels = next(iter(train_loader))
    print(f"Sample batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}")