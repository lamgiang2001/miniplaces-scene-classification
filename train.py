"""
Train and evaluate any model in models/ on MiniPlaces (e.g. logreg, mlp, cnn).

Uses data_pipeline.py (same folder). Reports top-1 / top-5 accuracy on:
  - our validation split (carved from train, used for model selection)
  - the held-out test set (official val/, evaluated once at the end with the
    checkpoint that had the best validation top-1)

Example:
  python train.py --data_root /path/to/miniplaces --model cnn --epochs 15
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from data_pipeline import build_dataloaders

IMG_SIZE = 128
NUM_CLASSES = 100


def build_model(name):
    """
    Load a custom model from models/<name>.py, or build a built-in model.

    Custom model files must define build_model() and return an nn.Module.
    """
    custom_model = Path(__file__).parent / "models" / f"{name}.py"
    if custom_model.is_file():
        module = importlib.import_module(f"models.{name}")
        if not hasattr(module, "build_model"):
            raise ValueError(f"models/{name}.py must define build_model().")
        model = module.build_model()
        if not isinstance(model, nn.Module):
            raise TypeError(f"models/{name}.py build_model() must return a torch.nn.Module.")
        return model

    in_features = 3 * IMG_SIZE * IMG_SIZE
    if name == "logreg":
        return nn.Sequential(nn.Flatten(), nn.Linear(in_features, NUM_CLASSES))
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )
    if name == "cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        return nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES),
        )
    raise ValueError(
        f"Unknown model {name!r}. Use logreg, mlp, cnn, or add models/{name}.py."
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1 = top5 = n = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        top5_idx = logits.topk(5, dim=1).indices
        top1 += (top5_idx[:, 0] == labels).sum().item()
        top5 += (top5_idx == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)
    return {"loss": loss_sum / n, "top1": top1 / n, "top5": top5 / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Folder containing data/train.txt and data/val.txt.")
    p.add_argument("--image_root", default=None,
                   help="Folder the label paths are relative to. "
                        "Default: <data_root>/images if it exists, else data_root.")
    p.add_argument("--model", required=True,
                   help="Built-in model (logreg, mlp, cnn) or a model file in models/.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--val_fraction", type=float, default=0.1,
                   help="Fraction of training images held out for validation (must be between 0 and 1).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--train_subset", type=int, default=0,
                   help="If > 0, train on only this many random training images "
                        "(quick smoke test; report only full runs).")
    args = p.parse_args()
    if args.epochs < 1:
        p.error("--epochs must be at least 1")
    if args.batch_size < 1:
        p.error("--batch_size must be at least 1")
    if not 0 < args.val_fraction < 1:
        p.error("--val_fraction must be between 0 and 1 (exclusive)")
    if args.lr <= 0:
        p.error("--lr must be greater than 0")
    if args.weight_decay < 0:
        p.error("--weight_decay must be non-negative")
    if args.num_workers < 0:
        p.error("--num_workers must be non-negative")
    if args.train_subset < 0:
        p.error("--train_subset must be non-negative")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root, batch_size=args.batch_size,
        val_fraction=args.val_fraction, num_workers=args.num_workers,
        seed=args.seed, image_root=args.image_root)

    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        if len(loader.dataset) == 0:
            raise SystemExit(f"The {split_name} split is empty. Check the label files and --data_root.")

    if args.train_subset > 0:
        from torch.utils.data import DataLoader, Subset
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_loader.dataset), generator=g)[:args.train_subset].tolist()
        train_loader = DataLoader(Subset(train_loader.dataset, idx), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        print(f"SMOKE TEST: training on {len(idx)} images only")

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    history, best_val_top1 = [], -1.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / seen,
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"]}
        history.append(row)
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.3f} | "
              f"val loss {val['loss']:.3f} | val top-1 {val['top1']:.3f} | val top-5 {val['top5']:.3f}")

        if val["top1"] > best_val_top1:
            best_val_top1 = val["top1"]
            torch.save(model.state_dict(), ckpt_path)

    train_minutes = (time.time() - start) / 60

    # Final test evaluation: once, with the checkpoint chosen on validation only.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    val_final = evaluate(model, val_loader, device)
    test_final = evaluate(model, test_loader, device)
    print(f"\nBest-val checkpoint | val top-1 {val_final['top1']:.3f} top-5 {val_final['top5']:.3f}"
          f" | TEST top-1 {test_final['top1']:.3f} top-5 {test_final['top5']:.3f}")

    with open(os.path.join(args.out_dir, f"{args.model}_results.json"), "w") as f:
        json.dump({"args": vars(args), "device": device, "params": n_params,
                   "train_minutes": train_minutes, "val": val_final, "test": test_final,
                   "history": history}, f, indent=2)


if __name__ == "__main__":
    main()