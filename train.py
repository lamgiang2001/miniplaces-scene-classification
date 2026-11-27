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
import csv
import importlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from data_pipeline import build_dataloaders, read_category_names

IMG_SIZE = 128
NUM_CLASSES = 100


def stratified_subset_indices(dataset, subset_size, seed):
    labels_by_class = {}
    for index, (_, label) in enumerate(dataset.pairs):
        labels_by_class.setdefault(label, []).append(index)

    subset_size = min(subset_size, len(dataset))
    class_labels = sorted(labels_by_class)
    per_class, remainder = divmod(subset_size, len(class_labels))
    rng = torch.Generator().manual_seed(seed)
    selected = []

    for class_position, label in enumerate(class_labels):
        class_indices = labels_by_class[label]
        count = per_class + int(class_position < remainder)
        order = torch.randperm(len(class_indices), generator=rng)[:count].tolist()
        selected.extend(class_indices[position] for position in order)

    return selected


def write_metrics_artifacts(history, out_dir, model_name):
    csv_path = os.path.join(out_dir, f"{model_name}_metrics.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_top1", "val_top5"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history)

    svg_path = os.path.join(out_dir, f"{model_name}_metrics.svg")
    width, height = 820, 720
    left, right = 80, 32
    chart_width = width - left - right
    epoch_count = len(history)
    x_position = lambda index: left + chart_width * index / max(epoch_count - 1, 1)
    panels = [
        {
            "top": 88,
            "height": 220,
            "title": "Training and validation loss",
            "ylabel": "Cross-entropy loss",
            "series": [
                ("train_loss", "#1769aa", "Train loss"),
                ("val_loss", "#d1495b", "Validation loss"),
            ],
        },
        {
            "top": 410,
            "height": 220,
            "title": "Validation accuracy",
            "ylabel": "Accuracy",
            "series": [
                ("val_top1", "#1769aa", "Top-1 accuracy"),
                ("val_top5", "#d1495b", "Top-5 accuracy"),
            ],
        },
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="410" y="34" text-anchor="middle" font-size="20">'
        f'{model_name.upper()} metrics by epoch</text>',
    ]

    for panel in panels:
        values = [row[key] for row in history for key, _, _ in panel["series"]]
        if panel["ylabel"] == "Accuracy":
            low, high = 0.0, 1.0
            tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            low = max(0.0, min(values) * 0.9)
            high = max(values) * 1.1
            if math.isclose(low, high):
                high = low + 1.0
            tick_values = [
                low + (high - low) * tick / 5 for tick in range(6)
            ]

        top, chart_height = panel["top"], panel["height"]
        parts.append(
            f'<text x="{left + chart_width/2}" y="{top-30}" '
            f'text-anchor="middle" font-size="16">{panel["title"]}</text>'
        )
        for tick_value in tick_values:
            fraction = (tick_value - low) / (high - low)
            y = top + chart_height * (1 - fraction)
            label = (
                f"{tick_value:.0%}"
                if panel["ylabel"] == "Accuracy"
                else f"{tick_value:.2f}"
            )
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
                    f'y2="{y:.1f}" stroke="#dddddd"/>',
                    f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
                    f'font-size="11">{label}</text>',
                ]
            )

        for index, row in enumerate(history):
            x = x_position(index)
            parts.append(
                f'<text x="{x:.1f}" y="{top+chart_height+22}" '
                f'text-anchor="middle" font-size="12">{row["epoch"]}</text>'
            )

        for series_index, (key, color, label) in enumerate(panel["series"]):
            coordinates = []
            for index, row in enumerate(history):
                fraction = (row[key] - low) / (high - low)
                y = top + chart_height * (1 - fraction)
                coordinates.append((x_position(index), y))
            point_list = " ".join(
                f"{x:.1f},{y:.1f}" for x, y in coordinates
            )
            parts.append(
                f'<polyline points="{point_list}" fill="none" '
                f'stroke="{color}" stroke-width="3"/>'
            )
            for x, y in coordinates:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
                )
            legend_x = left + series_index * 190
            legend_y = top - 8
            parts.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y}" '
                    f'x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" '
                    'stroke-width="3"/>',
                    f'<text x="{legend_x+30}" y="{legend_y+4}" '
                    f'font-size="12">{label}</text>',
                ]
            )

        parts.extend(
            [
                f'<text x="20" y="{top+chart_height/2}" text-anchor="middle" '
                f'transform="rotate(-90 20 {top+chart_height/2})" '
                f'font-size="12">{panel["ylabel"]}</text>',
                f'<text x="{left+chart_width/2}" '
                f'y="{top+chart_height+45}" text-anchor="middle" '
                'font-size="12">Epoch</text>',
            ]
        )

    parts.append("</svg>")
    with open(svg_path, "w", encoding="utf-8") as svg_file:
        svg_file.write("\n".join(parts))

    print(f"Saved epoch metrics table to {csv_path}")
    print(f"Saved loss and accuracy plots to {svg_path}")


def write_confusion_matrix_artifacts(
    confusion_counts,
    class_names,
    out_dir,
    model_name,
):
    csv_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["true_class_id", "true_class", *[
            f"pred_{class_id:02d}" for class_id in range(len(class_names))
        ]])
        for class_id, (class_name, row) in enumerate(
            zip(class_names, confusion_counts, strict=True)
        ):
            writer.writerow([class_id, class_name, *row])

    svg_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.svg")
    class_count = len(class_names)
    cell_size = 9
    left, top = 90, 90
    width = left + cell_size * class_count + 32
    height = top + cell_size * class_count + 92
    row_totals = [sum(row) for row in confusion_counts]
    max_rate = max(
        (
            count / row_totals[true_id]
            for true_id, row in enumerate(confusion_counts)
            if row_totals[true_id]
            for count in row
        ),
        default=1.0,
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">'
        f'{model_name.upper()} confusion matrix</text>',
        f'<text x="{left+cell_size*class_count/2}" y="52" '
        'text-anchor="middle" font-size="12">Predicted class ID</text>',
        f'<text x="20" y="{top+cell_size*class_count/2}" '
        'text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 20 {top+cell_size*class_count/2})">'
        'True class ID</text>',
    ]
    for class_id in range(class_count):
        tick_x = left + class_id * cell_size
        tick_y = top + class_id * cell_size
        parts.extend(
            [
                f'<text x="{tick_x+cell_size/2:.1f}" y="{top-7}" '
                'text-anchor="middle" font-size="7">'
                f'{class_id}</text>',
                f'<text x="{left-7}" y="{tick_y+cell_size*0.8:.1f}" '
                'text-anchor="end" font-size="7">'
                f'{class_id}</text>',
            ]
        )
        total = row_totals[class_id]
        for predicted_id, count in enumerate(confusion_counts[class_id]):
            rate = count / total if total else 0.0
            intensity = rate / max_rate if max_rate else 0.0
            red = round(255 * (1 - intensity))
            green = round(255 * (1 - intensity * 0.55))
            blue = 255
            color = f"#{red:02x}{green:02x}{blue:02x}"
            parts.append(
                f'<rect x="{left+predicted_id*cell_size}" '
                f'y="{top+class_id*cell_size}" width="{cell_size}" '
                f'height="{cell_size}" fill="{color}"/>'
            )

    parts.extend(
        [
            f'<text x="{left}" y="{height-20}" font-size="10">'
            'Each row is normalized by its true-class count; class IDs map '
            'to names in the confusion-matrix CSV.</text>',
            "</svg>",
        ]
    )
    with open(svg_path, "w", encoding="utf-8") as svg_file:
        svg_file.write("\n".join(parts))

    print(f"Saved confusion matrix table to {csv_path}")
    print(f"Saved normalized confusion-matrix heatmap to {svg_path}")


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


@torch.no_grad()
def write_prediction_csv(
    model,
    loader,
    device,
    image_root,
    class_names,
    output_path,
):
    from torch.utils.data import Subset

    dataset = loader.dataset
    selected_indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        selected_indices = [dataset.indices[index] for index in selected_indices]
        dataset = dataset.dataset

    rows = []
    confusion_counts = [
        [0 for _ in class_names] for _ in class_names
    ]
    model.eval()
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = logits.softmax(dim=1)
        confidences, predictions = probabilities.max(dim=1)
        rows.extend(
            zip(
                labels.tolist(),
                predictions.cpu().tolist(),
                confidences.cpu().tolist(),
                strict=True,
            )
        )

    if len(rows) != len(selected_indices):
        raise RuntimeError("Prediction count does not match validation samples.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(
            [
                "image_path",
                "true_class_id",
                "true_class",
                "predicted_class_id",
                "predicted_class",
                "correct",
                "top1_confidence_percent",
            ]
        )
        for dataset_index, (true_label, predicted_label, confidence) in zip(
            selected_indices,
            rows,
            strict=True,
        ):
            relative_path, dataset_label = dataset.pairs[dataset_index]
            if true_label != dataset_label:
                raise RuntimeError(
                    f"Validation label mismatch for {relative_path}."
                )
            writer.writerow(
                [
                    os.path.join(image_root, relative_path),
                    true_label,
                    class_names[true_label],
                    predicted_label,
                    class_names[predicted_label],
                    true_label == predicted_label,
                    round(confidence * 100, 2),
                ]
            )
            confusion_counts[true_label][predicted_label] += 1

    write_confusion_matrix_artifacts(
        confusion_counts,
        class_names,
        os.path.dirname(output_path) or ".",
        Path(output_path).name.removesuffix("_validation_predictions.csv"),
    )


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
                        "using a class-balanced subset.")
    p.add_argument("--val_subset", type=int, default=0,
                   help="If > 0, evaluate on a class-balanced subset of the "
                        "training-derived validation split.")
    p.add_argument("--skip_test", action="store_true",
                   help="Do not evaluate the official validation split as a test set.")
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
    if args.val_subset < 0:
        p.error("--val_subset must be non-negative")

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
        idx = stratified_subset_indices(
            train_loader.dataset, args.train_subset, args.seed
        )
        train_loader = DataLoader(Subset(train_loader.dataset, idx), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        print(f"SMOKE TEST: training on {len(idx)} images only")

    if args.val_subset > 0:
        from torch.utils.data import DataLoader, Subset
        idx = stratified_subset_indices(
            val_loader.dataset, args.val_subset, args.seed + 1
        )
        val_loader = DataLoader(Subset(val_loader.dataset, idx), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        print(f"Validation metrics use {len(idx)} held-out training images")

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    history, best_val_top1, best_val_metrics = [], -1.0, None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        progress_interval = max(1, len(train_loader) // 5)
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
            if batch_index % progress_interval == 0 or batch_index == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs} "
                    f"batch {batch_index}/{len(train_loader)}"
                )

        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / seen,
               "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"]}
        history.append(row)
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.3f} | "
              f"val loss {val['loss']:.3f} | val top-1 {val['top1']:.3f} | val top-5 {val['top5']:.3f}")

        if val["top1"] > best_val_top1:
            best_val_top1 = val["top1"]
            best_val_metrics = val
            torch.save(model.state_dict(), ckpt_path)

    train_minutes = (time.time() - start) / 60

    # Final test evaluation: once, with the checkpoint chosen on validation only.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    val_final = best_val_metrics
    image_root = args.image_root
    if image_root is None:
        images_dir = os.path.join(args.data_root, "images")
        image_root = images_dir if os.path.isdir(images_dir) else args.data_root
    class_names = read_category_names(
        os.path.join(args.data_root, "data", "categories.txt")
    )
    predictions_path = os.path.join(
        args.out_dir, f"{args.model}_validation_predictions.csv"
    )
    write_prediction_csv(
        model,
        val_loader,
        device,
        image_root,
        class_names,
        predictions_path,
    )
    print(f"Saved per-image validation predictions to {predictions_path}")

    test_final = None if args.skip_test else evaluate(model, test_loader, device)
    test_summary = (
        " | Official-val test evaluation skipped"
        if test_final is None
        else f" | TEST top-1 {test_final['top1']:.3f} top-5 {test_final['top5']:.3f}"
    )
    print(f"\nBest-val checkpoint | val top-1 {val_final['top1']:.3f} top-5 {val_final['top5']:.3f}"
          f"{test_summary}")

    write_metrics_artifacts(history, args.out_dir, args.model)

    with open(os.path.join(args.out_dir, f"{args.model}_results.json"), "w") as f:
        json.dump({"args": vars(args), "device": device, "params": n_params,
                   "train_minutes": train_minutes, "val": val_final, "test": test_final,
                   "history": history}, f, indent=2)


if __name__ == "__main__":
    main()