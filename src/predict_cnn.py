"""Run a trained MiniPlaces CNN over images and export predictions to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.image_io import IMAGE_EXTENSIONS, load_resized_rgb
from src.train_cnn import select_device
from src.models import MiniPlacesCNN


class ImageFolderDataset(Dataset[tuple[Tensor, str]]):
    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        image_path = self.image_paths[index]
        image = load_resized_rgb(image_path, self.image_size)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        return image_tensor, str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict scene classes for each image in a folder."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cnn.pt"))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Predict only the first N images, useful for a quick check.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    image_paths = sorted(
        path
        for path in args.image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"No supported images found under {args.image_dir}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    class_names: list[str] = checkpoint["class_names"]
    image_size: int = checkpoint["image_size"]
    model = MiniPlacesCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = select_device()
    model.to(device)
    model.eval()
    loader: DataLoader[tuple[Tensor, list[str]]] = DataLoader(
        ImageFolderDataset(image_paths, image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_path",
                "predicted_class",
                "top1_confidence_percent",
                "top2_class",
                "top2_confidence_percent",
                "top3_class",
                "top3_confidence_percent",
            ]
        )

        with torch.inference_mode():
            for images, paths in loader:
                probabilities = model(images.to(device)).softmax(dim=1)
                top_probabilities, top_indices = probabilities.topk(
                    min(3, len(class_names)),
                    dim=1,
                )
                for path, row_probabilities, row_indices in zip(
                    paths,
                    top_probabilities.cpu().tolist(),
                    top_indices.cpu().tolist(),
                    strict=True,
                ):
                    predictions: list[str | float] = []
                    for index, probability in zip(
                        row_indices,
                        row_probabilities,
                        strict=True,
                    ):
                        predictions.extend(
                            [class_names[index], round(probability * 100, 2)]
                        )
                    predictions.extend(["", ""] * (3 - len(row_indices)))
                    writer.writerow([path, *predictions])

    print(f"Predicted {len(image_paths)} images using {device}.")
    print(f"Saved predictions to {args.output}")
    print(
        "Confidence is the model's softmax score, not the probability that "
        "the prediction is correct."
    )


if __name__ == "__main__":
    main()
