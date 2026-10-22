"""Train and validate the MiniPlaces CNN using labeled training images."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.models import MiniPlacesCNN
from src.image_io import IMAGE_EXTENSIONS, load_resized_rgb


class SceneImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int,
        *,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self.samples[index]
        image = load_resized_rgb(image_path, self.image_size)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        return image_tensor, label


def collect_labeled_images(
    train_dir: Path,
    *,
    seed: int,
    validation_fraction: float,
    limit_per_class: int | None,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training image directory does not exist: {train_dir}")

    class_directories = sorted(
        directory
        for directory in train_dir.glob("*/*")
        if directory.is_dir()
    )
    if not class_directories:
        raise ValueError(
            f"No class directories found under {train_dir}; expected "
            "<train>/<letter>/<class>/<image>."
        )

    class_names = [directory.name for directory in class_directories]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class directory names must be unique.")

    random_generator = random.Random(seed)
    training_samples: list[tuple[Path, int]] = []
    validation_samples: list[tuple[Path, int]] = []

    for label, class_directory in enumerate(class_directories):
        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        random_generator.shuffle(image_paths)

        if limit_per_class is not None:
            image_paths = image_paths[:limit_per_class]
        if len(image_paths) < 2:
            raise ValueError(
                f"Class {class_directory.name!r} needs at least two images "
                "to create separate training and validation samples."
            )

        validation_count = max(1, round(len(image_paths) * validation_fraction))
        validation_count = min(validation_count, len(image_paths) - 1)
        validation_samples.extend(
            (path, label) for path in image_paths[:validation_count]
        )
        training_samples.extend(
            (path, label) for path in image_paths[validation_count:]
        )

    return training_samples, validation_samples, class_names


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: MiniPlacesCNN,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_function(logits, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct_predictions / sample_count


def write_validation_predictions(
    model: MiniPlacesCNN,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    class_names: list[str],
    device: torch.device,
    output_path: Path,
) -> float:
    model.eval()
    samples = data_loader.dataset.samples
    correct_predictions = 0
    sample_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_path",
                "true_class",
                "predicted_class",
                "correct",
                "top1_confidence_percent",
            ]
        )

        with torch.inference_mode():
            sample_offset = 0
            for images, labels in data_loader:
                probabilities = model(images.to(device)).softmax(dim=1)
                confidences, predictions = probabilities.max(dim=1)
                predictions = predictions.cpu().tolist()
                confidences = confidences.cpu().tolist()
                labels = labels.tolist()

                for label, prediction, confidence in zip(
                    labels,
                    predictions,
                    confidences,
                    strict=True,
                ):
                    image_path, _ = samples[sample_offset]
                    is_correct = prediction == label
                    correct_predictions += int(is_correct)
                    sample_count += 1
                    writer.writerow(
                        [
                            image_path,
                            class_names[label],
                            class_names[prediction],
                            is_correct,
                            round(confidence * 100, 2),
                        ]
                    )
                    sample_offset += 1

    return correct_predictions / sample_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MiniPlaces CNN."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/images/train"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/cnn.pt"))
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("results/validation_predictions.csv"),
        help="CSV path for true labels and predictions on the held-out split.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Use at most this many images per class for a quick pipeline check.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        parser.error("--epochs, --batch-size, and --image-size must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1.")
    if args.limit_per_class is not None and args.limit_per_class < 2:
        parser.error("--limit-per-class must be at least 2.")

    return args


def main() -> None:
    args = parse_args()
    training_samples, validation_samples, class_names = collect_labeled_images(
        args.train_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        limit_per_class=args.limit_per_class,
    )

    training_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        SceneImageDataset(
            training_samples,
            args.image_size,
            augment=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        SceneImageDataset(validation_samples, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = select_device()
    model = MiniPlacesCNN(num_classes=len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(
        f"Device: {device}; classes: {len(class_names)}; "
        f"training images: {len(training_samples)}; "
        f"validation images: {len(validation_samples)}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            training_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_top1={train_accuracy:.2%} "
            f"val_loss={validation_loss:.4f} "
            f"val_top1={validation_accuracy:.2%}"
        )

    csv_accuracy = write_validation_predictions(
        model,
        validation_loader,
        class_names,
        device,
        args.validation_output,
    )
    print(
        f"Validation CSV accuracy: {csv_accuracy:.2%}; "
        f"saved per-image results to {args.validation_output}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "validation_accuracy": validation_accuracy,
            "epochs": args.epochs,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
