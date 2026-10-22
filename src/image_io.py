"""Shared image loading helpers."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_READ_ATTEMPTS = 3


def load_resized_rgb(image_path: Path, image_size: int) -> Image.Image:
    """Load and resize an RGB image, retrying transient filesystem timeouts."""
    for attempt in range(IMAGE_READ_ATTEMPTS):
        try:
            with Image.open(image_path) as image_file:
                return image_file.convert("RGB").resize(
                    (image_size, image_size),
                    Image.Resampling.BILINEAR,
                )
        except TimeoutError as error:
            if attempt == IMAGE_READ_ATTEMPTS - 1:
                raise TimeoutError(
                    f"Timed out reading image {image_path} after "
                    f"{IMAGE_READ_ATTEMPTS} attempts. Check whether the dataset "
                    "is on an available local drive, then retry."
                ) from error
            time.sleep(0.25 * (attempt + 1))

    raise RuntimeError("Image read retry loop exited unexpectedly.")
