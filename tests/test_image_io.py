import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.image_io import load_resized_rgb


class ImageLoadingTests(unittest.TestCase):
    def test_retries_after_transient_timeout(self) -> None:
        image = Image.new("RGB", (8, 8))
        with patch(
            "src.image_io.Image.open",
            side_effect=[TimeoutError("temporary read timeout"), image],
        ) as open_image:
            resized = load_resized_rgb(Path("sample.jpg"), 4)

        self.assertEqual(resized.size, (4, 4))
        self.assertEqual(open_image.call_count, 2)

    def test_reports_image_path_after_retries_are_exhausted(self) -> None:
        image_path = Path("unavailable.jpg")
        with patch(
            "src.image_io.Image.open",
            side_effect=TimeoutError("read timeout"),
        ):
            with self.assertRaisesRegex(TimeoutError, str(image_path)):
                load_resized_rgb(image_path, 4)


if __name__ == "__main__":
    unittest.main()
