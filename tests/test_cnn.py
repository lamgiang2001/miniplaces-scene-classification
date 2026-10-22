import unittest

import torch

from src.models import MiniPlacesCNN


class MiniPlacesCNNTests(unittest.TestCase):
    def test_returns_logits_for_each_class(self) -> None:
        model = MiniPlacesCNN(num_classes=100)
        images = torch.randn(2, 3, 128, 128)

        logits = model(images)

        self.assertEqual(tuple(logits.shape), (2, 100))

    def test_parameter_count_is_in_compact_cnn_range(self) -> None:
        model = MiniPlacesCNN()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertGreaterEqual(parameter_count, 400_000)
        self.assertLessEqual(parameter_count, 1_200_000)

    def test_rejects_nonpositive_class_count(self) -> None:
        with self.assertRaises(ValueError):
            MiniPlacesCNN(num_classes=0)


if __name__ == "__main__":
    unittest.main()
