"""Convolutional model for MiniPlaces scene classification."""

from torch import Tensor, nn


class MiniPlacesCNN(nn.Module):
    """A compact CNN that returns one logit per scene class.

    Inputs are expected in NCHW format with three RGB channels. Images may be
    any spatial size that remains at least 4x4 after the four pooling stages.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")

        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(kernel_size=2),
            self._block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            self._block(64, 128),
            nn.MaxPool2d(kernel_size=2),
            self._block(128, 256),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits for a batch of RGB images."""
        return self.classifier(self.features(images))
