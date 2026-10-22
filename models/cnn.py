"""Expose the project CNN through the shared trainer's model interface."""

from src.models import MiniPlacesCNN


def build_model() -> MiniPlacesCNN:
    return MiniPlacesCNN(num_classes=100)
