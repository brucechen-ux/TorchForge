"""Dense feed-forward (MLP) building blocks used by transformer FFN layers."""

from .feedforward import FeedForward
from .gated_mlp import GatedMLP

__all__ = ["FeedForward", "GatedMLP"]
