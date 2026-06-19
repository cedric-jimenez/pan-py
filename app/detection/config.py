"""Configuration for YOLO models."""

from dataclasses import dataclass


@dataclass
class YOLOConfig:
    """Configuration for YOLO models.

    Centralizes all configurable parameters to avoid hard-coded values.
    """

    mask_threshold: float = 0.5  # Binary threshold for segmentation masks (50% confidence)
    verbose: bool = False  # Whether to show verbose YOLO output
