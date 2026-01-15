"""Configuration for YOLO models."""

from dataclasses import dataclass


@dataclass
class YOLOConfig:
    """Configuration for YOLO models.

    Centralizes all configurable parameters to avoid hard-coded values.
    """

    confidence_threshold: float = 0.25  # Default confidence threshold for detection
    mask_threshold: float = 0.5  # Binary threshold for segmentation masks (50% confidence)
    bg_color: tuple[int, int, int] = (150, 150, 150)  # Background color for segmented images (gray)
    verbose: bool = False  # Whether to show verbose YOLO output
