"""YOLO-based salamander detection and segmentation.

This package provides detection and segmentation capabilities for salamanders
using YOLO (You Only Look Once) models.

Public API:
    - SalamanderDetector: Detect and crop salamanders
    - SalamanderSegmenter: Segment salamanders with precise masks
    - YOLOConfig: Configuration for YOLO models
    - YOLOModelBase: Base class for custom YOLO models (advanced usage)
"""

from .base import YOLOModelBase
from .config import YOLOConfig
from .detector import SalamanderDetector
from .segmenter import SalamanderSegmenter

__all__ = [
    "YOLOConfig",
    "YOLOModelBase",
    "SalamanderDetector",
    "SalamanderSegmenter",
]
