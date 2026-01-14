"""Base class for YOLO-based models."""

import logging
import os
from abc import ABC
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .config import YOLOConfig

# Configure logger
logger = logging.getLogger(__name__)

# PyTorch 2.6+ compatibility note:
# PyTorch 2.6 changed the default value of weights_only from False to True in torch.load
# YOLO models require weights_only=False due to custom classes
# We handle this by monkey-patching torch.load during model loading (see load_model method)


class YOLOModelBase(ABC):
    """Base class for YOLO-based models (detection and segmentation).

    This class eliminates code duplication by providing common functionality
    for model loading, validation, and inference.
    """

    def __init__(
        self,
        model_path: str | None,
        env_var: str,
        default_path: str,
        config: YOLOConfig | None = None
    ):
        """Initialize the YOLO model.

        Args:
            model_path: Path to the YOLO .pt model file. If None, uses env_var or default_path
            env_var: Environment variable name for model path
            default_path: Default model path if env_var is not set
            config: YOLOConfig instance. If None, uses default configuration
        """
        if model_path is None:
            model_path = os.getenv(env_var, default_path)

        self.model_path = Path(model_path)
        self.model = None
        self.config = config if config is not None else YOLOConfig()
        self.load_model()

    @contextmanager
    def _patch_torch_load(self):
        """Context manager for temporarily patching torch.load.

        PyTorch 2.6+ requires weights_only=False for YOLO models.
        This context manager ensures thread-safe patching.
        """
        original_load = torch.load

        def patched_load(*args, **kwargs):
            # Force weights_only=False for YOLO model loading
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            yield
        finally:
            # Restore original torch.load
            torch.load = original_load

    def load_model(self) -> bool:
        """Load the YOLO model.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found at {self.model_path}")
                return False

            # Use context manager for safe torch.load patching
            with self._patch_torch_load():
                self.model = YOLO(str(self.model_path))
                logger.info(f"Model loaded successfully: {self.model_path.name}")
                if self.model is not None and hasattr(self.model, "names"):
                    logger.info(f"Model classes: {self.model.names}")

            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model is not None

    def _validate_model_loaded(self) -> None:
        """Validate that model is loaded, raise RuntimeError if not."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Please ensure the model file exists.")

    def _run_inference(self, image: Image.Image, conf_threshold: float):
        """Run YOLO inference on image.

        Args:
            image: PIL Image object
            conf_threshold: Confidence threshold for detection

        Returns:
            YOLO results object
        """
        return self.model(image, conf=conf_threshold, verbose=self.config.verbose)

    def _has_detections(self, results) -> bool:
        """Check if results contain any detections.

        Args:
            results: YOLO results object

        Returns:
            bool: True if detections found, False otherwise
        """
        return len(results) > 0 and len(results[0].boxes) > 0

    def _get_best_detection_index(self, boxes) -> int:
        """Find index of detection with highest confidence.

        Args:
            boxes: YOLO boxes object

        Returns:
            int: Index of best detection
        """
        confidences = boxes.conf.cpu().numpy()
        return int(np.argmax(confidences))
