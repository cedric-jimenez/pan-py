"""YOLO-based salamander detection logic."""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Configure logger
logger = logging.getLogger(__name__)

# PyTorch 2.6+ compatibility note:
# PyTorch 2.6 changed the default value of weights_only from False to True in torch.load
# YOLO models require weights_only=False due to custom classes
# We handle this by monkey-patching torch.load during model loading (see load_model method)


class SalamanderDetector:
    """Salamander detector using YOLO."""

    def __init__(self, model_path: str | None = None):
        """Initialize the detector with a YOLO model.

        Args:
            model_path: Path to the YOLO .pt model file.
                       If None, uses default path from environment or models/crop.pt
        """
        if model_path is None:
            model_path = os.getenv("YOLO_MODEL_PATH", "models/crop.pt")

        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self) -> bool:
        """Load the YOLO model.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found at {self.model_path}")
                return False

            # For PyTorch 2.6+, we need to use weights_only=False for YOLO models
            # Monkey-patch torch.load temporarily to force weights_only=False
            original_load = torch.load

            def patched_load(*args, **kwargs):
                # Force weights_only=False for YOLO model loading
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = patched_load
            try:
                self.model = YOLO(str(self.model_path))
                logger.info(f"Model loaded successfully: {self.model_path.name}")
                if self.model is not None and hasattr(self.model, "names"):
                    logger.info(f"Model classes: {self.model.names}")
            finally:
                # Restore original torch.load
                torch.load = original_load

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

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple[bool, dict | None]:
        """Detect salamander in an image.

        Args:
            image: PIL Image object
            conf_threshold: Confidence threshold for detection

        Returns:
            Tuple of (detected: bool, detection_data: dict or None)
            detection_data contains: bbox coordinates, confidence, and cropped image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please ensure the model file exists.")

        logger.info(
            f"Running detection: size={image.size}, mode={image.mode}, conf={conf_threshold}"
        )

        # Run inference with PIL image directly
        results = self.model(image, conf=conf_threshold, verbose=False)

        # Check if any detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            logger.info("No salamanders detected")
            return False, None

        # Get the detection with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        # Get bounding box coordinates (xyxy format)
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(confidences[best_idx])

        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1

        logger.info(
            f"Salamander detected: bbox=({x1},{y1},{x2},{y2}), "
            f"size={width}x{height}px, confidence={confidence:.2%}"
        )

        # Crop the image
        cropped = image.crop((x1, y1, x2, y2))

        detection_data = {
            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "confidence": confidence,
            "cropped_image": cropped,
        }

        return True, detection_data

class SalamanderSegmenter:
    """Salamander segmenter using YOLO-seg for precise mask-based cropping."""

    def __init__(self, model_path: str | None = None):
        """Initialize the segmenter with a YOLO segmentation model.

        Args:
            model_path: Path to the YOLO-seg .pt model file.
                       If None, uses default path from environment or models/segment.pt
        """
        if model_path is None:
            model_path = os.getenv("YOLO_SEGMENT_MODEL_PATH", "models/segment.pt")

        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self) -> bool:
        """Load the YOLO segmentation model.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Segmentation model file not found at {self.model_path}")
                return False

            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = patched_load
            try:
                self.model = YOLO(str(self.model_path))
                logger.info(f"Segmentation model loaded successfully: {self.model_path.name}")
                if self.model is not None and hasattr(self.model, "names"):
                    logger.info(f"Model classes: {self.model.names}")
            finally:
                torch.load = original_load

            return True
        except Exception as e:
            logger.error(f"Error loading segmentation model: {e}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def segment(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        bg_color: tuple = (150, 150, 150),
    ) -> tuple[bool, dict | None]:
        """Segment salamander and return masked image with background removed.

        Args:
            image: PIL Image object
            conf_threshold: Confidence threshold for detection
            bg_color: RGB tuple for background color (default: gray 150)

        Returns:
            Tuple of (detected: bool, segmentation_data: dict or None)
        """
        if self.model is None:
            raise RuntimeError("Segmentation model not loaded.")

        logger.info(
            f"Running segmentation: size={image.size}, mode={image.mode}, conf={conf_threshold}"
        )

        results = self.model(image, conf=conf_threshold, verbose=False)

        if len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            logger.info("No salamanders detected for segmentation")
            return False, None

        # Get best detection
        masks = results[0].masks
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        mask = masks.data[best_idx].cpu().numpy()
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(confidences[best_idx])

        x1, y1, x2, y2 = map(int, bbox)
        logger.info(
            f"Salamander segmented: bbox=({x1},{y1},{x2},{y2}), confidence={confidence:.2%}"
        )

        # Apply mask and crop
        segmented_image = self._apply_mask(image, mask, bbox, bg_color)

        return True, {
            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "confidence": confidence,
            "segmented_image": segmented_image,
        }

    def _apply_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        bbox: np.ndarray,
        bg_color: tuple,
    ) -> Image.Image:
        """Apply segmentation mask to image with custom background color.

        Args:
            image: Original PIL Image
            mask: Binary mask from YOLO segmentation
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            bg_color: RGB tuple for background

        Returns:
            Cropped PIL Image with mask applied
        """
        img_array = np.array(image)

        # Resize mask to image dimensions
        mask_resized = cv2.resize(mask, (image.width, image.height))
        mask_bool = mask_resized > 0.5

        # Create result with background color
        result = np.ones_like(img_array) * np.array(bg_color, dtype=np.uint8)
        result[mask_bool] = img_array[mask_bool]

        # Crop to bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cropped = Image.fromarray(result).crop((x1, y1, x2, y2))

        return cropped
