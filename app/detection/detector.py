"""Salamander detector using YOLO."""

import logging

from PIL import Image

from .base import YOLOModelBase
from .config import YOLOConfig

# Configure logger
logger = logging.getLogger(__name__)


class SalamanderDetector(YOLOModelBase):
    """Salamander detector using YOLO."""

    def __init__(self, model_path: str | None = None, config: YOLOConfig | None = None):
        """Initialize the detector with a YOLO model.

        Args:
            model_path: Path to the YOLO .pt model file.
                       If None, uses default path from environment or models/crop.pt
            config: YOLOConfig instance for configuration. If None, uses default configuration
        """
        super().__init__(
            model_path=model_path,
            env_var="YOLO_MODEL_PATH",
            default_path="models/crop.pt",
            config=config
        )

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> tuple[bool, dict | None]:
        """Detect salamander in an image.

        Args:
            image: PIL Image object
            conf_threshold: Confidence threshold for detection

        Returns:
            Tuple of (detected: bool, detection_data: dict or None)
            detection_data contains: bbox coordinates, confidence, and cropped image
        """
        self._validate_model_loaded()

        logger.info(
            f"Running detection: size={image.size}, mode={image.mode}, conf={conf_threshold}"
        )

        # Run inference
        results = self._run_inference(image, conf_threshold)

        # Check if any detections
        if not self._has_detections(results):
            logger.info("No salamanders detected")
            return False, None

        # Get the detection with highest confidence
        boxes = results[0].boxes
        best_idx = self._get_best_detection_index(boxes)

        # Get bounding box coordinates (xyxy format)
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(boxes.conf[best_idx].cpu().numpy())

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
