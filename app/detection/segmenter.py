"""Salamander segmenter using YOLO-seg."""

import logging

import cv2
import numpy as np
from PIL import Image

from .base import YOLOModelBase
from .config import YOLOConfig

# Configure logger
logger = logging.getLogger(__name__)


class SalamanderSegmenter(YOLOModelBase):
    """Salamander segmenter using YOLO-seg for precise mask-based cropping."""

    def __init__(self, model_path: str | None = None, config: YOLOConfig | None = None):
        """Initialize the segmenter with a YOLO segmentation model.

        Args:
            model_path: Path to the YOLO-seg .pt model file.
                       If None, uses default path from environment or models/segment.pt
            config: YOLOConfig instance for configuration. If None, uses default configuration
        """
        super().__init__(
            model_path=model_path,
            env_var="YOLO_SEGMENT_MODEL_PATH",
            default_path="models/segment.pt",
            config=config,
        )

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
        self._validate_model_loaded()

        logger.info(
            f"Running segmentation: size={image.size}, mode={image.mode}, conf={conf_threshold}"
        )

        # Run inference
        results = self._run_inference(image, conf_threshold)

        # Check if any segmentation masks found
        if len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            logger.info("No salamanders detected for segmentation")
            return False, None

        # Get best detection
        masks = results[0].masks
        boxes = results[0].boxes
        best_idx = self._get_best_detection_index(boxes)

        mask = masks.data[best_idx].cpu().numpy()
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(boxes.conf[best_idx].cpu().numpy())

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
        # Convert soft mask to binary using threshold from config
        mask_bool = mask_resized > self.config.mask_threshold

        # Create result with background color
        result = np.ones_like(img_array) * np.array(bg_color, dtype=np.uint8)
        result[mask_bool] = img_array[mask_bool]

        # Crop to bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cropped = Image.fromarray(result).crop((x1, y1, x2, y2))

        return cropped
