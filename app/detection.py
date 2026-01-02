"""YOLO-based salamander detection logic."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Add safe globals for PyTorch 2.6+ to allow YOLO model loading
try:
    from ultralytics.nn.tasks import DetectionModel

    torch.serialization.add_safe_globals([DetectionModel])
except (ImportError, AttributeError):
    # Fallback for older versions or if import fails
    pass


class SalamanderDetector:
    """Salamander detector using YOLO."""

    def __init__(self, model_path: str | None = None):
        """Initialize the detector with a YOLO model.

        Args:
            model_path: Path to the YOLO .pt model file.
                       If None, uses default path from environment or models/best.pt
        """
        if model_path is None:
            model_path = os.getenv("YOLO_MODEL_PATH", "models/best.pt")

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
                print(f"Warning: Model file not found at {self.model_path}")
                return False

            self.model = YOLO(str(self.model_path))
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
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

        # Convert PIL Image to numpy array for YOLO
        img_array = np.array(image)

        # Run inference
        results = self.model(img_array, conf=conf_threshold, verbose=False)

        # Check if any detections
        if len(results) == 0 or len(results[0].boxes) == 0:
            return False, None

        # Get the detection with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        # Get bounding box coordinates (xyxy format)
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(confidences[best_idx])

        x1, y1, x2, y2 = map(int, bbox)

        # Crop the image
        cropped = image.crop((x1, y1, x2, y2))

        detection_data = {
            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            "confidence": confidence,
            "cropped_image": cropped,
        }

        return True, detection_data
