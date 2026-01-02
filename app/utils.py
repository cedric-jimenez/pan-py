"""Utility functions for image processing."""

import base64
from io import BytesIO

from PIL import Image


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(img_data))
    return image
