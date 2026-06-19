"""Utility functions for image processing."""

import base64
from io import BytesIO

from PIL import Image


def pil_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        quality: JPEG quality (1-95, only used for JPEG format)

    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    if format.upper() == "JPEG":
        # Convert RGBA to RGB if needed for JPEG
        if image.mode in ("RGBA", "LA", "P"):
            # Create white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(
                image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None
            )
            image = background
        image.save(buffered, format=format, quality=quality, optimize=False)
    else:
        image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


