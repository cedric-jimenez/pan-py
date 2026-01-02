"""FastAPI application for salamander detection and cropping."""

import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app import __version__
from app.detection import SalamanderDetector
from app.models import BoundingBox, DetectionResponse, HealthResponse
from app.utils import pil_to_base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global detector instance
detector: SalamanderDetector | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and cleanup resources."""
    global detector
    # Startup: Load the YOLO model
    logger.info("Loading YOLO model...")
    detector = SalamanderDetector()
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Salamander Detection API",
    description="API for detecting and cropping salamanders in images using YOLO",
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information."""
    return HealthResponse(
        status="healthy",
        yolo_loaded=detector is not None and detector.is_model_loaded(),
        version=__version__,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        yolo_loaded=detector is not None and detector.is_model_loaded(),
        version=__version__,
    )


@app.post("/crop-salamander", response_model=DetectionResponse)
async def crop_salamander(
    file: UploadFile = File(..., description="Image file containing a salamander"),
    confidence: float = Query(
        0.25, ge=0.0, le=1.0, description="Confidence threshold for detection"
    ),
    return_base64: bool = Query(True, description="Whether to return the cropped image as base64"),
    image_format: str = Query(
        "JPEG", description="Output image format (JPEG or PNG). JPEG is much faster."
    ),
    image_quality: int = Query(
        85, ge=1, le=95, description="JPEG quality (1-95, only used for JPEG format)"
    ),
    max_size: int = Query(
        1280,
        ge=320,
        le=4096,
        description="Max dimension for detection (image resized if larger). Smaller = faster.",
    ),
):
    """
    Detect and crop a salamander from an uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)
        confidence: Confidence threshold for detection (0.0 to 1.0)
        return_base64: Whether to return the cropped image as base64 encoded string
        image_format: Output format (JPEG or PNG). JPEG is 10-20x faster than PNG.
        image_quality: JPEG quality (1-95). Lower = faster but lower quality.
        max_size: Max dimension for detection. Images larger than this will be resized
                  for faster detection. The crop is done on the original full-size image.

    Returns:
        DetectionResponse with detection results and optionally the cropped image
    """
    # Validate detector is loaded
    if detector is None or not detector.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the YOLO model file is available.",
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file.",
        )

    try:
        import time

        start_time = time.time()

        # Read and open image
        contents = await file.read()
        read_time = time.time() - start_time
        logger.info(
            f"Processing image: {file.filename} ({len(contents)} bytes) - read: {read_time:.3f}s"
        )

        decode_start = time.time()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        decode_time = time.time() - decode_start
        logger.info(f"Image decode and convert: {decode_time:.3f}s")

        original_width, original_height = image.size
        original_image = image

        # Resize image for faster detection if needed
        scale_factor = 1.0
        if max(original_width, original_height) > max_size:
            resize_start = time.time()
            scale_factor = max_size / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resize_time = time.time() - resize_start
            logger.info(
                f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} "
                f"(scale={scale_factor:.3f}) in {resize_time:.3f}s"
            )

        # Run detection
        detect_start = time.time()
        detected, detection_data = detector.detect(image, conf_threshold=confidence)
        detect_time = time.time() - detect_start
        logger.info(f"YOLO detection: {detect_time:.3f}s")

        if not detected or detection_data is None:
            total_time = time.time() - start_time
            logger.info(f"Total time (no detection): {total_time:.3f}s")
            return DetectionResponse(
                success=True,
                message="No salamander detected in the image",
                detected=False,
                bounding_box=None,
                cropped_image=None,
                original_width=original_width,
                original_height=original_height,
            )

        # Rescale bbox coordinates to original image size
        bbox = detection_data["bbox"]
        if scale_factor != 1.0:
            x1_orig = bbox["x1"] / scale_factor
            y1_orig = bbox["y1"] / scale_factor
            x2_orig = bbox["x2"] / scale_factor
            y2_orig = bbox["y2"] / scale_factor
            logger.info(
                f"Rescaling bbox from ({bbox['x1']:.1f},{bbox['y1']:.1f},{bbox['x2']:.1f},{bbox['y2']:.1f}) "
                f"to ({x1_orig:.1f},{y1_orig:.1f},{x2_orig:.1f},{y2_orig:.1f})"
            )
        else:
            x1_orig = bbox["x1"]
            y1_orig = bbox["y1"]
            x2_orig = bbox["x2"]
            y2_orig = bbox["y2"]

        # Crop from original full-size image
        crop_start = time.time()
        cropped_image = original_image.crop((x1_orig, y1_orig, x2_orig, y2_orig))
        crop_time = time.time() - crop_start
        logger.info(f"Image crop: {crop_time:.3f}s")

        # Prepare response
        bbox_data = BoundingBox(
            x1=x1_orig,
            y1=y1_orig,
            x2=x2_orig,
            y2=y2_orig,
            confidence=detection_data["confidence"],
        )

        cropped_base64 = None
        if return_base64:
            encode_start = time.time()
            cropped_base64 = pil_to_base64(
                cropped_image, format=image_format, quality=image_quality
            )
            encode_time = time.time() - encode_start
            logger.info(
                f"Image encoding ({image_format}, quality={image_quality}): {encode_time:.3f}s"
            )

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.3f}s")

        return DetectionResponse(
            success=True,
            message="Salamander detected and cropped successfully",
            detected=True,
            bounding_box=bbox_data,
            cropped_image=cropped_base64,
            original_width=original_width,
            original_height=original_height,
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}") from e


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if detector is None:
        return JSONResponse(status_code=503, content={"error": "Detector not initialized"})

    return {
        "model_loaded": detector.is_model_loaded(),
        "model_path": str(detector.model_path),
        "model_exists": detector.model_path.exists(),
    }
