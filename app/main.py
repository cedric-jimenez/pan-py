"""FastAPI application for salamander detection and cropping."""

import io
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pythonjsonlogger import jsonlogger

from app import __version__
from app.detection import SalamanderDetector, SalamanderSegmenter
from app.models import BoundingBox, DetectionResponse, HealthResponse, SegmentationResponse
from app.utils import pil_to_base64

# Configure JSON logging for Railway
log_handler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_handler)

logger = logging.getLogger(__name__)

# Global detector and segmenter instances
detector: SalamanderDetector | None = None
segmenter: SalamanderSegmenter | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and cleanup resources."""
    global detector, segmenter
    # Startup: Load the YOLO models
    logger.info("Loading YOLO detection model...")
    detector = SalamanderDetector()
    logger.info("Loading YOLO segmentation model...")
    segmenter = SalamanderSegmenter()
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
        segment_loaded=segmenter is not None and segmenter.is_model_loaded(),
        version=__version__,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        yolo_loaded=detector is not None and detector.is_model_loaded(),
        segment_loaded=segmenter is not None and segmenter.is_model_loaded(),
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


# Background color mapping
BACKGROUND_COLORS = {
    "gray": (150, 150, 150),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


@app.post("/segment-salamander", response_model=SegmentationResponse)
async def segment_salamander(
    file: UploadFile = File(..., description="Image file containing a salamander"),
    confidence: float = Query(
        0.25, ge=0.0, le=1.0, description="Confidence threshold for detection"
    ),
    background: str = Query(
        "gray", description="Background color: gray, white, or black"
    ),
    image_format: str = Query(
        "JPEG", description="Output image format (JPEG or PNG). JPEG is much faster."
    ),
    image_quality: int = Query(
        85, ge=1, le=95, description="JPEG quality (1-95, only used for JPEG format)"
    ),
):
    """
    Segment a salamander from an uploaded image using YOLO instance segmentation.

    This endpoint provides precise mask-based cropping that removes the background,
    unlike /crop-salamander which returns a rectangular crop with background included.

    Args:
        file: Image file (JPEG, PNG, etc.)
        confidence: Confidence threshold for detection (0.0 to 1.0)
        background: Background color for the segmented image (gray, white, black)
        image_format: Output format (JPEG or PNG). JPEG is faster.
        image_quality: JPEG quality (1-95). Lower = faster but lower quality.

    Returns:
        SegmentationResponse with segmentation results and the masked image
    """
    # Validate segmenter is loaded
    if segmenter is None or not segmenter.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded. Please ensure segment.pt is available.",
        )

    # Validate background color
    if background not in BACKGROUND_COLORS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid background color: {background}. Use: gray, white, or black.",
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
            f"Processing image for segmentation: {file.filename} ({len(contents)} bytes) - read: {read_time:.3f}s"
        )

        decode_start = time.time()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        decode_time = time.time() - decode_start
        logger.info(f"Image decode and convert: {decode_time:.3f}s")

        original_width, original_height = image.size

        # Run segmentation
        segment_start = time.time()
        bg_color = BACKGROUND_COLORS[background]
        detected, segmentation_data = segmenter.segment(
            image, conf_threshold=confidence, bg_color=bg_color
        )
        segment_time = time.time() - segment_start
        logger.info(f"YOLO segmentation: {segment_time:.3f}s")

        if not detected or segmentation_data is None:
            total_time = time.time() - start_time
            logger.info(f"Total time (no detection): {total_time:.3f}s")
            return SegmentationResponse(
                success=True,
                message="No salamander detected in the image",
                detected=False,
                bounding_box=None,
                segmented_image=None,
                original_width=original_width,
                original_height=original_height,
                background_color=background,
            )

        # Prepare bbox response
        bbox = segmentation_data["bbox"]
        bbox_data = BoundingBox(
            x1=bbox["x1"],
            y1=bbox["y1"],
            x2=bbox["x2"],
            y2=bbox["y2"],
            confidence=segmentation_data["confidence"],
        )

        # Encode segmented image to base64
        encode_start = time.time()
        segmented_base64 = pil_to_base64(
            segmentation_data["segmented_image"], format=image_format, quality=image_quality
        )
        encode_time = time.time() - encode_start
        logger.info(
            f"Image encoding ({image_format}, quality={image_quality}): {encode_time:.3f}s"
        )

        total_time = time.time() - start_time
        logger.info(f"Total segmentation time: {total_time:.3f}s")

        return SegmentationResponse(
            success=True,
            message="Salamander segmented successfully",
            detected=True,
            bounding_box=bbox_data,
            segmented_image=segmented_base64,
            original_width=original_width,
            original_height=original_height,
            background_color=background,
        )

    except Exception as e:
        logger.error(f"Error processing image for segmentation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}") from e


@app.get("/model-info")
async def model_info():
    """Get information about the loaded models."""
    return {
        "detection": {
            "loaded": detector is not None and detector.is_model_loaded(),
            "path": str(detector.model_path) if detector else None,
            "exists": detector.model_path.exists() if detector else False,
        },
        "segmentation": {
            "loaded": segmenter is not None and segmenter.is_model_loaded(),
            "path": str(segmenter.model_path) if segmenter else None,
            "exists": segmenter.model_path.exists() if segmenter else False,
        },
    }
