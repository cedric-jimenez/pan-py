"""FastAPI application for salamander detection and cropping."""

import io
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

# Global detector instance
detector: SalamanderDetector | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and cleanup resources."""
    global detector
    # Startup: Load the YOLO model
    print("Loading YOLO model...")
    detector = SalamanderDetector()
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


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
):
    """
    Detect and crop a salamander from an uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)
        confidence: Confidence threshold for detection (0.0 to 1.0)
        return_base64: Whether to return the cropped image as base64 encoded string

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
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_width, original_height = image.size

        # Run detection
        detected, detection_data = detector.detect(image, conf_threshold=confidence)

        if not detected or detection_data is None:
            return DetectionResponse(
                success=True,
                message="No salamander detected in the image",
                detected=False,
                bounding_box=None,
                cropped_image=None,
                original_width=original_width,
                original_height=original_height,
            )

        # Prepare response
        bbox_data = BoundingBox(
            x1=detection_data["bbox"]["x1"],
            y1=detection_data["bbox"]["y1"],
            x2=detection_data["bbox"]["x2"],
            y2=detection_data["bbox"]["y2"],
            confidence=detection_data["confidence"],
        )

        cropped_base64 = None
        if return_base64:
            cropped_base64 = pil_to_base64(detection_data["cropped_image"], format="PNG")

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
