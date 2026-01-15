"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    confidence: float = Field(..., description="Detection confidence score", ge=0, le=1)


class DetectionResponse(BaseModel):
    """Response model for salamander detection."""

    success: bool = Field(..., description="Whether detection was successful")
    message: str = Field(..., description="Status message")
    detected: bool = Field(..., description="Whether a salamander was detected")
    bounding_box: BoundingBox | None = Field(
        None, description="Bounding box of detected salamander"
    )
    cropped_image: str | None = Field(None, description="Base64 encoded cropped image")
    original_width: int = Field(..., description="Original image width")
    original_height: int = Field(..., description="Original image height")


class SegmentationResponse(BaseModel):
    """Response model for salamander segmentation."""

    success: bool = Field(..., description="Whether segmentation was successful")
    message: str = Field(..., description="Status message")
    detected: bool = Field(..., description="Whether a salamander was detected")
    bounding_box: BoundingBox | None = Field(
        None, description="Bounding box of detected salamander"
    )
    segmented_image: str | None = Field(
        None, description="Base64 encoded segmented image with background removed"
    )
    original_width: int = Field(..., description="Original image width")
    original_height: int = Field(..., description="Original image height")
    background_color: str = Field(..., description="Background color used (gray, white, black)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    yolo_loaded: bool = Field(..., description="Whether YOLO model is loaded")
    segment_loaded: bool = Field(False, description="Whether segmentation model is loaded")
    version: str = Field(..., description="API version")
