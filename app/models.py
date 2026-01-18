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
    embedder_loaded: bool = Field(False, description="Whether DINOv2 embedder is loaded")
    version: str = Field(..., description="API version")


class EmbeddingResponse(BaseModel):
    """Response model for embedding extraction."""

    success: bool = Field(..., description="Whether extraction was successful")
    message: str = Field(..., description="Status message")
    embedding: list[float] | None = Field(None, description="Normalized embedding vector")
    embedding_dim: int = Field(..., description="Embedding dimension")
    model: str = Field(..., description="Model used for extraction")


class VerificationResult(BaseModel):
    """Result of a single verification."""

    candidate_index: int = Field(..., description="Index of the candidate image")
    is_same: bool = Field(..., description="Whether likely the same individual")
    score: float = Field(..., description="Similarity score")
    confidence: str = Field(..., description="Confidence level: low, medium, high")
    matches: int = Field(..., description="Number of raw SIFT matches")
    inliers: int = Field(..., description="Number of RANSAC inliers")


class VerificationResponse(BaseModel):
    """Response model for SIFT verification."""

    success: bool = Field(..., description="Whether verification was successful")
    message: str = Field(..., description="Status message")
    results: list[VerificationResult] = Field(
        default_factory=list, description="Verification results sorted by score"
    )
