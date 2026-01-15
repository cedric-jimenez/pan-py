"""Tests for the main FastAPI application."""

import pytest
import warnings
from fastapi.testclient import TestClient

from app.main import app

# Suppress httpx deprecation warning about 'app' shortcut
warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "yolo_loaded" in data
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "detection" in data
    assert "segmentation" in data
    assert "loaded" in data["detection"]
    assert "loaded" in data["segmentation"]


@pytest.mark.asyncio
async def test_crop_salamander_missing_file():
    """Test crop endpoint without a file."""
    response = client.post("/crop-salamander")
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_crop_salamander_invalid_confidence():
    """Test crop endpoint with invalid confidence value."""
    response = client.post("/crop-salamander?confidence=2.0")
    assert response.status_code == 422  # Validation error (confidence > 1.0)
