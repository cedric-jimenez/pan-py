#!/usr/bin/env python3
"""Generate OpenAPI specification from FastAPI app.

This script generates an openapi.yml file from the FastAPI application.
Useful to keep the OpenAPI spec in sync with the actual API.

Usage:
    python scripts/generate_openapi.py
"""

import json
import sys
from pathlib import Path

import yaml

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


def generate_openapi_yaml():
    """Generate OpenAPI YAML specification from FastAPI app."""
    # Get OpenAPI schema from FastAPI
    openapi_schema = app.openapi()

    # Update info with additional details
    openapi_schema["info"].update(
        {
            "description": """API REST pour détecter et cropper des salamandres dans des images en utilisant un modèle YOLO custom.

## Fonctionnalités
- Détection automatique de salamandres avec YOLO
- Cropping intelligent centré sur la salamandre détectée
- Réponse en base64 pour faciliter l'utilisation côté frontend
- Support CORS pour intégration avec Next.js/Vercel

## Utilisation
1. Envoyez une image via POST /crop-salamander
2. Recevez la détection avec bounding box et image croppée en base64
3. Utilisez l'image croppée directement dans votre frontend
""",
            "contact": {"name": "Cédric Jimenez"},
            "license": {"name": "MIT"},
        }
    )

    # Add servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Serveur de développement local"},
        {
            "url": "https://your-api.railway.app",
            "description": "Serveur de production (Railway)",
        },
    ]

    # Save as YAML
    output_path = Path(__file__).parent.parent / "openapi.yml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            openapi_schema,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"✅ OpenAPI specification generated: {output_path}")
    print(f"   Endpoints: {len(openapi_schema['paths'])}")
    print(f"   Schemas: {len(openapi_schema['components']['schemas'])}")


if __name__ == "__main__":
    generate_openapi_yaml()
