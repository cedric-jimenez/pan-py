#!/usr/bin/env python3
"""
Convertit les images segmentées avec fond gris en images avec fond transparent.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def remove_gray_background(
    image_path: Path,
    output_path: Path,
    gray_value: int = 151,
    tolerance: int = 10,
) -> None:
    """Convertit le fond gris en transparent.

    Args:
        image_path: Chemin de l'image source.
        output_path: Chemin de l'image de sortie (PNG).
        gray_value: Valeur de gris du fond (0-255).
        tolerance: Tolérance pour la détection du gris.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    # Créer le canal alpha
    # Un pixel est considéré comme fond si R, G, B sont tous proches de gray_value
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    is_gray = (
        (np.abs(r.astype(int) - gray_value) <= tolerance)
        & (np.abs(g.astype(int) - gray_value) <= tolerance)
        & (np.abs(b.astype(int) - gray_value) <= tolerance)
    )

    # Alpha: 0 pour le fond, 255 pour le sujet
    alpha = np.where(is_gray, 0, 255).astype(np.uint8)

    # Créer l'image RGBA
    rgba = np.dstack((arr, alpha))
    result = Image.fromarray(rgba, mode="RGBA")
    result.save(output_path, "PNG")

    # Stats
    transparent_pct = (is_gray.sum() / is_gray.size) * 100
    print(f"  {image_path.name} -> {output_path.name} ({transparent_pct:.1f}% transparent)")


def main() -> None:
    input_dir = Path("poc/images")
    output_dir = Path("poc/images_transparent")
    output_dir.mkdir(exist_ok=True)

    print("Conversion des images avec fond transparent...")
    print("=" * 60)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in image_extensions:
            output_path = output_dir / f"{path.stem}.png"
            remove_gray_background(path, output_path)

    print("=" * 60)
    print(f"Images converties dans: {output_dir}")


if __name__ == "__main__":
    main()
