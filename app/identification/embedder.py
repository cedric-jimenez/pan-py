"""DINOv2 embedder for salamander identification."""

import logging
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class SalamanderEmbedder:
    """Extracts DINOv2 embeddings from salamander images.

    DINOv2 produces a single vector per image that can be stored in
    a vector database (e.g., pgvector) for similarity search.
    """

    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        """Initialize the embedder.

        Args:
            model_name: DINOv2 model variant. Options:
                - dinov2_vits14: ViT-S/14 (384D, fast, recommended)
                - dinov2_vitb14: ViT-B/14 (768D, more accurate)
        """
        self.model_name = model_name
        self.model: torch.nn.Module | None = None
        self.transform: transforms.Compose | None = None
        self.embedding_dim: int = 384 if "vits" in model_name else 768

    def load_model(self) -> None:
        """Load the DINOv2 model."""
        if self.model is not None:
            return

        logger.info(f"Loading DINOv2 model: {self.model_name}")

        # Suppress xFormers warning - xFormers is not needed for CPU-only inference
        warnings.filterwarnings(
            "ignore",
            message="xFormers is not available",
            category=UserWarning,
        )

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            verbose=False,
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.embedding_dim = self.model.embed_dim
        logger.info(f"DINOv2 model loaded (embedding dim: {self.embedding_dim})")

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Prepare image for embedding extraction.

        Handles RGBA images by compositing on black background.

        Args:
            image: PIL Image.

        Returns:
            RGB PIL Image.
        """
        if image.mode == "RGBA":
            # Composite on black background
            bg = Image.new("RGB", image.size, (0, 0, 0))
            bg.paste(image, mask=image.split()[3])
            return bg
        return image.convert("RGB")

    def embed(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from an image.

        Args:
            image: PIL Image (RGB or RGBA).

        Returns:
            Normalized embedding vector (numpy array).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare image
        image = self._prepare_image(image)
        tensor = self.transform(image).unsqueeze(0)

        # Extract embedding
        with torch.no_grad():
            embedding = self.model(tensor)

        # Normalize L2
        embedding_np: np.ndarray = embedding.numpy().flatten()
        embedding_np = embedding_np / np.linalg.norm(embedding_np)

        return embedding_np

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Extract embeddings from multiple images.

        Args:
            images: List of PIL Images.

        Returns:
            Array of shape (N, embedding_dim) with normalized embeddings.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare batch
        tensors = []
        for image in images:
            image = self._prepare_image(image)
            tensors.append(self.transform(image))

        batch = torch.stack(tensors)

        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model(batch)

        # Normalize L2
        embeddings_np: np.ndarray = embeddings.numpy()
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / norms

        return embeddings_np
