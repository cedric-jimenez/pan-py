"""DINOv2 embedder for salamander identification.

Uses GeM-pooled patch tokens instead of the CLS token to produce
embeddings that capture local pattern details (spot arrangement)
rather than just species-level features.
"""

import logging
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

# GeM pooling exponent — higher values emphasize distinctive patches
GEM_EXPONENT = 3.0


class _ResizePad:
    """Resize longest side to target size, pad shorter side to square.

    Preserves the full salamander pattern instead of cropping.
    """

    def __init__(self, size: int, fill: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = F.resize(image, [new_h, new_w])

        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        return F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)


class SalamanderEmbedder:
    """Extracts DINOv2 embeddings from salamander images.

    Uses GeM-pooled patch tokens for individual-level discrimination.
    The CLS token captures species-level similarity (all fire salamanders
    score ~0.92), while patch tokens preserve local pattern details
    needed to distinguish individuals by their unique spot arrangement.
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
                _ResizePad(224),
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

    @staticmethod
    def _gem_pool(patch_tokens: torch.Tensor, p: float = GEM_EXPONENT) -> torch.Tensor:
        """Generalized Mean Pooling on patch tokens.

        More discriminative than average pooling — emphasizes
        distinctive patches over common background patches.

        Args:
            patch_tokens: Tensor of shape (B, N_patches, D).
            p: Exponent. Higher = closer to max pooling.

        Returns:
            Pooled tensor of shape (B, D).
        """
        return patch_tokens.clamp(min=1e-6).pow(p).mean(dim=1).pow(1.0 / p)

    def _forward_features(self, tensor: torch.Tensor) -> dict:
        """Run DINOv2 forward pass returning all features.

        Args:
            tensor: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Dict with 'x_norm_clstoken' and 'x_norm_patchtokens'.
        """
        with torch.no_grad():
            return self.model.forward_features(tensor)

    def extract_features(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Extract both global embedding and patch tokens in a single pass.

        Args:
            image: PIL Image (RGB or RGBA).

        Returns:
            Tuple of (embedding, patch_tokens):
                - embedding: L2-normalized GeM-pooled vector (D,)
                - patch_tokens: L2-normalized patch tokens (N_patches, D)

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image = self._prepare_image(image)
        tensor = self.transform(image).unsqueeze(0)

        features = self._forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"]  # (1, N, D)

        # Global embedding via GeM pooling
        embedding = self._gem_pool(patch_tokens)  # (1, D)
        embedding_np: np.ndarray = embedding.numpy().flatten()
        embedding_np = embedding_np / np.linalg.norm(embedding_np)

        # L2-normalize each patch token for cosine similarity
        tokens_np: np.ndarray = patch_tokens.numpy()[0]  # (N, D)
        norms = np.linalg.norm(tokens_np, axis=1, keepdims=True)
        tokens_np = tokens_np / np.clip(norms, 1e-6, None)

        return embedding_np, tokens_np

    def embed(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from an image.

        Uses GeM-pooled patch tokens for better individual discrimination.

        Args:
            image: PIL Image (RGB or RGBA).

        Returns:
            Normalized embedding vector (numpy array).

        Raises:
            RuntimeError: If model is not loaded.
        """
        embedding, _ = self.extract_features(image)
        return embedding

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

        features = self._forward_features(batch)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N, D)
        embeddings = self._gem_pool(patch_tokens)  # (B, D)

        # L2 normalize
        embeddings_np: np.ndarray = embeddings.numpy()
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / norms

        return embeddings_np
