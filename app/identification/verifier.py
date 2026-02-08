"""DINOv2-based verifier for salamander identification.

Uses cosine similarity on DINOv2 embeddings instead of SIFT keypoint matching.
This approach is more robust for deformable objects like salamanders and
leverages color/pattern information that SIFT (grayscale-only) would miss.
"""

import logging
from typing import TypedDict

import numpy as np
from PIL import Image

from app.identification.embedder import SalamanderEmbedder

logger = logging.getLogger(__name__)


# Default cosine similarity thresholds
DEFAULT_HIGH_THRESHOLD = 0.70
DEFAULT_MEDIUM_THRESHOLD = 0.50
DEFAULT_LOW_THRESHOLD = 0.40


class VerificationResult(TypedDict):
    candidate_index: int
    is_same: bool
    score: float
    confidence: str
    cosine_similarity: float
    matches: int
    inliers: int


class SalamanderVerifier:
    """Verifies if two salamander images are the same individual.

    Uses DINOv2 embeddings and cosine similarity for robust matching
    that handles deformable objects and preserves color information.
    """

    def __init__(
        self,
        embedder: SalamanderEmbedder,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
    ) -> None:
        """Initialize the verifier.

        Args:
            embedder: A loaded SalamanderEmbedder instance.
            high_threshold: Cosine similarity above this → is_same=True, confidence=high.
            medium_threshold: Cosine similarity above this → is_same=True, confidence=medium.
            low_threshold: Cosine similarity above this → is_same=False, confidence=low.
                           Below this → is_same=False, confidence=high.
        """
        self.embedder = embedder
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

    def _classify(self, cosine_sim: float) -> tuple[bool, str, float]:
        """Classify a cosine similarity score into a decision.

        Args:
            cosine_sim: Cosine similarity value (-1 to 1).

        Returns:
            Tuple of (is_same, confidence, score).
            score is cosine_sim clamped to [0, 1].
        """
        score = float(max(0.0, min(1.0, cosine_sim)))

        if cosine_sim >= self.high_threshold:
            return True, "high", score
        if cosine_sim >= self.medium_threshold:
            return True, "medium", score
        if cosine_sim >= self.low_threshold:
            return False, "low", score
        return False, "high", score

    def verify(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> dict:
        """Verify if two images are the same individual.

        Args:
            image1: First PIL Image.
            image2: Second PIL Image.

        Returns:
            Dictionary with verification results.
        """
        emb1 = self.embedder.embed(image1)
        emb2 = self.embedder.embed(image2)

        cosine_sim = float(np.dot(emb1, emb2))
        is_same, confidence, score = self._classify(cosine_sim)

        return {
            "is_same": is_same,
            "score": score,
            "confidence": confidence,
            "cosine_similarity": cosine_sim,
            "matches": 0,
            "inliers": 0,
        }

    def verify_against_many(
        self,
        query_image: Image.Image,
        candidate_images: list[Image.Image],
    ) -> list[VerificationResult]:
        """Verify a query image against multiple candidates.

        Embeds the query once, then compares against each candidate.

        Args:
            query_image: Query PIL Image.
            candidate_images: List of candidate PIL Images.

        Returns:
            List of verification results, sorted by score descending.
        """
        query_emb = self.embedder.embed(query_image)

        if candidate_images:
            candidate_embs = self.embedder.embed_batch(candidate_images)
        else:
            return []

        # Cosine similarities (embeddings are already L2-normalized)
        similarities = candidate_embs @ query_emb

        results: list[VerificationResult] = []
        for idx, cosine_sim in enumerate(similarities):
            cosine_sim = float(cosine_sim)
            is_same, confidence, score = self._classify(cosine_sim)

            results.append(
                {
                    "candidate_index": idx,
                    "is_same": is_same,
                    "score": score,
                    "confidence": confidence,
                    "cosine_similarity": cosine_sim,
                    "matches": 0,
                    "inliers": 0,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
