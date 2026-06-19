"""DINOv2 patch-level verifier for salamander identification.

Uses mutual nearest neighbor matching on DINOv2 patch tokens to compare
individual spot patterns. This is far more discriminative than global
cosine similarity (CLS token) which only captures species-level features.
"""

import logging
from typing import TypedDict

import numpy as np
from PIL import Image

from app.identification.embedder import SalamanderEmbedder

logger = logging.getLogger(__name__)


# Default thresholds for patch matching score
DEFAULT_HIGH_THRESHOLD = 0.25
DEFAULT_MEDIUM_THRESHOLD = 0.15
DEFAULT_LOW_THRESHOLD = 0.10


class _VerifyResult(TypedDict):
    candidate_index: int
    is_same: bool
    score: float
    confidence: str
    cosine_similarity: float
    matches: int
    inliers: int


class SalamanderVerifier:
    """Verifies if two salamander images are the same individual.

    Uses DINOv2 patch-level mutual nearest neighbor matching.
    Each image is split into ~256 patches (16x16 grid at 224px / 14px patch size).
    Patches that are mutual best matches between the two images form
    correspondences — the ratio and quality of these correspondences
    determines whether the spot patterns match.
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
            high_threshold: Patch score above this → is_same=True, confidence=high.
            medium_threshold: Patch score above this → is_same=True, confidence=medium.
            low_threshold: Patch score above this → is_same=False, confidence=low.
                           Below this → is_same=False, confidence=high.
        """
        self.embedder = embedder
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

    def _classify(self, score: float) -> tuple[bool, str]:
        """Classify a patch matching score into a decision.

        Args:
            score: Patch matching score (0 to 1).

        Returns:
            Tuple of (is_same, confidence).
        """
        if score >= self.high_threshold:
            return True, "high"
        if score >= self.medium_threshold:
            return True, "medium"
        if score >= self.low_threshold:
            return False, "low"
        return False, "high"

    @staticmethod
    def _patch_match_score(patches1: np.ndarray, patches2: np.ndarray) -> float:
        """Compute patch-level matching score using mutual nearest neighbors.

        For each patch in image1, finds the best match in image2 (and vice versa).
        Only mutual best matches count — this filters out ambiguous correspondences
        (e.g. generic black/yellow patches that match many locations).

        Score = (n_mutual_matches / n_patches) * mean_similarity_of_mutual_matches

        Args:
            patches1: L2-normalized patch tokens from image 1, shape (N, D).
            patches2: L2-normalized patch tokens from image 2, shape (M, D).

        Returns:
            Matching score between 0 and 1.
        """
        # Cosine similarity matrix between all patch pairs
        sim_matrix = patches1 @ patches2.T  # (N, M)

        # Best match in each direction
        nn_1to2 = sim_matrix.argmax(axis=1)  # (N,)
        nn_2to1 = sim_matrix.argmax(axis=0)  # (M,)

        # Keep only mutual nearest neighbors
        mutual_sims = []
        for i, j in enumerate(nn_1to2):
            if nn_2to1[j] == i:
                mutual_sims.append(float(sim_matrix[i, j]))

        n_patches = len(patches1)
        if len(mutual_sims) == 0 or n_patches == 0:
            return 0.0

        match_ratio = len(mutual_sims) / n_patches
        mean_sim = float(np.mean(mutual_sims))

        return match_ratio * mean_sim

    def verify(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> dict:
        """Verify if two images are the same individual.

        Extracts patch tokens from both images and computes a matching
        score based on mutual nearest neighbor correspondences.

        Args:
            image1: First PIL Image.
            image2: Second PIL Image.

        Returns:
            Dictionary with verification results.
        """
        emb1, patches1 = self.embedder.extract_features(image1)
        emb2, patches2 = self.embedder.extract_features(image2)

        # Global cosine similarity (for reference / backward compat)
        cosine_sim = float(np.dot(emb1, emb2))

        # Patch-level matching (main score)
        score = self._patch_match_score(patches1, patches2)
        is_same, confidence = self._classify(score)

        return {
            "is_same": is_same,
            "score": float(score),
            "confidence": confidence,
            "cosine_similarity": cosine_sim,
            "matches": 0,
            "inliers": 0,
        }

    def verify_against_many(
        self,
        query_image: Image.Image,
        candidate_images: list[Image.Image],
    ) -> list[_VerifyResult]:
        """Verify a query image against multiple candidates.

        Extracts query features once, then compares against each candidate
        using patch-level matching.

        Args:
            query_image: Query PIL Image.
            candidate_images: List of candidate PIL Images.

        Returns:
            List of verification results, sorted by score descending.
        """
        if not candidate_images:
            return []

        query_emb, query_patches = self.embedder.extract_features(query_image)

        results: list[_VerifyResult] = []
        for idx, candidate in enumerate(candidate_images):
            cand_emb, cand_patches = self.embedder.extract_features(candidate)

            cosine_sim = float(np.dot(query_emb, cand_emb))
            score = self._patch_match_score(query_patches, cand_patches)
            is_same, confidence = self._classify(score)

            results.append(
                {
                    "candidate_index": idx,
                    "is_same": is_same,
                    "score": float(score),
                    "confidence": confidence,
                    "cosine_similarity": cosine_sim,
                    "matches": 0,
                    "inliers": 0,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
