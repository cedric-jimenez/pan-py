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


# Default thresholds for patch matching score.
# Calibrated on the labelled set in docs/images/ (14 individuals, foreground-
# masked symmetric scoring): same-individual pairs mean ~0.48, different ~0.30.
# A pair is judged "same" only at >= medium. See poc/eval_identification.py.
DEFAULT_HIGH_THRESHOLD = 0.55  # confident same (~100% precision on the eval set)
DEFAULT_MEDIUM_THRESHOLD = 0.50  # is_same=True boundary
DEFAULT_LOW_THRESHOLD = 0.40  # below this: confidently different


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
    def _patch_match_score(
        patches1: np.ndarray,
        patches2: np.ndarray,
        fg_mask1: np.ndarray | None = None,
        fg_mask2: np.ndarray | None = None,
    ) -> float:
        """Compute patch-level matching score using mutual nearest neighbors.

        For each patch in image1, finds the best match in image2 (and vice versa).
        Only mutual best matches count — this filters out ambiguous correspondences
        (e.g. generic black/yellow patches that match many locations).

        Background patches are dropped first (via the foreground masks) so that
        shared backgrounds — white crops, grey-150 segmenter fill, black padding —
        can't manufacture matches between different individuals. The match ratio
        is normalized by the *smaller* foreground set (symmetric), so a larger
        image cannot dilute the score.

        Score = (n_mutual_matches / min(N_fg, M_fg)) * mean_similarity_of_matches

        Args:
            patches1: L2-normalized patch tokens from image 1, shape (N, D).
            patches2: L2-normalized patch tokens from image 2, shape (M, D).
            fg_mask1: Optional (N,) bool foreground mask for image 1.
            fg_mask2: Optional (M,) bool foreground mask for image 2.

        Returns:
            Matching score between 0 and 1.
        """
        # Restrict to foreground patches when masks are available and non-empty;
        # fall back to all patches otherwise so the score is never degenerate.
        if fg_mask1 is not None and fg_mask1.any():
            patches1 = patches1[fg_mask1]
        if fg_mask2 is not None and fg_mask2.any():
            patches2 = patches2[fg_mask2]

        if len(patches1) == 0 or len(patches2) == 0:
            return 0.0

        # Cosine similarity matrix between all patch pairs
        sim_matrix = patches1 @ patches2.T  # (N, M)

        # Best match in each direction
        nn_1to2 = sim_matrix.argmax(axis=1)  # (N,)
        nn_2to1 = sim_matrix.argmax(axis=0)  # (M,)

        # Keep only mutual nearest neighbors (vectorized)
        indices = np.arange(len(patches1))
        mutual_mask = nn_2to1[nn_1to2] == indices
        if not mutual_mask.any():
            return 0.0

        mutual_sims = sim_matrix[indices[mutual_mask], nn_1to2[mutual_mask]]
        denom = min(len(patches1), len(patches2))
        match_ratio = float(mutual_mask.sum()) / denom
        return match_ratio * float(mutual_sims.mean())

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
        emb1, patches1, fg1 = self.embedder.extract_features(image1)
        emb2, patches2, fg2 = self.embedder.extract_features(image2)

        # Global cosine similarity (for reference / backward compat)
        cosine_sim = float(np.dot(emb1, emb2))

        # Patch-level matching (main score)
        score = self._patch_match_score(patches1, patches2, fg1, fg2)
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
        cosine_threshold: float = 0.0,
    ) -> list[_VerifyResult]:
        """Verify a query image against multiple candidates.

        Extracts query features once, then compares against each candidate
        using patch-level matching. Candidates whose cosine similarity falls
        below cosine_threshold are fast-rejected without running patch matching.

        Args:
            query_image: Query PIL Image.
            candidate_images: List of candidate PIL Images.
            cosine_threshold: Skip patch matching for candidates with cosine
                similarity below this value (0.0 = disabled).

        Returns:
            List of verification results, sorted by score descending.
        """
        if not candidate_images:
            return []

        query_emb, query_patches, query_fg = self.embedder.extract_features(query_image)

        results: list[_VerifyResult] = []
        for idx, candidate in enumerate(candidate_images):
            cand_emb, cand_patches, cand_fg = self.embedder.extract_features(candidate)

            cosine_sim = float(np.dot(query_emb, cand_emb))

            if cosine_sim < cosine_threshold:
                results.append(
                    {
                        "candidate_index": idx,
                        "is_same": False,
                        "score": 0.0,
                        "confidence": "high",
                        "cosine_similarity": cosine_sim,
                        "matches": 0,
                        "inliers": 0,
                    }
                )
                continue

            score = self._patch_match_score(query_patches, cand_patches, query_fg, cand_fg)
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
