"""DINOv2 patch-level verifier for salamander identification (v2).

Uses CSLS-scored mutual nearest neighbor matching on DINOv2 patch tokens
with padding filtering and spatial coherence verification.

Improvements over v0 (raw MNN on cosine similarity):
  - A1: Padding filtering — removes low-norm patches from black padding regions
  - A2: CSLS anti-hub scoring — penalises hub patches that match everything
  - A3: Spatial coherence — keeps only geometrically consistent matches
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

# --- v2 algorithm constants ---
GRID_SIZE = 16  # 224px / 14px patch size
CONTENT_RATIO_THRESHOLD = 0.5  # Patches with less than 50% content pixels are padding
CSLS_K = 10  # Number of neighbours for CSLS hub penalty
SPATIAL_INLIER_THRESHOLD = 2.0  # Max deviation from median displacement (in patches)


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

    Uses DINOv2 patch-level mutual nearest neighbor matching with CSLS
    scoring and spatial coherence filtering.
    Each image is split into ~256 patches (16x16 grid at 224px / 14px patch size).
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
        content_ratio1: np.ndarray,
        content_ratio2: np.ndarray,
    ) -> tuple[float, int, int]:
        """Compute patch-level matching score using CSLS + spatial coherence.

        Algorithm:
          A1 - Filter out padding patches (low content ratio from geometric mask).
          A2 - Build CSLS similarity matrix (penalises hub patches).
          MNN on CSLS matrix to find mutual nearest neighbours.
          A3 - Keep only spatially coherent matches (inliers).
          Score = (n_inliers / n_mutual) * mean_csls_of_inliers, clamped to [0, 1].

        Args:
            patches1: L2-normalised patch tokens from image 1, shape (N, D).
            patches2: L2-normalised patch tokens from image 2, shape (M, D).
            content_ratio1: Per-patch content ratio for image 1, shape (N,).
            content_ratio2: Per-patch content ratio for image 2, shape (M,).

        Returns:
            Tuple of (score, n_mutual_matches, n_inliers).
        """
        # --- A1: Padding filtering ---
        mask1 = content_ratio1 >= CONTENT_RATIO_THRESHOLD
        mask2 = content_ratio2 >= CONTENT_RATIO_THRESHOLD

        # Original indices of kept patches (needed for grid positions in A3)
        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]

        p1 = patches1[mask1]  # (N', D)
        p2 = patches2[mask2]  # (M', D)

        if len(p1) == 0 or len(p2) == 0:
            return 0.0, 0, 0

        # --- A2: CSLS scoring ---
        # Cosine similarity matrix (patches are already L2-normalised)
        cos_sim = p1 @ p2.T  # (N', M')

        # Mean similarity to k nearest neighbours for each patch
        k1 = min(CSLS_K, cos_sim.shape[1])
        k2 = min(CSLS_K, cos_sim.shape[0])

        # For each row (patch in p1): mean of top-k similarities to p2
        # np.partition is O(n) — partitions so that the k-th largest is in place
        top_k_1 = np.partition(cos_sim, -k1, axis=1)[:, -k1:]  # (N', k1)
        mean_knn_1 = top_k_1.mean(axis=1)  # (N',)

        # For each column (patch in p2): mean of top-k similarities to p1
        top_k_2 = np.partition(cos_sim, -k2, axis=0)[-k2:, :]  # (k2, M')
        mean_knn_2 = top_k_2.mean(axis=0)  # (M',)

        # CSLS(x, y) = 2 * cos(x, y) - mean_kNN(x) - mean_kNN(y)
        csls = 2.0 * cos_sim - mean_knn_1[:, np.newaxis] - mean_knn_2[np.newaxis, :]

        # --- MNN on CSLS ---
        nn_1to2 = csls.argmax(axis=1)  # (N',)
        nn_2to1 = csls.argmax(axis=0)  # (M',)

        # Collect mutual nearest neighbour pairs
        mutual_i = []  # indices into p1
        mutual_j = []  # indices into p2
        mutual_csls = []
        for i, j in enumerate(nn_1to2):
            if nn_2to1[j] == i:
                mutual_i.append(i)
                mutual_j.append(j)
                mutual_csls.append(float(csls[i, j]))

        n_mutual = len(mutual_i)
        if n_mutual == 0:
            return 0.0, 0, 0

        # --- A3: Spatial coherence ---
        # Recover grid positions from original patch indices
        # Patches are in raster scan order: row = idx // GRID_SIZE, col = idx % GRID_SIZE
        orig_i = idx1[mutual_i]  # original indices in full patch array
        orig_j = idx2[mutual_j]

        pos_i = np.column_stack([orig_i // GRID_SIZE, orig_i % GRID_SIZE])  # (n_mutual, 2)
        pos_j = np.column_stack([orig_j // GRID_SIZE, orig_j % GRID_SIZE])  # (n_mutual, 2)

        displacements = pos_j - pos_i  # (n_mutual, 2)

        # Robust estimate of global displacement: component-wise median
        median_disp = np.median(displacements, axis=0)  # (2,)

        # Deviation from median displacement
        deviations = np.linalg.norm(displacements - median_disp, axis=1)  # (n_mutual,)
        inlier_mask = deviations <= SPATIAL_INLIER_THRESHOLD

        n_inliers = int(inlier_mask.sum())
        if n_inliers == 0:
            return 0.0, n_mutual, 0

        # Score = spatial_ratio * mean_csls_of_inliers (clamped to [0, 1])
        inlier_csls = np.array(mutual_csls)[inlier_mask]
        mean_csls = float(np.clip(inlier_csls.mean(), 0.0, 1.0))
        spatial_ratio = n_inliers / n_mutual

        score = float(np.clip(spatial_ratio * mean_csls, 0.0, 1.0))
        return score, n_mutual, n_inliers

    def verify(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> dict:
        """Verify if two images are the same individual.

        Extracts patch tokens from both images and computes a matching
        score based on CSLS mutual nearest neighbours with spatial coherence.

        Args:
            image1: First PIL Image.
            image2: Second PIL Image.

        Returns:
            Dictionary with verification results.
        """
        emb1, patches1, cr1 = self.embedder.extract_features(image1)
        emb2, patches2, cr2 = self.embedder.extract_features(image2)

        # Global cosine similarity (for reference / backward compat)
        cosine_sim = float(np.dot(emb1, emb2))

        # Patch-level matching (main score)
        score, matches, inliers = self._patch_match_score(
            patches1, patches2, cr1, cr2,
        )
        is_same, confidence = self._classify(score)

        return {
            "is_same": is_same,
            "score": float(score),
            "confidence": confidence,
            "cosine_similarity": cosine_sim,
            "matches": matches,
            "inliers": inliers,
        }

    def verify_against_many(
        self,
        query_image: Image.Image,
        candidate_images: list[Image.Image],
    ) -> list[VerificationResult]:
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

        query_emb, query_patches, query_cr = self.embedder.extract_features(
            query_image,
        )

        results: list[VerificationResult] = []
        for idx, candidate in enumerate(candidate_images):
            cand_emb, cand_patches, cand_cr = self.embedder.extract_features(
                candidate,
            )

            cosine_sim = float(np.dot(query_emb, cand_emb))
            score, matches, inliers = self._patch_match_score(
                query_patches, cand_patches, query_cr, cand_cr,
            )
            is_same, confidence = self._classify(score)

            results.append(
                {
                    "candidate_index": idx,
                    "is_same": is_same,
                    "score": float(score),
                    "confidence": confidence,
                    "cosine_similarity": cosine_sim,
                    "matches": matches,
                    "inliers": inliers,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
