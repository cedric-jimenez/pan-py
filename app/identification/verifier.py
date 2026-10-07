"""SIFT + RANSAC verifier for salamander individual identification.

Identifies individuals by the *geometry* of their yellow spot pattern, which is
an individual fingerprint. SIFT keypoints are detected on the high-contrast
pattern, matched with a Lowe ratio test, then RANSAC keeps only the spatially
consistent correspondences. Different individuals share almost no consistent
keypoints, so the RANSAC inlier ratio cleanly separates same from different.

This replaces the earlier DINOv2 patch-matching score, which captured generic
appearance ("a fire salamander") rather than individual spot geometry and so
produced many false positives. On the labelled set (docs/images/, 14
individuals) SIFT+RANSAC reaches 100% precision at the is_same threshold with
~80% recall and 97.7% top-1 retrieval, versus ~80%/33% for the patch score.
See poc/benchmark_methods.py for the comparison.
"""

import logging
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Thresholds on the RANSAC inlier ratio: inliers / min(#keypoints of the pair).
# Calibrated on docs/images/ (14 individuals): same-individual median ~0.31,
# different-individual max ~0.06. See poc/benchmark_methods.py.
DEFAULT_HIGH_THRESHOLD = 0.15  # very confident same
DEFAULT_MEDIUM_THRESHOLD = 0.08  # ratio band for is_same (also gated by inlier count)
DEFAULT_LOW_THRESHOLD = 0.05  # below this: confidently different

# Absolute floor on the RANSAC inlier COUNT required to call two images the same
# individual, applied on top of the ratio bands. The ratio alone is confounded by
# keypoint count (a sparse crop inflates it), and on real data different individuals
# reach an inlier *ratio* up to ~0.07-0.10 by coincidence — but their inlier *count*
# never exceeds ~10, while genuine re-captures that are retrievable at all score 24+
# (see poc/eval_identification_e2e.py inlier-floor sweep + prod /similar). A floor of
# 12 gives 0 false positives across both labelled sets and the prod catalogue. Very
# distant re-captures (~2 years) fall below this but already fail today (dropped at
# cosine retrieval, or scoring in the <10 noise), so the floor doesn't regress them.
DEFAULT_MIN_INLIERS = 12

# SIFT / matching parameters.
_IMAGE_SIZE = 224  # resize-pad target (matches the embedder geometry)
_LOWE_RATIO = 0.75  # second-nearest-neighbour ratio test
_RANSAC_REPROJ_THRESHOLD = 5.0  # pixels, for cv2.findHomography
_MIN_MATCHES = 4  # cv2.findHomography needs at least 4 correspondences

# A SIFT feature set: (keypoint coordinates (N,2), descriptors (N,128) or None).
_SiftFeatures = tuple[np.ndarray, "np.ndarray | None"]


class _VerifyResult(TypedDict):
    candidate_index: int
    is_same: bool
    score: float
    confidence: str
    cosine_similarity: float
    matches: int
    inliers: int


class SalamanderVerifier:
    """Verifies whether two salamander images show the same individual.

    Uses SIFT keypoint matching with RANSAC geometric verification on the spot
    pattern. The score is the fraction of keypoints that form geometrically
    consistent correspondences between the two images.
    """

    def __init__(
        self,
        embedder: object | None = None,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
        min_inliers: int = DEFAULT_MIN_INLIERS,
    ) -> None:
        """Initialize the verifier.

        Args:
            embedder: Deprecated and unused — kept so existing construction
                (``SalamanderVerifier(embedder=...)``) keeps working. SIFT
                matching needs no neural model.
            high_threshold: Inlier ratio above this → is_same=True, confidence=high.
            medium_threshold: Inlier ratio above this → is_same=True, confidence=medium.
            low_threshold: Inlier ratio above this (but below medium) → is_same=False,
                confidence=low. Below this → is_same=False, confidence=high.
            min_inliers: Minimum RANSAC inlier COUNT to allow is_same=True, applied
                on top of the ratio bands (see DEFAULT_MIN_INLIERS). A pair that
                clears a ratio band but has fewer inliers is downgraded to
                (is_same=False, confidence="low").
        """
        del embedder  # intentionally unused
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.min_inliers = min_inliers
        self._sift = cv2.SIFT_create()  # type: ignore[attr-defined]
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _classify(self, score: float, inliers: int) -> tuple[bool, str]:
        """Classify a (inlier-ratio, inlier-count) pair into (is_same, confidence).

        The ratio sets the band; an absolute inlier-count floor then vetoes any
        "same" call backed by too few geometrically consistent points — these are
        the coincidental cross-individual matches that the ratio alone lets through.
        """
        if score >= self.high_threshold:
            is_same, confidence = True, "high"
        elif score >= self.medium_threshold:
            is_same, confidence = True, "medium"
        elif score >= self.low_threshold:
            is_same, confidence = False, "low"
        else:
            is_same, confidence = False, "high"
        if is_same and inliers < self.min_inliers:
            return False, "low"  # too few inliers to trust a same-individual call
        return is_same, confidence

    def _extract(self, image: Image.Image) -> _SiftFeatures:
        """Detect SIFT keypoints on the resize-padded greyscale image.

        Background (uniform white/grey/black) produces no keypoints, so SIFT
        naturally concentrates on the salamander's spot edges.
        """
        rgb = image.convert("RGB")
        w, h = rgb.size
        scale = _IMAGE_SIZE / max(w, h)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        canvas = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), (0, 0, 0))
        canvas.paste(
            rgb.resize((new_w, new_h)), ((_IMAGE_SIZE - new_w) // 2, (_IMAGE_SIZE - new_h) // 2)
        )

        gray = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        pts = (
            np.array([kp.pt for kp in keypoints], dtype=np.float32)
            if keypoints
            else np.zeros((0, 2), dtype=np.float32)
        )
        return pts, descriptors

    def _match_score(self, a: _SiftFeatures, b: _SiftFeatures) -> tuple[float, int, int]:
        """Score a pair via Lowe-ratio matching + RANSAC geometric verification.

        Returns (inlier_ratio, n_good_matches, n_inliers).
        """
        pts_a, des_a = a
        pts_b, des_b = b
        if des_a is None or des_b is None or len(des_a) < _MIN_MATCHES or len(des_b) < _MIN_MATCHES:
            return 0.0, 0, 0

        # Lowe ratio test on the two nearest neighbours.
        knn = self._matcher.knnMatch(des_a, des_b, k=2)
        good = [
            pair[0]
            for pair in knn
            if len(pair) == 2 and pair[0].distance < _LOWE_RATIO * pair[1].distance
        ]
        if len(good) < _MIN_MATCHES:
            return 0.0, len(good), 0

        # RANSAC: keep only geometrically consistent correspondences.
        src = np.array([pts_a[m.queryIdx] for m in good], dtype=np.float32)
        dst = np.array([pts_b[m.trainIdx] for m in good], dtype=np.float32)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_REPROJ_THRESHOLD)
        if mask is None:
            return 0.0, len(good), 0

        inliers = int(mask.sum())
        denom = min(len(pts_a), len(pts_b))
        score = inliers / denom if denom else 0.0
        return score, len(good), inliers

    def verify(self, image1: Image.Image, image2: Image.Image) -> dict:
        """Verify whether two images show the same individual."""
        score, matches, inliers = self._match_score(self._extract(image1), self._extract(image2))
        is_same, confidence = self._classify(score, inliers)
        return {
            "is_same": is_same,
            "score": float(score),
            "confidence": confidence,
            "cosine_similarity": 0.0,  # vestigial: kept for API back-compat
            "matches": matches,
            "inliers": inliers,
        }

    def verify_against_many(
        self,
        query_image: Image.Image,
        candidate_images: list[Image.Image],
        cosine_threshold: float = 0.0,
    ) -> list[_VerifyResult]:
        """Verify a query image against many candidates, sorted by score desc.

        Args:
            query_image: Query PIL Image.
            candidate_images: Candidate PIL Images.
            cosine_threshold: Deprecated and ignored (SIFT needs no fast-reject).
        """
        del cosine_threshold  # intentionally unused
        if not candidate_images:
            return []

        query_features = self._extract(query_image)
        results: list[_VerifyResult] = []
        for idx, candidate in enumerate(candidate_images):
            score, matches, inliers = self._match_score(query_features, self._extract(candidate))
            is_same, confidence = self._classify(score, inliers)
            results.append(
                {
                    "candidate_index": idx,
                    "is_same": is_same,
                    "score": float(score),
                    "confidence": confidence,
                    "cosine_similarity": 0.0,
                    "matches": matches,
                    "inliers": inliers,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
