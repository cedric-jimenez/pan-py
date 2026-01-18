"""SIFT-based verifier for salamander identification."""

import logging
from typing import TypedDict

import cv2
import numpy as np
from cv2 import KeyPoint
from PIL import Image

logger = logging.getLogger(__name__)


# SIFT configuration
DEFAULT_N_FEATURES = 1000
DEFAULT_RATIO_THRESH = 0.75
DEFAULT_RANSAC_THRESH = 5.0
MIN_MATCHES_FOR_RANSAC = 4


class VerificationResult(TypedDict):
    candidate_index: int
    is_same: bool
    score: float
    confidence: str
    matches: int
    inliers: int
    keypoints_query: int
    keypoints_candidate: int


class SalamanderVerifier:
    """Verifies if two salamander images are the same individual using SIFT.

    SIFT detects local keypoints and matches them between images.
    RANSAC filters geometrically inconsistent matches.
    """

    def __init__(
        self,
        n_features: int = DEFAULT_N_FEATURES,
        ratio_thresh: float = DEFAULT_RATIO_THRESH,
        ransac_thresh: float = DEFAULT_RANSAC_THRESH,
    ) -> None:
        """Initialize the verifier.

        Args:
            n_features: Maximum number of SIFT keypoints per image.
            ratio_thresh: Lowe's ratio test threshold.
            ransac_thresh: RANSAC reprojection threshold in pixels.
        """
        self.n_features = n_features
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.sift = cv2.SIFT_create(nfeatures=n_features)  # type: ignore[attr-defined]

    def _pil_to_gray(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to grayscale numpy array.

        Handles RGBA by compositing on black background.

        Args:
            image: PIL Image.

        Returns:
            Grayscale numpy array.
        """
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (0, 0, 0))
            bg.paste(image, mask=image.split()[3])
            image = bg
        return np.array(image.convert("L"))

    def _extract_features(
        self,
        image: Image.Image,
    ) -> tuple[list, np.ndarray | None]:
        """Extract SIFT keypoints and descriptors.

        Args:
            image: PIL Image.

        Returns:
            Tuple (keypoints, descriptors).
        """
        gray = self._pil_to_gray(image)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def _match_descriptors(
        self,
        desc1: np.ndarray | None,
        desc2: np.ndarray | None,
    ) -> list[cv2.DMatch]:
        """Match descriptors using FLANN with Lowe's ratio test.

        Args:
            desc1: Descriptors from image 1.
            desc2: Descriptors from image 2.

        Returns:
            List of good matches.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        # FLANN matcher
        index_params: dict[str, bool | int | float | str] = {
            "algorithm": 1,
            "trees": 5,
        }
        search_params: dict[str, bool | int | float | str] = {
            "checks": 50,
        }

        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            # Fallback to BruteForce
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        return good_matches

    def _filter_ransac(
        self,
        kp1: list[KeyPoint],
        kp2: list[KeyPoint],
        matches: list[cv2.DMatch],
    ) -> tuple[list[cv2.DMatch], int]:
        """Filter matches using RANSAC.

        Args:
            kp1: Keypoints from image 1.
            kp2: Keypoints from image 2.
            matches: Raw matches.

        Returns:
            Tuple (filtered matches, number of inliers).
        """
        if len(matches) < MIN_MATCHES_FOR_RANSAC:
            return matches, len(matches)

        pts1 = np.array(
            [kp1[m.queryIdx].pt for m in matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        pts2 = np.array(
            [kp2[m.trainIdx].pt for m in matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        try:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_thresh)
        except cv2.error:
            return matches, len(matches)

        if mask is None:
            return matches, len(matches)

        mask = mask.ravel().astype(bool)
        inliers = [m for m, is_inlier in zip(matches, mask, strict=True) if is_inlier]

        return inliers, len(inliers)

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
            Dictionary with verification results:
                - is_same: Boolean indicating if same individual
                - score: Similarity score (0-1)
                - confidence: Confidence level (low/medium/high)
                - matches: Number of raw matches
                - inliers: Number of RANSAC inliers
                - keypoints1: Number of keypoints in image 1
                - keypoints2: Number of keypoints in image 2
        """
        # Extract features
        kp1, desc1 = self._extract_features(image1)
        kp2, desc2 = self._extract_features(image2)

        n_kp1 = len(kp1) if kp1 else 0
        n_kp2 = len(kp2) if kp2 else 0

        # Match
        matches = self._match_descriptors(desc1, desc2)
        n_matches = len(matches)

        # RANSAC
        inliers, n_inliers = self._filter_ransac(kp1, kp2, matches)

        # Compute score
        if n_kp1 == 0 or n_kp2 == 0:
            score = 0.0
        else:
            match_ratio = n_matches / np.sqrt(n_kp1 * n_kp2)
            inlier_ratio = n_inliers / n_matches if n_matches > 0 else 0.0
            score = match_ratio * (0.5 + 0.5 * inlier_ratio)

        # Determine result
        if score >= 0.10:
            is_same = True
            confidence = "high"
        elif score >= 0.05:
            is_same = True
            confidence = "medium"
        elif score >= 0.03:
            is_same = False
            confidence = "low"
        else:
            is_same = False
            confidence = "high"

        return {
            "is_same": is_same,
            "score": float(score),
            "confidence": confidence,
            "matches": n_matches,
            "inliers": n_inliers,
            "keypoints1": n_kp1,
            "keypoints2": n_kp2,
        }

    def verify_against_many(
        self,
        query_image: Image.Image,
        candidate_images: list[Image.Image],
    ) -> list[VerificationResult]:
        """Verify a query image against multiple candidates.

        Args:
            query_image: Query PIL Image.
            candidate_images: List of candidate PIL Images.

        Returns:
            List of verification results, sorted by score descending.
        """
        # Extract query features once
        kp_query, desc_query = self._extract_features(query_image)
        n_kp_query = len(kp_query) if kp_query else 0

        results: list[VerificationResult] = []
        for idx, candidate in enumerate(candidate_images):
            kp_cand, desc_cand = self._extract_features(candidate)
            n_kp_cand = len(kp_cand) if kp_cand else 0

            # Match
            matches = self._match_descriptors(desc_query, desc_cand)
            n_matches = len(matches)

            # RANSAC
            _, n_inliers = self._filter_ransac(kp_query, kp_cand, matches)

            # Score
            if n_kp_query == 0 or n_kp_cand == 0:
                score = 0.0
            else:
                match_ratio = n_matches / np.sqrt(n_kp_query * n_kp_cand)
                inlier_ratio = n_inliers / n_matches if n_matches > 0 else 0.0
                score = match_ratio * (0.5 + 0.5 * inlier_ratio)

            # Determine result
            if score >= 0.10:
                is_same = True
                confidence = "high"
            elif score >= 0.05:
                is_same = True
                confidence = "medium"
            elif score >= 0.03:
                is_same = False
                confidence = "low"
            else:
                is_same = False
                confidence = "high"

            results.append(
                {
                    "candidate_index": idx,
                    "is_same": is_same,
                    "score": float(score),
                    "confidence": confidence,
                    "matches": n_matches,
                    "inliers": n_inliers,
                    "keypoints_query": n_kp_query,
                    "keypoints_candidate": n_kp_cand,
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
