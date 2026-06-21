"""Tests for the identification scoring logic.

These exercise the pure numpy/PIL helpers (foreground masking, patch matching,
threshold classification) without loading the DINOv2 model.
"""

import numpy as np
from PIL import Image

from app.identification.embedder import SalamanderEmbedder
from app.identification.verifier import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MEDIUM_THRESHOLD,
    SalamanderVerifier,
)


def _verifier() -> SalamanderVerifier:
    # _classify and _patch_match_score never touch the embedder.
    return SalamanderVerifier(embedder=None)  # type: ignore[arg-type]


# --- _classify ---------------------------------------------------------------
def test_classify_bands():
    v = _verifier()
    assert v._classify(DEFAULT_HIGH_THRESHOLD + 0.01) == (True, "high")
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD + 0.01) == (True, "medium")
    assert v._classify(DEFAULT_LOW_THRESHOLD + 0.01) == (False, "low")
    assert v._classify(DEFAULT_LOW_THRESHOLD - 0.01) == (False, "high")


def test_is_same_boundary_is_medium_threshold():
    """A pair counts as 'same' only at or above the medium threshold."""
    v = _verifier()
    is_same_just_below, _ = v._classify(DEFAULT_MEDIUM_THRESHOLD - 0.001)
    is_same_at, _ = v._classify(DEFAULT_MEDIUM_THRESHOLD)
    assert is_same_just_below is False
    assert is_same_at is True


# --- _patch_match_score ------------------------------------------------------
def test_identical_patches_score_high():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((16, 8))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    score = SalamanderVerifier._patch_match_score(p, p.copy())
    assert score > 0.99  # every patch is its own mutual nearest neighbour


def test_empty_patches_score_zero():
    p = np.zeros((0, 8))
    assert SalamanderVerifier._patch_match_score(p, p) == 0.0


def test_foreground_mask_ignores_shared_background():
    """Two images with identical background but different foreground should
    score far lower once the background patches are masked out."""
    rng = np.random.default_rng(1)
    bg = rng.standard_normal((8, 8))
    bg /= np.linalg.norm(bg, axis=1, keepdims=True)
    fg_a = rng.standard_normal((8, 8))
    fg_a /= np.linalg.norm(fg_a, axis=1, keepdims=True)
    fg_b = rng.standard_normal((8, 8))
    fg_b /= np.linalg.norm(fg_b, axis=1, keepdims=True)

    patches1 = np.vstack([bg, fg_a])
    patches2 = np.vstack([bg, fg_b])
    mask = np.array([False] * 8 + [True] * 8)  # first 8 are background

    with_bg = SalamanderVerifier._patch_match_score(patches1, patches2)
    masked = SalamanderVerifier._patch_match_score(patches1, patches2, mask, mask)
    assert masked < with_bg


def test_empty_foreground_mask_falls_back_to_all_patches():
    rng = np.random.default_rng(2)
    p = rng.standard_normal((16, 8))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    none_fg = np.zeros(16, dtype=bool)
    # Should not return 0 just because the mask is empty.
    score = SalamanderVerifier._patch_match_score(p, p.copy(), none_fg, none_fg)
    assert score > 0.99


# --- foreground mask ---------------------------------------------------------
def test_foreground_mask_all_grey_is_background():
    img = Image.new("RGB", (224, 224), (150, 150, 150))  # segmenter fill colour
    mask = SalamanderEmbedder._foreground_mask(img, 256)
    assert mask.shape == (256,)
    assert not mask.any()


def test_foreground_mask_detects_coloured_region():
    img = Image.new("RGB", (224, 224), (150, 150, 150))
    # Paint a saturated yellow block (salamander spot) in the centre.
    img.paste((230, 200, 20), (80, 80, 144, 144))
    mask = SalamanderEmbedder._foreground_mask(img, 256)
    assert mask.any()
    assert not mask.all()  # background still masked out


def test_foreground_mask_non_square_returns_all_true():
    img = Image.new("RGB", (224, 224), (150, 150, 150))
    mask = SalamanderEmbedder._foreground_mask(img, 255)  # not a perfect square
    assert mask.all()
