"""Tests for the SIFT+RANSAC identification logic.

Uses small synthetic "spot pattern" images so the tests run without the labelled
dataset or any neural model.
"""

from PIL import Image, ImageDraw

from app.identification.verifier import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MEDIUM_THRESHOLD,
    SalamanderVerifier,
)

# A fixed set of "spots" — distinct blobs give SIFT plenty of keypoints.
_SPOTS = [
    (40, 50, 14),
    (90, 40, 10),
    (150, 70, 18),
    (60, 120, 12),
    (120, 140, 16),
    (180, 160, 11),
    (30, 180, 13),
    (160, 110, 9),
    (100, 90, 15),
    (70, 200, 10),
]
_OTHER_SPOTS = [
    (55, 30, 11),
    (110, 75, 17),
    (35, 95, 9),
    (170, 45, 13),
    (140, 175, 15),
    (80, 150, 12),
    (190, 120, 10),
    (50, 160, 16),
    (130, 60, 14),
    (95, 195, 11),
]


def _spot_image(spots, offset=(0, 0)) -> Image.Image:
    """Draw yellow blobs on a black background (a stand-in salamander pattern)."""
    img = Image.new("RGB", (224, 224), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    ox, oy = offset
    for x, y, r in spots:
        draw.ellipse([x + ox - r, y + oy - r, x + ox + r, y + oy + r], fill=(230, 200, 20))
    return img


def _verifier() -> SalamanderVerifier:
    return SalamanderVerifier()


# --- _classify ---------------------------------------------------------------
def test_classify_bands():
    v = _verifier()
    assert v._classify(DEFAULT_HIGH_THRESHOLD + 0.01) == (True, "high")
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD + 0.001) == (True, "medium")
    assert v._classify(DEFAULT_LOW_THRESHOLD + 0.001) == (False, "low")
    assert v._classify(DEFAULT_LOW_THRESHOLD - 0.001) == (False, "high")


def test_is_same_boundary_is_medium_threshold():
    v = _verifier()
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD - 0.001)[0] is False
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD)[0] is True


# --- SIFT matching -----------------------------------------------------------
def test_identical_pattern_scores_same():
    v = _verifier()
    img = _spot_image(_SPOTS)
    result = v.verify(img, img.copy())
    assert result["is_same"] is True
    assert result["inliers"] > 0
    assert result["matches"] >= result["inliers"]


def test_translated_pattern_still_matches():
    """RANSAC should tolerate a rigid shift of the same pattern."""
    v = _verifier()
    result = v.verify(_spot_image(_SPOTS), _spot_image(_SPOTS, offset=(8, 6)))
    assert result["is_same"] is True


def test_different_patterns_score_different():
    v = _verifier()
    result = v.verify(_spot_image(_SPOTS), _spot_image(_OTHER_SPOTS))
    assert result["is_same"] is False
    assert result["score"] < DEFAULT_MEDIUM_THRESHOLD


def test_blank_image_scores_zero():
    """No keypoints -> no match, never a false positive."""
    v = _verifier()
    blank = Image.new("RGB", (224, 224), (150, 150, 150))
    result = v.verify(blank, _spot_image(_SPOTS))
    assert result["score"] == 0.0
    assert result["is_same"] is False
    assert result["matches"] == 0
    assert result["inliers"] == 0


# --- verify_against_many -----------------------------------------------------
def test_verify_against_many_ranks_true_match_first():
    v = _verifier()
    query = _spot_image(_SPOTS)
    candidates = [
        _spot_image(_OTHER_SPOTS),  # different individual
        _spot_image(_SPOTS, offset=(5, 5)),  # same individual, shifted
        Image.new("RGB", (224, 224), (150, 150, 150)),  # blank
    ]
    results = v.verify_against_many(query, candidates)
    assert results[0]["candidate_index"] == 1  # the true match ranks first
    assert results[0]["is_same"] is True
    # scores are sorted descending
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)


def test_verify_against_many_empty_returns_empty():
    assert _verifier().verify_against_many(_spot_image(_SPOTS), []) == []


def test_embedder_kwarg_is_accepted_for_backcompat():
    """Old construction SalamanderVerifier(embedder=...) must still work."""
    v = SalamanderVerifier(embedder=object())
    assert isinstance(v, SalamanderVerifier)


def test_score_is_json_serializable_float():
    """score must be a plain float (not numpy) for the JSON response."""
    v = _verifier()
    result = v.verify(_spot_image(_SPOTS), _spot_image(_SPOTS))
    assert type(result["score"]) is float
