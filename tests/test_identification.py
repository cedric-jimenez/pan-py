"""Tests for the SIFT+RANSAC identification logic.

Uses small synthetic "spot pattern" images so the tests run without the labelled
dataset or any neural model.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.identification.feature_store import (
    EMPTY_PHOTO,
    BucketPersistence,
    FeatureStore,
    PhotoFeatures,
    _to_compact,
    decode_shard,
    encode_shard,
    shard_of,
)
from app.identification.straighten import straighten, strip_features
from app.identification.verifier import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MEDIUM_THRESHOLD,
    DEFAULT_MIN_INLIERS,
    EMPTY_FEATURES,
    SalamanderVerifier,
    extract_sift_features,
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
_ENOUGH = 50  # inlier count well above DEFAULT_MIN_INLIERS, so the ratio band decides


def test_classify_bands():
    v = _verifier()
    assert v._classify(DEFAULT_HIGH_THRESHOLD + 0.01, _ENOUGH) == (True, "high")
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD + 0.001, _ENOUGH) == (True, "medium")
    assert v._classify(DEFAULT_LOW_THRESHOLD + 0.001, _ENOUGH) == (False, "low")
    assert v._classify(DEFAULT_LOW_THRESHOLD - 0.001, _ENOUGH) == (False, "high")


def test_is_same_boundary_is_medium_threshold():
    v = _verifier()
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD - 0.001, _ENOUGH)[0] is False
    assert v._classify(DEFAULT_MEDIUM_THRESHOLD, _ENOUGH)[0] is True


def test_inlier_floor_vetoes_same():
    """A pair clearing the ratio band but with too few inliers is not 'same'."""
    v = _verifier()
    # Score well into the 'high/same' band, but inlier count below the floor.
    assert v._classify(DEFAULT_HIGH_THRESHOLD + 0.1, DEFAULT_MIN_INLIERS - 1) == (False, "low")
    # Same score, enough inliers -> stays same.
    assert v._classify(DEFAULT_HIGH_THRESHOLD + 0.1, DEFAULT_MIN_INLIERS) == (True, "high")


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


# --- pre-extracted features / FeatureStore -----------------------------------
def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_verify_features_matches_image_path():
    """Scoring cached (compact) features gives the same result as scoring images."""
    v = _verifier()
    query = _spot_image(_SPOTS)
    candidates = [_spot_image(_OTHER_SPOTS), _spot_image(_SPOTS, offset=(5, 5))]
    from_images = v.verify_against_many(query, candidates)

    sift = cv2.SIFT_create()
    feats = [_to_compact(extract_sift_features(img, sift)) for img in [query, *candidates]]
    assert v.verify_features_against_many(feats[0], feats[1:]) == from_images

    # A candidate that couldn't be loaded scores 0 and is never "same".
    empty = v.verify_features_against_many(feats[0], [EMPTY_FEATURES])[0]
    assert (empty["score"], empty["is_same"], empty["inliers"]) == (0.0, False, 0)


def test_compact_descriptors_are_lossless():
    """OpenCV SIFT descriptors are integer-valued, so uint8 storage loses nothing."""
    _, des = extract_sift_features(_spot_image(_SPOTS), cv2.SIFT_create())
    _, compact = _to_compact((np.zeros((0, 2)), des))
    assert compact.dtype == np.uint8
    assert np.array_equal(compact.astype(np.float32), des)


def test_feature_store_url_allowlist():
    store = FeatureStore(allowed_hosts=(".r2.dev",))
    assert store.is_allowed("https://pub-abc.r2.dev/x.jpg")
    assert not store.is_allowed("http://pub-abc.r2.dev/x.jpg")  # https only
    assert not store.is_allowed("https://evil.example.com/x.jpg")
    assert not store.is_allowed("https://r2.dev.evil.com/x.jpg")
    assert not store.is_allowed("https://169.254.169.254/latest/meta-data")


def test_feature_store_caches_and_skips_failures(monkeypatch):
    """Each URL is downloaded once; failed downloads return None and aren't cached."""
    png = _png_bytes(_spot_image(_SPOTS))
    calls: list[str] = []

    def fake_download(self, url):
        calls.append(url)
        if "missing" in url:
            raise OSError("404")
        return png

    monkeypatch.setattr(FeatureStore, "_download", fake_download)
    store = FeatureStore()
    urls = ["https://a.r2.dev/1.png", "https://a.r2.dev/missing.png"]

    first = store.get_many(urls)
    assert first[0] is not None and first[1] is None
    second = store.get_many(urls)
    assert second[0] is not None
    # 1.png fetched once (then cached), missing.png retried
    assert calls.count(urls[0]) == 1 and calls.count(urls[1]) == 2
    assert len(store) == 1


# --- bucket persistence ------------------------------------------------------
class _NoSuchKey(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class _FakeS3:
    """Minimal in-memory stand-in for a boto3 S3 client."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.puts: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 (boto3 casing)
        if Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body
        self.puts.append(Key)


def _same_features(a, b) -> bool:
    if a[1] is None or b[1] is None:
        return a[1] is None and b[1] is None and len(a[0]) == len(b[0])
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_shard_encoding_roundtrip():
    sift = cv2.SIFT_create()
    one = _to_compact(extract_sift_features(_spot_image(_SPOTS), sift))
    two = _to_compact(extract_sift_features(_spot_image(_OTHER_SPOTS), sift))
    entries = {
        "https://a.r2.dev/1.png": PhotoFeatures(one, two),
        "https://a.r2.dev/blank.png": EMPTY_PHOTO,
        "https://a.r2.dev/2.png": PhotoFeatures(two, EMPTY_FEATURES),
    }
    decoded = decode_shard(encode_shard(entries))
    assert list(decoded) == list(entries)
    for url, photo in entries.items():
        assert _same_features(decoded[url].base, photo.base)
        assert _same_features(decoded[url].strip, photo.strip)
    assert decode_shard(encode_shard({})) == {}


def test_features_survive_a_restart_via_bucket(monkeypatch):
    """A new process reloads features from the bucket instead of re-downloading."""
    png = _png_bytes(_spot_image(_SPOTS))
    downloads: list[str] = []

    def fake_download(self, url):
        downloads.append(url)
        return png

    monkeypatch.setattr(FeatureStore, "_download", fake_download)
    s3 = _FakeS3()
    urls = [f"https://a.r2.dev/{i}.png" for i in range(5)]

    first = FeatureStore(persistence=BucketPersistence(s3, "bucket"))
    before = first.get_many(urls)
    first.flush()
    assert len(downloads) == 5
    # only the shards that hold the new URLs were written
    assert len(s3.puts) == len({shard_of(u) for u in urls})

    restarted = FeatureStore(persistence=BucketPersistence(s3, "bucket"))
    assert restarted.wait_until_loaded(5)
    after = restarted.get_many(urls)
    assert len(downloads) == 5  # nothing re-downloaded
    for a, b in zip(before, after, strict=True):
        assert _same_features(a.base, b.base) and _same_features(a.strip, b.strip)


def test_new_photo_rewrites_its_shard_without_losing_others(monkeypatch):
    png = _png_bytes(_spot_image(_SPOTS))
    monkeypatch.setattr(FeatureStore, "_download", lambda self, url: png)
    s3 = _FakeS3()
    store = FeatureStore(persistence=BucketPersistence(s3, "bucket"))
    urls = [f"https://a.r2.dev/{i}.png" for i in range(40)]
    store.get_many(urls)
    store.get_many(["https://a.r2.dev/new.png"])
    store.flush()

    restarted = FeatureStore(persistence=BucketPersistence(s3, "bucket"))
    restarted.wait_until_loaded(5)
    assert len(restarted) == 41


def test_broken_bucket_degrades_to_memory_only(monkeypatch):
    class _Broken:
        def get_object(self, **_):
            raise ConnectionError("bucket down")

        def put_object(self, **_):
            raise ConnectionError("bucket down")

    png = _png_bytes(_spot_image(_SPOTS))
    monkeypatch.setattr(FeatureStore, "_download", lambda self, url: png)
    store = FeatureStore(persistence=BucketPersistence(_Broken(), "bucket"))
    assert store.wait_until_loaded(5)
    assert store.get_many(["https://a.r2.dev/1.png"])[0] is not None
    store.flush()  # failed write is logged, not raised


def test_bucket_persistence_from_env_disabled_without_bucket(monkeypatch):
    for name in ("BUCKET", "AWS_S3_BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)
    assert BucketPersistence.from_env() is None


# --- straightening -----------------------------------------------------------
def _salamander(spots, bend: float, size=(260, 420)) -> Image.Image:
    """Synthetic salamander on grey: black body tapering to the tail, yellow spots
    placed in body coordinates (arc position, side offset), so the same `spots`
    drawn with a different `bend` is the same individual in another pose."""
    w, h = size
    img = Image.new("RGB", size, (150, 150, 150))
    draw = ImageDraw.Draw(img)
    t = np.linspace(0, 1, 200)
    x = w / 2 + bend * w * 0.35 * np.sin(np.pi * t)
    y = 20 + t * (h - 40)
    pts = np.stack([x, y], axis=1)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    half = 26 * (1 - 0.8 * t) + 4  # head (top) wider than the tail
    for i in range(len(t)):
        cx, cy = pts[i]
        draw.ellipse([cx - half[i], cy - half[i], cx + half[i], cy + half[i]], fill=(15, 15, 15))
    for ts, side, r in spots:
        i = int(ts * (len(t) - 1))
        cx, cy = pts[i] + side * half[i] * normal[i]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(230, 200, 20))
    return img


_SAL_SPOTS = [
    (0.08, 0.0, 9), (0.18, -0.5, 6), (0.22, 0.5, 7), (0.3, -0.2, 5), (0.36, 0.45, 6),
    (0.42, -0.5, 7), (0.5, 0.1, 5), (0.55, 0.5, 6), (0.6, -0.4, 5), (0.27, 0.15, 4),
]  # fmt: skip
_SAL_OTHER = [
    (0.1, 0.4, 7), (0.15, -0.3, 8), (0.25, 0.0, 6), (0.33, 0.5, 5), (0.38, -0.5, 6),
    (0.46, 0.3, 7), (0.52, -0.2, 5), (0.58, 0.0, 6), (0.62, 0.45, 5), (0.2, 0.55, 4),
]  # fmt: skip


def test_straighten_gives_head_up_vertical_strip():
    strip, mask = straighten(_salamander(_SAL_SPOTS, bend=0.8))
    assert strip.shape[0] > strip.shape[1]  # tall, narrow strip
    rows = mask.sum(axis=1)
    body_rows = np.nonzero(rows)[0]
    first, last = body_rows[0], body_rows[-1]
    third = (last - first) // 3
    # the wider (head) end is at the top
    assert rows[first : first + third].mean() > rows[last - third : last].mean()


def test_strip_ranks_same_individual_across_poses_above_another():
    v = _verifier()
    sift = cv2.SIFT_create()
    query = strip_features(_salamander(_SAL_SPOTS, bend=0.0), sift)
    same_bent = strip_features(_salamander(_SAL_SPOTS, bend=0.9), sift)
    other_bent = strip_features(_salamander(_SAL_OTHER, bend=0.9), sift)
    results = v.verify_features_against_many(
        EMPTY_FEATURES,
        [EMPTY_FEATURES, EMPTY_FEATURES],
        query_strip=query,
        candidate_strips=[same_bent, other_bent],
    )
    by_index = {r["candidate_index"]: r for r in results}
    assert by_index[0]["strip_inliers"] > by_index[1]["strip_inliers"]
    # strips never change the automatic decision (base features are empty here)
    assert not any(r["is_same"] for r in results)


def test_strip_features_on_blank_image_is_empty():
    blank = Image.new("RGB", (200, 300), (150, 150, 150))
    pts, des = strip_features(blank, cv2.SIFT_create())
    assert len(pts) == 0 and des is None


def test_straighten_failure_keeps_base_features(monkeypatch):
    """A crash while straightening must not lose the photo (its is_same features)."""
    import app.identification.straighten as straighten_module

    def boom(image):
        raise cv2.error("simulated")

    monkeypatch.setattr(straighten_module, "straighten", boom)
    png = _png_bytes(_spot_image(_SPOTS))
    monkeypatch.setattr(FeatureStore, "_download", lambda self, url: png)
    photo = FeatureStore().get_many(["https://a.r2.dev/1.png"])[0]
    assert photo is not None
    assert len(photo.base[0]) > 0
    assert len(photo.strip[0]) == 0


def test_strip_inliers_absent_without_strips():
    v = _verifier()
    feats = extract_sift_features(_spot_image(_SPOTS), cv2.SIFT_create())
    result = v.verify_features_against_many(feats, [feats])[0]
    assert "strip_inliers" not in result
