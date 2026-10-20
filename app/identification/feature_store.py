"""Cache of SIFT features for images referenced by URL, persisted to a bucket.

The /verify-urls endpoint compares a query against *every* photo of a user, not
just a short cosine-ranked list. Re-downloading and re-analysing each photo on
every search would be far too slow, so each image URL is fetched and analysed
once, then its SIFT features are kept in memory. Pan's segmented-image URLs are
immutable (a new upload gets a new UUID URL), so an entry never goes stale.

pan-py runs on Railway with app sleeping, so the process (and its memory) is
regularly stopped. When an S3-compatible bucket is configured (Railway bucket,
see BucketPersistence.from_env), the features are also saved there in a fixed
number of shard files: on start-up all shards are read back (a handful of
requests), and after a search only the shards that gained new photos are
rewritten. Without a bucket the cache is memory-only.
"""

import hashlib
import io
import logging
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from app.identification.straighten import PX_PER_UNIT, strip_features
from app.identification.verifier import (
    _IMAGE_SIZE,
    EMPTY_FEATURES,
    SiftFeatures,
    extract_sift_features,
)

logger = logging.getLogger(__name__)


class PhotoFeatures(NamedTuple):
    """Everything cached for one photo."""

    base: SiftFeatures  # prod SIFT on the resize-padded image (drives is_same)
    strip: SiftFeatures  # SIFT on the straightened strip (ranks candidates to verify)


EMPTY_PHOTO = PhotoFeatures(EMPTY_FEATURES, EMPTY_FEATURES)

# Only fetch from these hosts (suffix match) to avoid turning the service into an
# open proxy. Pan stores images on Cloudflare R2 public buckets (*.r2.dev).
DEFAULT_ALLOWED_HOSTS = tuple(
    h.strip().lower()
    for h in os.getenv("VERIFY_URL_ALLOWED_HOSTS", ".r2.dev").split(",")
    if h.strip()
)
_FETCH_WORKERS = 16
_FETCH_TIMEOUT_S = 15
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
# R2 rejects the default "Python-urllib" user agent with 403.
_USER_AGENT = "pan-py/feature-store"

# Persisted shards. Bump the version whenever extraction changes (SIFT params,
# resize geometry…) so stale features are never reused.
N_SHARDS = 64
_SHARD_PREFIX = f"sift-features/v2-{_IMAGE_SIZE}px-strip{PX_PER_UNIT}/"
# How long a search waits for the start-up load from the bucket.
_LOAD_WAIT_S = 30


def _to_compact(features: SiftFeatures) -> SiftFeatures:
    """Store descriptors as uint8: OpenCV SIFT values are integers in 0..255."""
    pts, des = features
    if des is not None:
        des = np.clip(np.rint(des), 0, 255).astype(np.uint8)
    return pts, des


def shard_of(url: str) -> int:
    """Stable shard index for a URL (independent of Python's hash seed)."""
    return int.from_bytes(hashlib.sha256(url.encode()).digest()[:4], "big") % N_SHARDS


def _pack(features: list[SiftFeatures]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.array([len(f[0]) for f in features], dtype=np.int32)
    pts = [f[0].astype(np.float32).reshape(-1, 2) for f in features]
    empty = np.zeros((0, 128), dtype=np.uint8)
    des = [d.astype(np.uint8) if (d := f[1]) is not None else empty for f in features]
    return (
        counts,
        np.concatenate(pts) if pts else np.zeros((0, 2), dtype=np.float32),
        np.concatenate(des) if des else empty,
    )


def _unpack(counts: np.ndarray, pts: np.ndarray, des: np.ndarray) -> list[SiftFeatures]:
    out: list[SiftFeatures] = []
    start = 0
    for n in counts.tolist():
        end = start + n
        out.append((pts[start:end], des[start:end] if n else None))
        start = end
    return out


def encode_shard(entries: dict[str, PhotoFeatures]) -> bytes:
    """Pack {url: PhotoFeatures} into one compressed .npz (no pickle)."""
    urls = list(entries)
    counts, pts, des = _pack([entries[u].base for u in urls])
    s_counts, s_pts, s_des = _pack([entries[u].strip for u in urls])
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        urls=np.frombuffer("\n".join(urls).encode(), dtype=np.uint8),
        counts=counts,
        pts=pts,
        des=des,
        strip_counts=s_counts,
        strip_pts=s_pts,
        strip_des=s_des,
    )
    return buf.getvalue()


def decode_shard(data: bytes) -> dict[str, PhotoFeatures]:
    """Inverse of encode_shard."""
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        raw_urls = npz["urls"].tobytes().decode()
        urls = raw_urls.split("\n") if raw_urls else []
        base = _unpack(npz["counts"], npz["pts"], npz["des"])
        strip = _unpack(npz["strip_counts"], npz["strip_pts"], npz["strip_des"])
    return {url: PhotoFeatures(b, s) for url, b, s in zip(urls, base, strip, strict=True)}


class BucketPersistence:
    """Reads/writes feature shards in an S3-compatible bucket."""

    def __init__(self, client: Any, bucket: str, prefix: str = _SHARD_PREFIX) -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    @classmethod
    def from_env(cls) -> "BucketPersistence | None":
        """Build from Railway bucket variables, or None when not configured.

        Uses Railway's names (BUCKET, ENDPOINT, ACCESS_KEY_ID, SECRET_ACCESS_KEY,
        REGION) and falls back to the AWS SDK names (AWS_S3_BUCKET_NAME,
        AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).
        BUCKET_URL_STYLE = virtual (default, Railway's default) or path.
        """
        env = os.environ
        bucket = env.get("BUCKET") or env.get("AWS_S3_BUCKET_NAME")
        if not bucket:
            return None
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=env.get("ENDPOINT") or env.get("AWS_ENDPOINT_URL"),
            aws_access_key_id=env.get("ACCESS_KEY_ID") or env.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=env.get("SECRET_ACCESS_KEY") or env.get("AWS_SECRET_ACCESS_KEY"),
            region_name=env.get("REGION") or env.get("AWS_REGION") or "auto",
            config=Config(
                s3={"addressing_style": env.get("BUCKET_URL_STYLE", "virtual")},
                retries={"max_attempts": 3},
                # Shards are loaded with 16 threads; the default pool keeps 10.
                max_pool_connections=16,
                connect_timeout=5,
                read_timeout=30,
            ),
        )
        return cls(client, bucket)

    def _key(self, shard: int) -> str:
        return f"{self.prefix}shard-{shard:02d}.npz"

    def load_shard(self, shard: int) -> dict[str, PhotoFeatures]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(shard))
        except Exception as e:
            # A shard that was never written: S3 says NoSuchKey, some providers 404.
            code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                return {}
            raise
        return decode_shard(obj["Body"].read())

    def save_shard(self, shard: int, entries: dict[str, PhotoFeatures]) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(shard), Body=encode_shard(entries))


class FeatureStore:
    """Thread-safe cache of SIFT features keyed by image URL.

    Holds every analysed photo in memory (~40 KB each with the straightened
    strip: ~400 MB for 10k photos), without eviction so the persisted shards
    always stay complete.
    """

    def __init__(
        self,
        allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
        persistence: BucketPersistence | None = None,
    ) -> None:
        self.allowed_hosts = allowed_hosts
        self.persistence = persistence
        self._cache: dict[str, PhotoFeatures] = {}
        self._lock = threading.Lock()
        self._local = threading.local()  # one SIFT detector per worker thread
        # Shard writes run one at a time in the background, off the request path.
        self._writer = ThreadPoolExecutor(max_workers=1) if persistence else None
        self._loaded = threading.Event()
        if persistence is None:
            self._loaded.set()
        else:
            threading.Thread(target=self._load_from_bucket, daemon=True).start()

    def __len__(self) -> int:
        return len(self._cache)

    def _load_from_bucket(self) -> None:
        """Read every shard back into memory (start-up, background thread)."""
        assert self.persistence is not None
        persistence = self.persistence
        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                shards = list(pool.map(persistence.load_shard, range(N_SHARDS)))
            with self._lock:
                for entries in shards:
                    for url, features in entries.items():
                        self._cache.setdefault(url, features)
            logger.info(f"feature_store: loaded {len(self._cache)} entries from bucket")
        except Exception as e:  # bucket down/misconfigured: degrade to memory-only
            logger.warning(f"feature_store: could not load from bucket: {e}")
        finally:
            self._loaded.set()

    def wait_until_loaded(self, timeout: float | None = None) -> bool:
        return self._loaded.wait(timeout)

    def flush(self) -> None:
        """Wait for pending bucket writes (tests / shutdown)."""
        if self._writer is not None:
            self._writer.submit(lambda: None).result()

    def is_allowed(self, url: str) -> bool:
        """Only https URLs whose host ends with an allowed suffix."""
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return parsed.scheme == "https" and any(
            host == h.lstrip(".") or host.endswith(h if h.startswith(".") else "." + h)
            for h in self.allowed_hosts
        )

    def _sift(self) -> Any:
        sift = getattr(self._local, "sift", None)
        if sift is None:
            sift = cv2.SIFT_create()  # type: ignore[attr-defined]
            self._local.sift = sift
        return sift

    def _download(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:  # noqa: S310
            data: bytes = response.read(_MAX_IMAGE_BYTES + 1)
        if len(data) > _MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return data

    def _load(self, url: str) -> PhotoFeatures | None:
        """Download + analyse one image. None if it can't be fetched/decoded."""
        try:
            image = Image.open(io.BytesIO(self._download(url)))
            image.load()
            sift = self._sift()
            base = _to_compact(extract_sift_features(image, sift))
            # strip_features never raises: a straightening failure only empties
            # the strip, the base features (is_same) are kept.
            return PhotoFeatures(base, _to_compact(strip_features(image, sift)))
        except Exception as e:  # network error, 403/404, corrupt image…
            logger.warning(f"feature_store: could not load {url}: {e}")
            return None

    def _persist(self, shards: set[int]) -> None:
        """Rewrite the given shards from the current cache (background)."""
        assert self.persistence is not None
        snapshots: dict[int, dict[str, PhotoFeatures]] = {s: {} for s in shards}
        with self._lock:
            for url, features in self._cache.items():
                shard = shard_of(url)
                if shard in snapshots:
                    snapshots[shard][url] = features
        for shard, entries in snapshots.items():
            try:
                self.persistence.save_shard(shard, entries)
            except Exception as e:  # keep serving; the shard is rewritten next time
                logger.warning(f"feature_store: could not save shard {shard}: {e}")

    def get_many(self, urls: list[str]) -> list[PhotoFeatures | None]:
        """Return features for each URL (None where loading failed), in order."""
        if not self.wait_until_loaded(_LOAD_WAIT_S):
            logger.warning("feature_store: bucket load still running, continuing without it")

        with self._lock:
            found = {u: self._cache[u] for u in urls if u in self._cache}
        missing = list(dict.fromkeys(u for u in urls if u not in found))

        if missing:
            with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
                loaded = list(pool.map(self._load, missing))
            added = {u: f for u, f in zip(missing, loaded, strict=True) if f is not None}
            with self._lock:
                self._cache.update(added)  # failures aren't cached: retried next search
            found.update(added)
            if added and self._writer is not None:
                self._writer.submit(self._persist, {shard_of(u) for u in added})

        logger.info(
            f"feature_store: {len(urls)} requested, {len(urls) - len(missing)} cached, "
            f"{len(missing)} fetched, cache size={len(self._cache)}"
        )
        return [found.get(url) for url in urls]
