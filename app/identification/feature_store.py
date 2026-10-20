"""In-memory cache of SIFT features for images referenced by URL.

The /verify-urls endpoint compares a query against *every* photo of a user, not
just a short cosine-ranked list. Re-downloading and re-analysing each photo on
every search would be far too slow, so each image URL is fetched and analysed
once, then its SIFT features are kept here. Pan's segmented-image URLs are
immutable (a new upload gets a new UUID URL), so a cached entry never goes stale.

The cache lives in process memory: it is rebuilt lazily after a restart (the
first search after a deploy is slower, later ones only pay for matching).
"""

import io
import logging
import os
import threading
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from app.identification.verifier import SiftFeatures, extract_sift_features

logger = logging.getLogger(__name__)

# ~200 keypoints x 128 uint8 bytes ≈ 25 KB per image → 10k entries ≈ 250 MB worst case.
DEFAULT_MAX_ENTRIES = int(os.getenv("FEATURE_CACHE_MAX_ENTRIES", "10000"))
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


def _to_compact(features: SiftFeatures) -> SiftFeatures:
    """Store descriptors as uint8: OpenCV SIFT values are integers in 0..255."""
    pts, des = features
    if des is not None:
        des = np.clip(np.rint(des), 0, 255).astype(np.uint8)
    return pts, des


class FeatureStore:
    """Thread-safe LRU cache of SIFT features keyed by image URL."""

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
    ) -> None:
        self.max_entries = max_entries
        self.allowed_hosts = allowed_hosts
        self._cache: OrderedDict[str, SiftFeatures] = OrderedDict()
        self._lock = threading.Lock()
        self._local = threading.local()  # one SIFT detector per worker thread

    def __len__(self) -> int:
        return len(self._cache)

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

    def _load(self, url: str) -> SiftFeatures | None:
        """Download + analyse one image. None if it can't be fetched/decoded."""
        try:
            image = Image.open(io.BytesIO(self._download(url)))
            return _to_compact(extract_sift_features(image, self._sift()))
        except Exception as e:  # network error, 403/404, corrupt image…
            logger.warning(f"feature_store: could not load {url}: {e}")
            return None

    def get_many(self, urls: list[str]) -> list[SiftFeatures | None]:
        """Return features for each URL (None where loading failed), in order."""
        found: dict[str, SiftFeatures] = {}
        with self._lock:
            for url in urls:
                if url in self._cache:
                    self._cache.move_to_end(url)
                    found[url] = self._cache[url]
        missing = list(dict.fromkeys(u for u in urls if u not in found))

        if missing:
            with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
                loaded = list(pool.map(self._load, missing))
            with self._lock:
                for url, features in zip(missing, loaded, strict=True):
                    if features is None:
                        continue  # don't cache failures: retry on the next search
                    found[url] = features
                    self._cache[url] = features
                    self._cache.move_to_end(url)
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)

        logger.info(
            f"feature_store: {len(urls)} requested, {len(urls) - len(missing)} cached, "
            f"{len(missing)} fetched, cache size={len(self._cache)}"
        )
        return [found.get(url) for url in urls]
