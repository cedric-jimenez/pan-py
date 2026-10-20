"""Straighten ('unbend') a segmented salamander along its midline.

A salamander's body curves differently on every photo, while the RANSAC check in
the verifier assumes one rigid transform for the whole animal. Re-sampling the
skin perpendicular to the trunk midline gives a straight, head-up strip at a
scale normalised on the trunk width, so the same individual photographed in two
poses (e.g. a year apart) lines up much better.

Measured on the labelled set (16 queries vs 172 candidates): it does NOT lift
long-gap pairs over the is_same inlier floor, but it ranks the true match much
higher (top-5: 9 -> 12; Titine 2024 for a Nov-2025 query: rank 59 -> 3). Pan uses
it to rank "candidates to verify" for a human, never for the automatic is_same.

Input: the segmented image from /segment-salamander (animal on uniform grey 150).
"""

import logging
from typing import Any

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from app.identification.verifier import EMPTY_FEATURES, SiftFeatures

logger = logging.getLogger(__name__)

_GREY = 150
PX_PER_UNIT = 24  # strip pixels per trunk half-width
LENGTH_UNITS = 14.0  # head tip -> this many half-widths down the midline (head + trunk)
WIDTH_FACTOR = 1.3  # strip half-width, in trunk half-widths


def body_mask(rgb: np.ndarray) -> np.ndarray:
    """Largest non-grey blob (the salamander), holes filled."""
    a = rgb.astype(np.int16)
    m = (np.abs(a - _GREY).max(axis=2) > 25).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    filled: np.ndarray = ndimage.binary_fill_holes(lab == biggest)
    return filled


def _distance_to_edge(mask: np.ndarray) -> np.ndarray:
    """Distance of each body pixel to the silhouette edge.

    The image border counts as edge: without the padding, a mask with no
    background pixel (e.g. a non-grey image) gets 'infinite' distances.
    """
    padded = np.pad(mask.astype(np.uint8), 1)
    dt: np.ndarray = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return dt


def _midline(mask: np.ndarray) -> np.ndarray:
    """Longest path through the trunk skeleton, as (N, 2) x,y points."""
    # Remove thin parts (legs, toes, tail tip) with an opening sized on the body
    # width, so the skeleton follows the trunk instead of running into a leg
    # (e.g. when the head is cut by the image border).
    dt = _distance_to_edge(mask)
    r = max(2, int(0.55 * dt.max()))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    trunk = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
    if trunk.sum() < 0.3 * mask.sum():
        trunk = mask
    smooth = cv2.GaussianBlur(trunk.astype(np.float32), (0, 0), 2) > 0.5
    ys, xs = np.nonzero(skeletonize(smooth))
    if len(xs) < 10:
        raise ValueError("skeleton too small")
    index = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs, strict=True))}
    graph = nx.Graph()
    for (y, x), i in index.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = index.get((y + dy, x + dx))
                if j is not None and j != i:
                    graph.add_edge(i, j, weight=float(np.hypot(dy, dx)))
    if graph.number_of_nodes() == 0:
        raise ValueError("empty skeleton graph")
    graph = graph.subgraph(max(nx.connected_components(graph), key=len))
    # tree diameter: farthest node from any node, then farthest from that one
    first = next(iter(graph.nodes))
    dist = nx.single_source_dijkstra_path_length(graph, first)
    end_a = max(dist, key=lambda n: dist[n])
    dist, paths = nx.single_source_dijkstra(graph, end_a)
    end_b = max(dist, key=lambda n: dist[n])
    return np.array([[xs[i], ys[i]] for i in paths[end_b]], dtype=np.float64)


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def _extend_to_tips(pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The skeleton stops short of the snout/tail: walk on along the end tangents."""
    for at_start in (True, False):
        p, q = (pts[0], pts[1]) if at_start else (pts[-1], pts[-2])
        step = (p - q) / (np.linalg.norm(p - q) + 1e-9)
        extra, cur = [], p.copy()
        while True:
            cur = cur + step
            x, y = int(round(cur[0])), int(round(cur[1]))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or not mask[y, x]:
                break
            extra.append(cur.copy())
        if extra:
            pts = np.vstack([extra[::-1], pts]) if at_start else np.vstack([pts, extra])
    return pts


def straighten(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return (straightened RGB strip, its body mask), head at the top.

    Raises ValueError when no usable midline is found.
    """
    rgb = np.asarray(image.convert("RGB"))
    mask = body_mask(rgb)
    if not mask.any():
        raise ValueError("no salamander found")
    dt = _distance_to_edge(mask)
    pts = _resample(_midline(mask), 300)
    pts = ndimage.gaussian_filter1d(pts, sigma=6, axis=0, mode="nearest")
    pts = _resample(_extend_to_tips(pts, mask), 400)

    xi = np.clip(pts[:, 0].round().astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(pts[:, 1].round().astype(int), 0, mask.shape[0] - 1)
    width = np.where(mask[yi, xi], dt[yi, xi], 0.0)
    quarter = len(pts) // 4
    if width[:quarter].mean() < width[-quarter:].mean():  # head = wider end
        pts, width = pts[::-1], width[::-1]

    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    trunk = (s > 0.15 * s[-1]) & (s < 0.45 * s[-1])
    unit = float(np.median(width[trunk])) if trunk.any() else float(width.max())
    if unit < 1:
        raise ValueError("body too thin")
    keep_len = min(s[-1], LENGTH_UNITS * unit)
    pts = pts[s <= keep_len]
    out_h = max(16, int(round(keep_len / unit * PX_PER_UNIT)))
    out_w = int(round(2 * WIDTH_FACTOR * PX_PER_UNIT))

    rows = _resample(pts, out_h)
    tangent = ndimage.gaussian_filter1d(np.gradient(rows, axis=0), sigma=2, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offsets = np.linspace(-WIDTH_FACTOR * unit, WIDTH_FACTOR * unit, out_w)
    map_x = (rows[:, 0:1] + offsets[None, :] * normal[:, 0:1]).astype(np.float32)
    map_y = (rows[:, 1:2] + offsets[None, :] * normal[:, 1:2]).astype(np.float32)
    strip = cv2.remap(
        rgb,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(_GREY,) * 3,
    )
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    strip_mask = (
        cv2.remap(
            mask_u8,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        > 127
    )
    strip[~strip_mask] = _GREY
    return strip, strip_mask


def strip_features(image: Image.Image, sift: Any) -> SiftFeatures:
    """SIFT features of the straightened strip, body interior only (no outline).

    Returns EMPTY_FEATURES when the salamander can't be straightened. Never
    raises: a failure here must not lose the photo's base features.
    """
    try:
        strip, strip_mask = straighten(image)
    except Exception as e:  # odd silhouettes, cv2 errors…: no strip, keep going
        logger.info(f"straighten: no strip ({type(e).__name__}: {e})")
        return EMPTY_FEATURES
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY), None)
    if not keypoints or descriptors is None:
        return EMPTY_FEATURES
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    k = max(3, PX_PER_UNIT // 4)
    inner = cv2.erode(strip_mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    x = np.clip(pts[:, 0].round().astype(int), 0, inner.shape[1] - 1)
    y = np.clip(pts[:, 1].round().astype(int), 0, inner.shape[0] - 1)
    keep = inner[y, x]
    return pts[keep], descriptors[keep]
