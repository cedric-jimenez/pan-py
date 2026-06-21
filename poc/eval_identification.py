"""Evaluation harness for salamander individual identification.

Measures how well the patch-matching verifier separates same-individual pairs
from different-individual pairs, using the labeled dataset in docs/images/<idN>/.

Run from the repo root:
    ./venv/bin/python poc/eval_identification.py
    ./venv/bin/python poc/eval_identification.py --methods v0 fg_mask

Each "method" is a scoring function (patches1, patches2, meta1, meta2) -> float.
This lets us compare algorithm variants on identical features (the DINOv2
forward pass is run once per image and cached).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Make `app` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.identification.embedder import SalamanderEmbedder  # noqa: E402

DATASET_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
PATCH_GRID = 16  # 224 / 14


@dataclass
class ImageFeatures:
    """Cached features for one image."""

    label: str  # individual id, e.g. "id1"
    path: Path
    embedding: np.ndarray  # (D,) L2-normalized GeM embedding
    patches: np.ndarray  # (N, D) L2-normalized patch tokens
    fg_mask: np.ndarray  # (N,) bool, True = foreground (salamander body)


# ---------------------------------------------------------------------------
# Foreground detection
# ---------------------------------------------------------------------------
def compute_fg_mask(image: Image.Image, grid: int = PATCH_GRID) -> np.ndarray:
    """Per-patch foreground mask from the raw crop.

    Background in these crops is light (white/grey); the salamander is dark
    (black) with yellow spots. A patch is foreground if it is not uniformly
    light. Returns a flat (grid*grid,) bool array matching DINOv2 patch order
    (row-major), after the same resize-pad-to-square the embedder applies.
    """
    # Replicate _ResizePad geometry so mask aligns with patch tokens.
    rgb = image.convert("RGB")
    w, h = rgb.size
    size = 224
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = rgb.resize((new_w, new_h))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    arr = np.asarray(canvas).astype(np.float32)  # (224,224,3)
    cell = size // grid
    mask = np.zeros(grid * grid, dtype=bool)
    for gy in range(grid):
        for gx in range(grid):
            block = arr[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            r, g, b = block[..., 0].mean(), block[..., 1].mean(), block[..., 2].mean()
            brightness = (r + g + b) / 3.0
            saturation = max(r, g, b) - min(r, g, b)
            # Background is achromatic (r≈g≈b): white, grey-150 segmenter fill,
            # or near-black resize-pad. Foreground keeps the salamander's
            # textured dark body and its saturated yellow spots.
            is_grey_bg = saturation < 28 and brightness > 110
            is_pad = saturation < 28 and brightness < 18
            mask[gy * grid + gx] = not (is_grey_bg or is_pad)
    return mask


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------
def _mutual_nn(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of mutual nearest-neighbor patch pairs. Returns (rows, cols)."""
    nn_1to2 = sim.argmax(axis=1)
    nn_2to1 = sim.argmax(axis=0)
    rows = np.arange(sim.shape[0])
    mutual = nn_2to1[nn_1to2] == rows
    return rows[mutual], nn_1to2[mutual]


def score_v0(f1: ImageFeatures, f2: ImageFeatures) -> float:
    """Current production algorithm: match_ratio * mean_similarity."""
    sim = f1.patches @ f2.patches.T
    n = sim.shape[0]
    if n == 0:
        return 0.0
    rows, cols = _mutual_nn(sim)
    if rows.size == 0:
        return 0.0
    return float(rows.size) / n * float(sim[rows, cols].mean())


def score_fg_mask(f1: ImageFeatures, f2: ImageFeatures) -> float:
    """Like v0 but only match foreground (body) patches, ignoring background."""
    p1 = f1.patches[f1.fg_mask]
    p2 = f2.patches[f2.fg_mask]
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return 0.0
    sim = p1 @ p2.T
    rows, cols = _mutual_nn(sim)
    if rows.size == 0:
        return 0.0
    return float(rows.size) / p1.shape[0] * float(sim[rows, cols].mean())


def score_fg_topk(f1: ImageFeatures, f2: ImageFeatures, k: int = 30) -> float:
    """Foreground mutual NN, scored by the mean of the top-k match similarities.

    The most distinctive spot correspondences should drive the decision rather
    than the bulk of mediocre matches.
    """
    p1 = f1.patches[f1.fg_mask]
    p2 = f2.patches[f2.fg_mask]
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return 0.0
    sim = p1 @ p2.T
    rows, cols = _mutual_nn(sim)
    if rows.size == 0:
        return 0.0
    sims = np.sort(sim[rows, cols])[::-1][:k]
    coverage = min(1.0, rows.size / k)
    return float(sims.mean()) * coverage


def score_fg_sym(f1: ImageFeatures, f2: ImageFeatures) -> float:
    """Foreground mutual NN with symmetric coverage normalization.

    match_ratio normalized by the smaller foreground set (so a large image
    can't dilute the ratio), times mean mutual similarity.
    """
    p1 = f1.patches[f1.fg_mask]
    p2 = f2.patches[f2.fg_mask]
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return 0.0
    sim = p1 @ p2.T
    rows, cols = _mutual_nn(sim)
    if rows.size == 0:
        return 0.0
    denom = min(p1.shape[0], p2.shape[0])
    return float(rows.size) / denom * float(sim[rows, cols].mean())


def score_fg_strict(f1: ImageFeatures, f2: ImageFeatures) -> float:
    """Foreground mutual NN counting only high-confidence correspondences.

    Mutual matches below a similarity floor are discarded before scoring, so
    generic body patches that weakly match don't inflate different-individual
    scores.
    """
    floor = 0.6
    p1 = f1.patches[f1.fg_mask]
    p2 = f2.patches[f2.fg_mask]
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return 0.0
    sim = p1 @ p2.T
    rows, cols = _mutual_nn(sim)
    if rows.size == 0:
        return 0.0
    sims = sim[rows, cols]
    keep = sims >= floor
    if not keep.any():
        return 0.0
    denom = min(p1.shape[0], p2.shape[0])
    return float(keep.sum()) / denom * float(sims[keep].mean())


def score_cosine(f1: ImageFeatures, f2: ImageFeatures) -> float:
    """Global GeM-embedding cosine similarity (baseline reference)."""
    return float(np.dot(f1.embedding, f2.embedding))


METHODS = {
    "v0": score_v0,
    "fg_mask": score_fg_mask,
    "fg_sym": score_fg_sym,
    "fg_strict": score_fg_strict,
    "fg_topk": score_fg_topk,
    "cosine": score_cosine,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def best_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing accuracy over the pooled scores. Returns (thr, acc)."""
    scores = np.concatenate([pos, neg])
    best_acc, best_thr = 0.0, 0.0
    for thr in np.unique(scores):
        tp = (pos >= thr).sum()
        tn = (neg < thr).sum()
        acc = (tp + tn) / len(scores)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, best_acc


def auc_roc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive outscores a random negative (Mann-Whitney)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return wins / (pos.size * neg.size)


def report(name: str, pairs: list, scores: np.ndarray) -> None:
    labels = np.array([same for same, _, _ in pairs])
    pos = scores[labels]
    neg = scores[~labels]
    thr, acc = best_threshold(pos, neg)
    auc = auc_roc(pos, neg)
    # FP rate at the production high-decision boundary (>=0.15 => "same").
    prod_thr = 0.15
    fp_prod = (neg >= prod_thr).sum()
    print(f"\n=== {name} ===")
    print(f"  same  pairs (n={pos.size}): min={pos.min():.3f} mean={pos.mean():.3f} max={pos.max():.3f}")
    print(f"  diff  pairs (n={neg.size}): min={neg.min():.3f} mean={neg.mean():.3f} max={neg.max():.3f}")
    gap = pos.mean() - neg.mean()
    print(f"  mean gap (same-diff): {gap:+.3f}")
    print(f"  AUC-ROC: {auc:.3f}")
    print(f"  best-threshold accuracy: {acc:.1%} @ thr={thr:.3f}")
    print(f"  false positives at prod thr {prod_thr}: {fp_prod}/{neg.size} "
          f"({fp_prod / neg.size:.1%})")
    # Worst false positives (different individuals scoring high).
    diff_pairs = [(s, p[1], p[2]) for s, p in zip(scores[~labels], [p for p in pairs if not p[0]])]
    diff_pairs.sort(reverse=True)
    print("  top diff-pair scores (potential false positives):")
    for s, a, b in diff_pairs[:5]:
        print(f"    {s:.3f}  {a}  vs  {b}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    args = ap.parse_args()

    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = sorted(p for ext in exts for p in DATASET_DIR.glob(f"*/{ext}"))
    if not paths:
        sys.exit(f"No images found under {DATASET_DIR}")
    per_individual = {}
    for p in paths:
        per_individual.setdefault(p.parent.name, 0)
        per_individual[p.parent.name] += 1
    singletons = [k for k, v in per_individual.items() if v < 2]
    if singletons:
        print(f"Note: individuals with <2 photos generate no same-pairs: {singletons}")
    print(f"Loading DINOv2 and embedding {len(paths)} images...")

    embedder = SalamanderEmbedder()
    embedder.load_model()

    feats: list[ImageFeatures] = []
    for p in paths:
        img = Image.open(p)
        # Use the embedder's canonical foreground mask so eval == production.
        emb, patches, fg = embedder.extract_features(img)
        feats.append(ImageFeatures(p.parent.name, p, emb, patches, fg))
    print(f"Embedded {len(feats)} images across {len({f.label for f in feats})} individuals.")
    fg_frac = np.mean([f.fg_mask.mean() for f in feats])
    print(f"Mean foreground patch fraction: {fg_frac:.1%}")

    # All unordered pairs with same/different labels.
    pairs = []  # (is_same, name_a, name_b) with index alignment to feats combos
    combos = list(itertools.combinations(range(len(feats)), 2))
    for i, j in combos:
        same = feats[i].label == feats[j].label
        pairs.append((same, f"{feats[i].label}/{feats[i].path.stem[:8]}",
                      f"{feats[j].label}/{feats[j].path.stem[:8]}"))
    n_same = sum(1 for s, _, _ in pairs if s)
    print(f"{len(pairs)} pairs total: {n_same} same, {len(pairs) - n_same} different")

    for m in args.methods:
        fn = METHODS[m]
        scores = np.array([fn(feats[i], feats[j]) for i, j in combos])
        report(m, pairs, scores)


if __name__ == "__main__":
    main()
