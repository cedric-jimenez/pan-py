"""Benchmark candidate similarity algorithms on the labelled set.

Compares, on docs/images/<idN>/, several scoring methods using two metrics:
  - top-1 retrieval: for each query, is the nearest *other* photo the same individual?
  - pairwise AUC-ROC: separability of same vs different pairs.

Methods:
  cosine      - DINOv2 GeM embedding cosine (current pgvector retrieval signal)
  fg_sym      - foreground-masked symmetric patch matching (current verifier)
  combined    - z(cosine) + z(fg_sym) ensemble
  patch_ransac- fg_sym matches filtered by RANSAC affine geometric consistency (A)
  sift_ransac - classic SIFT keypoints + RANSAC inliers (B)

Run: ./venv/bin/python poc/benchmark_methods.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.identification.embedder import SalamanderEmbedder  # noqa: E402
import eval_identification as E  # noqa: E402

GRID = 16


def load_features():
    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in E.DATASET_DIR.glob(f"*/{ext}")
    )
    emb = SalamanderEmbedder()
    emb.load_model()
    feats = []
    sift = cv2.SIFT_create()
    for p in paths:
        img = Image.open(p)
        e, patches, fg = emb.extract_features(img)
        # SIFT on greyscale of the resize-padded canvas (same geometry as patches).
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = 224 / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        canvas = Image.new("RGB", (224, 224), (0, 0, 0))
        canvas.paste(rgb.resize((nw, nh)), ((224 - nw) // 2, (224 - nh) // 2))
        gray = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
        feats.append(
            {
                "label": p.parent.name,
                "name": f"{p.parent.name}/{p.stem[:8]}",
                "emb": e,
                "patches": patches,
                "fg": fg,
                "sift_des": des,
                "sift_pts": pts,
            }
        )
    return feats


# --- scorers -----------------------------------------------------------------
def s_cosine(a, b):
    return float(np.dot(a["emb"], b["emb"]))


def _fg_mutual(a, b):
    """Return (orig_idx_a, orig_idx_b, sims) for mutual NN foreground matches."""
    ma, mb = a["fg"], b["fg"]
    ia = np.nonzero(ma)[0] if ma.any() else np.arange(len(a["patches"]))
    ib = np.nonzero(mb)[0] if mb.any() else np.arange(len(b["patches"]))
    pa, pb = a["patches"][ia], b["patches"][ib]
    sim = pa @ pb.T
    nn_ab = sim.argmax(1)
    nn_ba = sim.argmax(0)
    rows = np.arange(len(pa))
    mut = nn_ba[nn_ab] == rows
    return ia[rows[mut]], ib[nn_ab[mut]], sim[rows[mut], nn_ab[mut]], min(len(pa), len(pb))


def s_fg_sym(a, b):
    _, _, sims, denom = _fg_mutual(a, b)
    if sims.size == 0:
        return 0.0
    return len(sims) / denom * float(sims.mean())


def s_patch_ransac(a, b):
    """fg_sym matches kept only if geometrically consistent (RANSAC affine)."""
    ya, yb, sims, denom = _fg_mutual(a, b)
    if len(sims) < 3:
        return 0.0
    src = np.column_stack([ya % GRID, ya // GRID]).astype(np.float32)
    dst = np.column_stack([yb % GRID, yb // GRID]).astype(np.float32)
    _, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5
    )
    if inl is None:
        return 0.0
    inl = inl.ravel().astype(bool)
    if inl.sum() == 0:
        return 0.0
    return inl.sum() / denom * float(sims[inl].mean())


def s_sift_ransac(a, b):
    da, db = a["sift_des"], b["sift_des"]
    if da is None or db is None or len(da) < 4 or len(db) < 4:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0.0
    src = np.array([a["sift_pts"][m.queryIdx] for m in good], np.float32)
    dst = np.array([b["sift_pts"][m.trainIdx] for m in good], np.float32)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    inliers = int(mask.sum())
    denom = min(len(a["sift_pts"]), len(b["sift_pts"]))
    return inliers / denom if denom else 0.0


# --- metrics -----------------------------------------------------------------
def evaluate(feats, scorer):
    n = len(feats)
    S = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        S[i, j] = S[j, i] = scorer(feats[i], feats[j])
    labels = np.array([f["label"] for f in feats])
    # top-1
    hits = tot = 0
    for i in range(n):
        if (labels == labels[i]).sum() < 2:
            continue
        order = np.argsort(-S[i])
        order = order[order != i]
        hits += labels[order[0]] == labels[i]
        tot += 1
    top1 = hits / tot
    # AUC
    pos, neg = [], []
    for i, j in itertools.combinations(range(n), 2):
        (pos if labels[i] == labels[j] else neg).append(S[i, j])
    pos, neg = np.array(pos), np.array(neg)
    auc = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos) / (pos.size * neg.size)
    return top1, auc, tot


def main():
    print("Loading features (DINOv2 + SIFT)...")
    feats = load_features()
    print(f"{len(feats)} images, {len({f['label'] for f in feats})} individuals\n")

    # combined needs frozen norm constants computed here
    combos = list(itertools.combinations(range(len(feats)), 2))
    cos = np.array([s_cosine(feats[i], feats[j]) for i, j in combos])
    pat = np.array([s_fg_sym(feats[i], feats[j]) for i, j in combos])
    cmu, csd, pmu, psd = cos.mean(), cos.std(), pat.mean(), pat.std()

    def s_combined(a, b):
        return 0.5 * ((s_cosine(a, b) - cmu) / csd) + 0.5 * ((s_fg_sym(a, b) - pmu) / psd)

    methods = {
        "cosine (retrieval actuel)": s_cosine,
        "fg_sym (verifier actuel)": s_fg_sym,
        "combined cos+patch": s_combined,
        "patch+RANSAC (A)": s_patch_ransac,
        "sift+RANSAC (B)": s_sift_ransac,
    }
    print(f"{'method':30} {'top-1':>8} {'AUC':>8}")
    print("-" * 48)
    for name, fn in methods.items():
        top1, auc, tot = evaluate(feats, fn)
        print(f"{name:30} {top1:7.1%} {auc:8.3f}")
    print(f"\n(top-1 sur {tot} requetes)")


if __name__ == "__main__":
    main()
