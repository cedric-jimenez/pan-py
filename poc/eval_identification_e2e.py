"""End-to-end identification evaluation on a labelled photo corpus.

Unlike ``eval_identification.py`` (which scores the verifier alone on already
cropped ``docs/images/<idN>/`` images), this harness runs the **full production
identification chain** on raw field photos and decomposes the result per stage,
so you can see *where* identification gets approximate:

    segment (YOLO-seg)  ->  embed (DINOv2 GeM, 384-d)  ->  verify (SIFT+RANSAC)
            │                        │                             │
      detection miss?          retrieval recall?            discrimination?

It reports three independent blocks of numbers:

  1. DETECTION      how many photos the segmenter actually finds a salamander in
                    (a miss = no embedding stored in prod = a silent ID failure).
  2. RETRIEVAL      the pgvector cosine pre-filter: for each query, is a
                    same-individual photo in the top-K? recall@K + median rank
                    of the first true match. This is the known weak link
                    (DINOv2 cosine barely separates individuals).
  3. VERIFY + E2E   the SIFT re-ranker: same/diff score separation (AUC-ROC, FP
                    rate at the prod threshold) AND the end-to-end top-1
                    identification accuracy that mirrors /api/photos/[id]/similar
                    (take top-N by cosine, re-rank by verify, is #1 the right
                    individual?). Also reported: cosine-only top-1, to isolate
                    how much the verifier actually adds.

Dataset layout: ``<root>/<label>/<image>``. Each sub-folder name is the
individual label; an individual needs >=2 photos to produce same-pairs.

Run from the pan-py repo root (needs the ML deps + models, i.e. Python 3.11
with requirements.txt installed — NOT the bare 3.14 venv):

    ./venv/bin/python poc/eval_identification_e2e.py \
        --dataset /home/perso/www/Echantillons/photos-terrain/par-individu --segment

    # already-cropped set (no detection stage):
    ./venv/bin/python poc/eval_identification_e2e.py --dataset docs/images

    ./venv/bin/python poc/eval_identification_e2e.py --dataset <dir> --segment --json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Make `app` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.segmenter import SalamanderSegmenter  # noqa: E402
from app.identification.embedder import SalamanderEmbedder  # noqa: E402
from app.identification.verifier import SalamanderVerifier  # noqa: E402

DEFAULT_DATASET = "/home/perso/www/Echantillons/photos-terrain/par-individu"
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


@dataclass
class Sample:
    """One evaluated photo after the detect/segment stage."""

    label: str
    path: Path
    image: Image.Image  # segmented (or raw, with --no-segment) PIL image
    embedding: np.ndarray  # (D,) L2-normalized


# ---------------------------------------------------------------------------
# Metric helpers (self-contained, mirror eval_identification.py)
# ---------------------------------------------------------------------------
def auc_roc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive outscores a random negative) — Mann-Whitney U."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (pos.size * neg.size))


def best_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing accuracy over pooled scores."""
    if pos.size == 0 or neg.size == 0:
        return 0.0, float("nan")
    scores = np.concatenate([pos, neg])
    best_acc, best_thr = 0.0, 0.0
    for thr in np.unique(scores):
        tp = (pos >= thr).sum()
        tn = (neg < thr).sum()
        acc = (tp + tn) / len(scores)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, float(best_acc)


# ---------------------------------------------------------------------------
# Data loading + pipeline
# ---------------------------------------------------------------------------
def discover(root: Path) -> list[tuple[str, Path]]:
    """Return (label, path) for every image under <root>/<label>/."""
    return sorted((p.parent.name, p) for ext in IMAGE_EXTS for p in root.glob(f"*/{ext}"))


@dataclass
class PipelineStats:
    total: int = 0
    undetected: list[str] = field(default_factory=list)  # "label/stem" of seg misses


def build_samples(
    items: list[tuple[str, Path]],
    *,
    segment: bool,
    seg: SalamanderSegmenter | None,
    emb: SalamanderEmbedder,
    conf: float,
) -> tuple[list[Sample], PipelineStats]:
    """Run detect(optional) -> embed on every image, collecting failures."""
    samples: list[Sample] = []
    stats = PipelineStats(total=len(items))
    for label, path in items:
        img = Image.open(path)
        if segment:
            assert seg is not None
            detected, data = seg.segment(img, conf_threshold=conf)
            if not detected or data is None:
                stats.undetected.append(f"{label}/{path.stem}")
                continue
            img = data["segmented_image"]
        vec = emb.embed(img)
        samples.append(Sample(label=label, path=path, image=img, embedding=vec))
    return samples, stats


# ---------------------------------------------------------------------------
# Stage 2 — retrieval (cosine pre-filter)
# ---------------------------------------------------------------------------
def retrieval_metrics(samples: list[Sample], ks: list[int]) -> dict:
    """Leave-one-out cosine retrieval. recall@K + rank of first true match."""
    labels = [s.label for s in samples]
    mat = np.stack([s.embedding for s in samples])  # (N, D), L2-normalized
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)  # never retrieve the query itself

    label_counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    first_hit_ranks: list[int] = []  # 1-indexed rank of first same-individual
    recall_hits = {k: 0 for k in ks}
    n_queries = 0
    for i, lbl in enumerate(labels):
        if label_counts[lbl] < 2:
            continue  # no positive exists for this query
        n_queries += 1
        order = np.argsort(-sim[i])  # candidate indices, best first
        ranked_labels = [labels[j] for j in order]
        rank = next(r for r, rl in enumerate(ranked_labels, 1) if rl == lbl)
        first_hit_ranks.append(rank)
        for k in ks:
            if rank <= k:
                recall_hits[k] += 1

    ranks = np.array(first_hit_ranks) if first_hit_ranks else np.array([np.nan])
    return {
        "n_queries": n_queries,
        "recall_at": {k: (recall_hits[k] / n_queries if n_queries else float("nan")) for k in ks},
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "max_rank": float(np.max(ranks)),
    }


# ---------------------------------------------------------------------------
# Stage 3 — verify discrimination + end-to-end identification
# ---------------------------------------------------------------------------
def verify_pair_metrics(samples: list[Sample], verifier: SalamanderVerifier) -> dict:
    """All unordered pairs: same/diff separation on BOTH score and inlier count."""
    pos, neg = [], []  # scores
    pos_inl, neg_inl = [], []  # inlier counts
    worst_fp: list[tuple[float, int, str, str]] = []
    for i, j in itertools.combinations(range(len(samples)), 2):
        v = verifier.verify(samples[i].image, samples[j].image)
        score, inliers = v["score"], v["inliers"]
        if samples[i].label == samples[j].label:
            pos.append(score)
            pos_inl.append(inliers)
        else:
            neg.append(score)
            neg_inl.append(inliers)
            worst_fp.append((score, inliers, samples[i].label, samples[j].label))
    pos_a, neg_a = np.array(pos), np.array(neg)
    pos_i, neg_i = np.array(pos_inl), np.array(neg_inl)
    thr, acc = best_threshold(pos_a, neg_a)
    prod_thr = verifier.high_threshold
    worst_fp.sort(reverse=True)
    return {
        "n_same": int(pos_a.size),
        "n_diff": int(neg_a.size),
        "same": _dist(pos_a),
        "diff": _dist(neg_a),
        "same_inliers": _dist(pos_i),
        "diff_inliers": _dist(neg_i),
        "same_scores": pos_a.tolist(),
        "diff_scores": neg_a.tolist(),
        "same_inlier_counts": pos_i.tolist(),
        "diff_inlier_counts": neg_i.tolist(),
        "auc_roc": auc_roc(pos_a, neg_a),
        "auc_roc_inliers": auc_roc(pos_i, neg_i),
        "best_threshold": {"thr": thr, "accuracy": acc},
        "prod_threshold": prod_thr,
        "false_positives_at_prod": int((neg_a >= prod_thr).sum()),
        "top_false_positives": [
            {"score": round(s, 3), "inliers": inl, "a": a, "b": b}
            for s, inl, a, b in worst_fp[:5]
        ],
    }


def inlier_floor_sweep(ver: dict, floors: list[int]) -> list[dict]:
    """For each candidate inlier floor: same-pairs kept (recall) vs diff-pairs
    that still pass (false positives). Isolates inlier count as a gate."""
    same = np.array(ver["same_inlier_counts"])
    diff = np.array(ver["diff_inlier_counts"])
    rows = []
    for f in floors:
        rows.append({
            "floor": f,
            "same_kept": float((same >= f).sum() / same.size) if same.size else float("nan"),
            "diff_pass": int((diff >= f).sum()),
            "n_diff": int(diff.size),
        })
    return rows


def e2e_metrics(samples: list[Sample], verifier: SalamanderVerifier, topn: int) -> dict:
    """Mirror /similar: top-N by cosine, re-rank by verify, judge top-1.

    Reports three accuracies over queries whose individual has >=2 photos:
      - cosine_top1:  top-1 by cosine alone (no verify)
      - verify_top1:  top-1 after verify re-rank of the cosine top-N pool
      - verify_top1_thresholded: same, but a top candidate below the prod
                       threshold is treated as "no match" (wrong if a true
                       match existed) — the honest production behaviour.
    """
    labels = [s.label for s in samples]
    mat = np.stack([s.embedding for s in samples])
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    prod_thr = verifier.high_threshold

    n = cos_ok = ver_ok = ver_thr_ok = 0
    per_query: list[dict] = []  # {"correct": bool, "score": float} for the verify top-1
    for i, lbl in enumerate(labels):
        if counts[lbl] < 2:
            continue
        n += 1
        order = [j for j in np.argsort(-sim[i]) if np.isfinite(sim[i][j])]
        pool = order[:topn]
        if labels[pool[0]] == lbl:
            cos_ok += 1
        verdicts = verifier.verify_against_many(samples[i].image, [samples[j].image for j in pool])
        best = verdicts[0]  # sorted by score desc
        best_j = pool[best["candidate_index"]]
        correct = labels[best_j] == lbl
        per_query.append({"correct": correct, "score": float(best["score"])})
        if correct:
            ver_ok += 1
            if best["score"] >= prod_thr:
                ver_thr_ok += 1
    return {
        "n_queries": n,
        "topn_pool": topn,
        "cosine_top1_accuracy": cos_ok / n if n else float("nan"),
        "verify_top1_accuracy": ver_ok / n if n else float("nan"),
        "verify_top1_thresholded_accuracy": ver_thr_ok / n if n else float("nan"),
        "per_query": per_query,
    }


def threshold_sweep(ver: dict, e2e: dict, thresholds: list[float]) -> list[dict]:
    """Per-threshold tradeoff at the is_same boundary: recall, FP, end-to-end ID.

    For each candidate is_same threshold T:
      - verify_recall  = fraction of same-pairs scoring >= T   (true matches kept)
      - false_positives = diff-pairs scoring >= T              (imposters let in)
      - e2e_identified = queries whose verify top-1 is the right individual AND
                         scores >= T (i.e. actually surfaced as "same" in prod)
    """
    same = np.array(ver["same_scores"])
    diff = np.array(ver["diff_scores"])
    pq = e2e["per_query"]
    n_same, n_diff, n_q = same.size, diff.size, len(pq)
    rows = []
    for t in thresholds:
        identified = sum(1 for q in pq if q["correct"] and q["score"] >= t)
        rows.append({
            "threshold": t,
            "verify_recall": float((same >= t).sum() / n_same) if n_same else float("nan"),
            "false_positives": int((diff >= t).sum()),
            "n_diff": n_diff,
            "e2e_identified_accuracy": identified / n_q if n_q else float("nan"),
        })
    return rows


def _dist(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 3),
        "mean": round(float(a.mean()), 3),
        "max": round(float(a.max()), 3),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(ds: Path, stats: PipelineStats, per_ind: dict, retr: dict,
                 ver: dict, e2e: dict, sweep: list[dict], inl_sweep: list[dict]) -> None:
    print(f"\n{'=' * 70}\nIDENTIFICATION — END-TO-END EVAL\n{'=' * 70}")
    print(f"dataset: {ds}")
    print(f"individuals: {len(per_ind)}  |  photos: {stats.total}  |  "
          f"per-individual: {dict(sorted(per_ind.items()))}")

    print("\n[1] DETECTION (segment stage)")
    found = stats.total - len(stats.undetected)
    print(f"  salamander found: {found}/{stats.total} "
          f"({found / stats.total:.0%})" if stats.total else "  (no photos)")
    if stats.undetected:
        print(f"  MISSED (no embedding would be stored in prod): {len(stats.undetected)}")
        for u in stats.undetected[:10]:
            print(f"    - {u}")

    print(f"\n[2] RETRIEVAL (cosine pre-filter, leave-one-out, n={retr['n_queries']} queries)")
    r = retr["recall_at"]
    print("  recall@K:  " + "  ".join(f"@{k}={r[k]:.0%}" for k in sorted(r)))
    print(f"  rank of first true match:  median={retr['median_rank']:.0f}  "
          f"mean={retr['mean_rank']:.1f}  worst={retr['max_rank']:.0f}")
    print("  (if median rank is high / recall@10 is low -> the true match is")
    print("   buried in the candidate pool: the weak link is retrieval, not verify.)")

    print("\n[3] VERIFY (SIFT+RANSAC) — discrimination over all pairs")
    print(f"  same pairs (n={ver['n_same']}): {ver['same']}")
    print(f"  diff pairs (n={ver['n_diff']}): {ver['diff']}")
    print(f"  AUC-ROC: {ver['auc_roc']:.3f}   (1.0 = perfect separation, 0.5 = random)")
    bt = ver["best_threshold"]
    print(f"  best-threshold accuracy: {bt['accuracy']:.1%} @ thr={bt['thr']:.3f}")
    print(f"  false positives at prod thr {ver['prod_threshold']}: "
          f"{ver['false_positives_at_prod']}/{ver['n_diff']}")
    if ver["top_false_positives"]:
        print("  worst diff-pair scores (would be false matches):")
        for fp in ver["top_false_positives"]:
            print(f"    score={fp['score']:.3f} inliers={fp['inliers']:>2}  {fp['a']} vs {fp['b']}")

    print("\n[3b] INLIER COUNT as the discriminator (prod data suggests count > ratio)")
    print(f"  same pairs inliers: {ver['same_inliers']}")
    print(f"  diff pairs inliers: {ver['diff_inliers']}")
    print(f"  AUC-ROC on inlier count: {ver['auc_roc_inliers']:.3f}  "
          f"(vs {ver['auc_roc']:.3f} on ratio score)")
    print(f"  {'floor':>6} {'same_kept(recall)':>18} {'diff_pass(FP)':>16}")
    for row in inl_sweep:
        print(f"  {row['floor']:>6} {row['same_kept']:>17.0%} "
              f"{row['diff_pass']:>10}/{row['n_diff']:<4}")

    print(f"\n[4] END-TO-END IDENTIFICATION (mirrors /similar, pool=top-{e2e['topn_pool']}, "
          f"n={e2e['n_queries']})")
    print(f"  cosine-only     top-1 accuracy: {e2e['cosine_top1_accuracy']:.1%}")
    print(f"  +verify re-rank top-1 accuracy: {e2e['verify_top1_accuracy']:.1%}  (no threshold)")

    print("\n[5] is_same THRESHOLD SWEEP (the knob that decides 'same individual')")
    print("    Pan surfaces a match when is_same=True, i.e. verify score >= medium_threshold.")
    print(f"    {'thr':>6} {'verify_recall':>14} {'false_pos':>12} {'e2e_identified':>15}")
    for row in sweep:
        print(f"    {row['threshold']:>6.3f} "
              f"{row['verify_recall']:>13.0%} "
              f"{row['false_positives']:>7}/{row['n_diff']:<4} "
              f"{row['e2e_identified_accuracy']:>14.0%}")
    print(f"{'=' * 70}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="root dir of <label>/<image>")
    ap.add_argument("--segment", action="store_true",
                    help="run YOLO-seg first (use for raw field photos)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--topn", type=int, default=25, help="cosine candidate pool for verify re-rank")
    ap.add_argument("--json", type=Path, default=None, help="also write metrics as JSON here")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    items = discover(root)
    if not items:
        sys.exit(f"No images found under {root} (expected <root>/<label>/<image>)")
    per_ind: dict[str, int] = {}
    for lbl, _ in items:
        per_ind[lbl] = per_ind.get(lbl, 0) + 1

    print(f"Loading models (segment={args.segment}) and processing {len(items)} photos...")
    seg = None
    if args.segment:
        seg = SalamanderSegmenter()
        seg.load_model()
    emb = SalamanderEmbedder()
    emb.load_model()
    verifier = SalamanderVerifier()

    samples, stats = build_samples(items, segment=args.segment, seg=seg, emb=emb, conf=args.conf)
    if len(samples) < 2:
        sys.exit(f"Only {len(samples)} photos survived the pipeline — cannot evaluate.")

    ks = [k for k in (1, 5, 10, 25, 50, 100) if k < len(samples)] or [1]
    retr = retrieval_metrics(samples, ks)
    ver = verify_pair_metrics(samples, verifier)
    e2e = e2e_metrics(samples, verifier, min(args.topn, len(samples) - 1))
    sweep = threshold_sweep(ver, e2e, [0.05, 0.06, 0.07, 0.08, 0.10, 0.15])
    inl_sweep = inlier_floor_sweep(ver, [6, 8, 10, 12, 15, 20])

    print_report(root, stats, per_ind, retr, ver, e2e, sweep, inl_sweep)

    if args.json:
        # Drop the bulky raw score arrays from the JSON dump.
        drop = ("same_scores", "diff_scores", "same_inlier_counts", "diff_inlier_counts")
        ver_out = {k: v for k, v in ver.items() if k not in drop}
        e2e_out = {k: v for k, v in e2e.items() if k != "per_query"}
        args.json.write_text(json.dumps(
            {"dataset": str(root), "per_individual": per_ind,
             "detection": {"total": stats.total, "undetected": stats.undetected},
             "retrieval": retr, "verify": ver_out, "end_to_end": e2e_out,
             "threshold_sweep": sweep, "inlier_floor_sweep": inl_sweep}, indent=2))
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
