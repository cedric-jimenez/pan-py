#!/usr/bin/env python3
"""Test script for DINOv2 verification algorithm evaluation.

Evaluates the SalamanderVerifier on a set of test images with known ground truth.
Reports confusion matrix, classification metrics, and score distributions.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from app.identification import SalamanderEmbedder, SalamanderVerifier

# Ground truth: pairs that should be classified as SAME individual
# Format: set of frozenset pairs for fast lookup
SAME_INDIVIDUAL_PAIRS: set[frozenset[str]] = {
    frozenset({"titine-1.jpg", "titine-2.jpg"}),
}


def load_images(images_dir: Path) -> dict[str, Image.Image]:
    """Load all images from directory.

    Args:
        images_dir: Directory containing test images.

    Returns:
        Dictionary mapping filename to PIL Image.
    """
    images = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in images_dir.glob(ext):
            images[path.name] = Image.open(path).convert("RGB")
    return images


def extract_all_features(
    images: dict[str, Image.Image],
    embedder: SalamanderEmbedder,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract features for all images (optimized single pass).

    Args:
        images: Dictionary mapping filename to PIL Image.
        embedder: Loaded SalamanderEmbedder instance.

    Returns:
        Dictionary mapping filename to (embedding, patch_tokens, patch_norms).
    """
    features = {}
    for name, image in images.items():
        emb, patches, content_ratio = embedder.extract_features(image)
        features[name] = (emb, patches, content_ratio)
    return features


def is_same_ground_truth(name1: str, name2: str) -> bool:
    """Check if two images should be classified as same individual."""
    return frozenset({name1, name2}) in SAME_INDIVIDUAL_PAIRS


def compute_pairwise_scores(
    image_names: list[str],
    features: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    verifier: SalamanderVerifier,
) -> list[dict]:
    """Compute verification scores for all image pairs.

    Args:
        image_names: List of image filenames.
        features: Pre-extracted features for each image.
        verifier: SalamanderVerifier instance.

    Returns:
        List of result dictionaries for each pair.
    """
    results = []
    for name1, name2 in itertools.combinations(image_names, 2):
        emb1, patches1, cr1 = features[name1]
        emb2, patches2, cr2 = features[name2]

        # Compute scores using verifier's methods
        cosine_sim = float(np.dot(emb1, emb2))
        score, matches, inliers = verifier._patch_match_score(
            patches1, patches2, cr1, cr2,
        )
        is_same, confidence = verifier._classify(score)

        results.append({
            "image1": name1,
            "image2": name2,
            "score": score,
            "is_same": is_same,
            "confidence": confidence,
            "cosine_similarity": cosine_sim,
            "matches": matches,
            "inliers": inliers,
            "expected_same": is_same_ground_truth(name1, name2),
        })

    return results


def classify_results(results: list[dict]) -> dict[str, list[dict]]:
    """Classify results into TP, TN, FP, FN categories.

    Args:
        results: List of pairwise comparison results.

    Returns:
        Dictionary with keys 'TP', 'TN', 'FP', 'FN', each containing list of results.
    """
    classification = {"TP": [], "TN": [], "FP": [], "FN": []}

    for r in results:
        predicted_same = r["is_same"]
        actual_same = r["expected_same"]

        if actual_same and predicted_same:
            classification["TP"].append(r)
        elif not actual_same and not predicted_same:
            classification["TN"].append(r)
        elif not actual_same and predicted_same:
            classification["FP"].append(r)
        else:  # actual_same and not predicted_same
            classification["FN"].append(r)

    return classification


def compute_metrics(classification: dict[str, list[dict]]) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        classification: Dictionary with TP, TN, FP, FN lists.

    Returns:
        Dictionary with accuracy, precision, recall, f1, specificity.
    """
    tp = len(classification["TP"])
    tn = len(classification["TN"])
    fp = len(classification["FP"])
    fn = len(classification["FN"])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_header(
    model_name: str,
    n_images: int,
    n_pairs: int,
    thresholds: tuple[float, float, float],
) -> None:
    """Print evaluation header."""
    print("=" * 80)
    print("VERIFICATION ALGORITHM EVALUATION")
    print("=" * 80)
    print(f"Model: {model_name} | Images: {n_images} | Pairs: {n_pairs}")
    print()
    high, medium, low = thresholds
    print(f"THRESHOLDS: high >= {high:.2f} | medium >= {medium:.2f} | low >= {low:.2f}")
    print()


def print_confusion_matrix(classification: dict[str, list[dict]], metrics: dict[str, float]) -> None:
    """Print confusion matrix and metrics."""
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print(f"                 Predicted SAME    Predicted DIFF")
    print(f"Actual SAME      {metrics['tp']:5d} (TP)        {metrics['fn']:5d} (FN)")
    print(f"Actual DIFF      {metrics['fp']:5d} (FP)        {metrics['tn']:5d} (TN)")
    print()
    print(
        f"Accuracy: {metrics['accuracy']*100:.1f}% | "
        f"Precision: {metrics['precision']*100:.1f}% | "
        f"Recall: {metrics['recall']*100:.1f}% | "
        f"F1: {metrics['f1']:.2f}"
    )
    print(f"Specificity: {metrics['specificity']*100:.1f}%")
    print()


def print_score_analysis(results: list[dict]) -> None:
    """Print score distribution analysis."""
    same_scores = [r["score"] for r in results if r["expected_same"]]
    diff_scores = [r["score"] for r in results if not r["expected_same"]]

    print("=" * 80)
    print("SCORE DISTRIBUTION")
    print("=" * 80)

    if same_scores:
        print(
            f"Same pairs:      min={min(same_scores):.3f}, "
            f"max={max(same_scores):.3f}, "
            f"mean={np.mean(same_scores):.3f}"
        )
    else:
        print("Same pairs:      (none)")

    if diff_scores:
        print(
            f"Different pairs: min={min(diff_scores):.3f}, "
            f"max={max(diff_scores):.3f}, "
            f"mean={np.mean(diff_scores):.3f}"
        )
    else:
        print("Different pairs: (none)")

    if same_scores and diff_scores:
        gap = min(same_scores) - max(diff_scores)
        if gap > 0:
            print(f"Gap: {gap:.3f} (good separation)")
        else:
            print(f"Gap: {gap:.3f} (OVERLAP - threshold tuning needed)")
    print()


def print_detailed_results(results: list[dict], top_n: int = 10) -> None:
    """Print detailed results table, sorted by score."""
    print("=" * 80)
    print(f"DETAILED RESULTS (top {top_n} by score)")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    def short_name(name: str) -> str:
        """Shorten filename for display."""
        if name.startswith("IMG_"):
            return name[4:7]  # Just the unique part
        return name.replace(".jpg", "").replace(".png", "")[:10]

    print(f"{'#':>2}  {'Image 1':<12} {'Image 2':<12} {'Score':>7}  {'Match':>5} {'Inlr':>5}  {'Pred':>6}  {'Expect':>6}  OK")
    print("-" * 85)

    for i, r in enumerate(sorted_results[:top_n], 1):
        conf_abbrev = {"high": "H", "medium": "M", "low": "L"}.get(r["confidence"], "?")
        pred = f"SAME({conf_abbrev})" if r["is_same"] else f"DIFF({conf_abbrev})"
        expect = "SAME" if r["expected_same"] else "DIFF"
        ok = "Y" if r["is_same"] == r["expected_same"] else "X"
        matches = r.get("matches", 0)
        inliers = r.get("inliers", 0)
        print(
            f"{i:2d}  {short_name(r['image1']):<12} {short_name(r['image2']):<12} "
            f"{r['score']:>7.3f}  {matches:>5} {inliers:>5}  {pred:>6}  {expect:>6}  {ok}"
        )
    print()


def print_similarity_matrix(
    image_names: list[str],
    results: list[dict],
) -> None:
    """Print N×N similarity matrix."""
    n = len(image_names)
    name_to_idx = {name: i for i, name in enumerate(image_names)}

    # Build score matrix
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)  # Self-similarity = 1

    for r in results:
        i = name_to_idx[r["image1"]]
        j = name_to_idx[r["image2"]]
        matrix[i, j] = r["score"]
        matrix[j, i] = r["score"]

    print("=" * 80)
    print("SIMILARITY MATRIX (patch match scores)")
    print("=" * 80)

    def short_name(name: str) -> str:
        if name.startswith("IMG_"):
            return name[4:7]
        return name.replace(".jpg", "")[:6]

    # Header
    print(f"{'':>8}", end="")
    for name in image_names:
        print(f"{short_name(name):>8}", end="")
    print()

    # Rows
    for i, name in enumerate(image_names):
        print(f"{short_name(name):>8}", end="")
        for j in range(n):
            print(f"{matrix[i, j]:>8.3f}", end="")
        print()
    print()


def print_errors(classification: dict[str, list[dict]]) -> None:
    """Print details of misclassified pairs."""
    fp = classification["FP"]
    fn = classification["FN"]

    if not fp and not fn:
        return

    print("=" * 80)
    print("CLASSIFICATION ERRORS")
    print("=" * 80)

    if fp:
        print("\nFalse Positives (predicted SAME, actually DIFFERENT):")
        for r in fp:
            print(f"  - {r['image1']} vs {r['image2']}: score={r['score']:.3f}")

    if fn:
        print("\nFalse Negatives (predicted DIFFERENT, actually SAME):")
        for r in fn:
            print(f"  - {r['image1']} vs {r['image2']}: score={r['score']:.3f}")
    print()


def output_json(
    results: list[dict],
    classification: dict[str, list[dict]],
    metrics: dict[str, float],
    thresholds: tuple[float, float, float],
) -> None:
    """Output results as JSON."""
    output = {
        "thresholds": {
            "high": thresholds[0],
            "medium": thresholds[1],
            "low": thresholds[2],
        },
        "metrics": metrics,
        "pairs": results,
    }
    print(json.dumps(output, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DINOv2 verification algorithm on test images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent / "images",
        help="Directory containing test images (default: poc/images)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.25,
        help="High confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.15,
        help="Medium confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.10,
        help="Low confidence threshold (default: 0.10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vits14",
        help="DINOv2 model name (default: dinov2_vits14)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show additional details",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Show full similarity matrix",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Validate images directory
    if not args.images_dir.is_dir():
        print(f"Error: Images directory not found: {args.images_dir}", file=sys.stderr)
        return 1

    # Load images
    images = load_images(args.images_dir)
    if not images:
        print(f"Error: No images found in {args.images_dir}", file=sys.stderr)
        return 1

    image_names = sorted(images.keys())
    n_pairs = len(image_names) * (len(image_names) - 1) // 2

    # Initialize embedder and verifier
    print("Loading model...", file=sys.stderr)
    embedder = SalamanderEmbedder(model_name=args.model)
    embedder.load_model()

    verifier = SalamanderVerifier(
        embedder=embedder,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
        low_threshold=args.low_threshold,
    )

    # Extract features once for all images
    print("Extracting features...", file=sys.stderr)
    features = extract_all_features(images, embedder)

    # Compute pairwise scores
    print("Computing pairwise scores...", file=sys.stderr)
    results = compute_pairwise_scores(image_names, features, verifier)

    # Classify and compute metrics
    classification = classify_results(results)
    metrics = compute_metrics(classification)

    thresholds = (args.high_threshold, args.medium_threshold, args.low_threshold)

    # Output
    if args.json:
        output_json(results, classification, metrics, thresholds)
    else:
        print_header(args.model, len(images), n_pairs, thresholds)
        print_confusion_matrix(classification, metrics)
        print_score_analysis(results)
        print_detailed_results(results, top_n=15 if args.verbose else 10)
        print_errors(classification)

        if args.matrix:
            print_similarity_matrix(image_names, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
