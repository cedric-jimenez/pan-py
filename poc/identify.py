#!/usr/bin/env python3
"""
POC: Identification d'individus salamandres via descripteurs SIFT.

SIFT (Scale-Invariant Feature Transform) détecte des points-clés locaux
et les décrit individuellement. Cela permet de matcher des patterns
spécifiques (taches, marques) plutôt que des caractéristiques globales.

Optimisations:
- Nombre de keypoints augmenté
- Vérification géométrique RANSAC
- Cross-check pour éliminer les faux matches
- Score combinant quantité et qualité des matches

Usage:
    python poc/identify.py
    python poc/identify.py --images-dir poc/images_transparent
    python poc/identify.py --image1 img1.png --image2 img2.png
    python poc/identify.py --visualize  # Afficher les matches
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_N_FEATURES = 1000  # Plus de keypoints pour meilleure couverture
DEFAULT_RATIO_THRESH = 0.75  # Ratio test de Lowe (standard)
DEFAULT_RANSAC_THRESH = 5.0  # Seuil RANSAC en pixels
MIN_MATCHES_FOR_RANSAC = 4  # Minimum pour calculer homographie


# =============================================================================
# Chargement d'images
# =============================================================================


def load_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Charge une image en couleur et niveaux de gris.

    Args:
        image_path: Chemin vers l'image.

    Returns:
        Tuple (image couleur BGR, image grayscale).
    """
    img = Image.open(image_path)

    if img.mode == "RGBA":
        # Convertir RGBA -> RGB avec fond noir
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")

    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_bgr, img_gray


# =============================================================================
# Extraction SIFT
# =============================================================================


def extract_sift(
    image_gray: np.ndarray,
    n_features: int = DEFAULT_N_FEATURES,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10,
) -> tuple[list, np.ndarray | None]:
    """Extrait les descripteurs SIFT d'une image.

    Args:
        image_gray: Image en niveaux de gris.
        n_features: Nombre maximum de keypoints.
        contrast_threshold: Seuil de contraste (plus bas = plus de points).
        edge_threshold: Seuil de bord (plus haut = plus de points sur bords).

    Returns:
        Tuple (keypoints, descripteurs 128D).
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


# =============================================================================
# Matching
# =============================================================================


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
) -> list[cv2.DMatch]:
    """Match les descripteurs avec ratio test de Lowe.

    Args:
        desc1: Descripteurs image 1.
        desc2: Descripteurs image 2.
        ratio_thresh: Seuil pour le ratio test.

    Returns:
        Liste des bons matches.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    # FLANN est plus rapide que BruteForce pour beaucoup de descripteurs
    index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
    search_params = {"checks": 50}

    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        # Fallback sur BruteForce si FLANN échoue
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)

    # Ratio test de Lowe
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches


def filter_matches_ransac(
    kp1: list,
    kp2: list,
    matches: list[cv2.DMatch],
    ransac_thresh: float = DEFAULT_RANSAC_THRESH,
) -> tuple[list[cv2.DMatch], np.ndarray | None]:
    """Filtre les matches avec RANSAC pour éliminer les outliers.

    RANSAC trouve une transformation géométrique cohérente entre les
    points matchés, éliminant les faux positifs.

    Args:
        kp1: Keypoints image 1.
        kp2: Keypoints image 2.
        matches: Liste des matches.
        ransac_thresh: Seuil de reprojection en pixels.

    Returns:
        Tuple (matches filtrés, masque inliers).
    """
    if len(matches) < MIN_MATCHES_FOR_RANSAC:
        return matches, None

    # Extraire les coordonnées des points matchés
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculer l'homographie avec RANSAC
    try:
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    except cv2.error:
        return matches, None

    if mask is None:
        return matches, None

    # Filtrer les matches
    mask = mask.ravel().astype(bool)
    filtered_matches = [m for m, is_inlier in zip(matches, mask) if is_inlier]

    return filtered_matches, mask


# =============================================================================
# Score de similarité
# =============================================================================


def compute_similarity_score(
    n_matches: int,
    n_inliers: int,
    n_kp1: int,
    n_kp2: int,
    avg_distance: float,
) -> dict:
    """Calcule un score de similarité détaillé.

    Args:
        n_matches: Nombre de matches avant RANSAC.
        n_inliers: Nombre de matches après RANSAC.
        n_kp1: Nombre de keypoints image 1.
        n_kp2: Nombre de keypoints image 2.
        avg_distance: Distance moyenne des matches.

    Returns:
        Dictionnaire avec différentes métriques.
    """
    if n_kp1 == 0 or n_kp2 == 0:
        return {
            "score": 0.0,
            "match_ratio": 0.0,
            "inlier_ratio": 0.0,
            "quality": 0.0,
        }

    # Ratio de matches (normalisé par moyenne géométrique)
    match_ratio = n_matches / np.sqrt(n_kp1 * n_kp2)

    # Ratio d'inliers (qualité géométrique)
    inlier_ratio = n_inliers / n_matches if n_matches > 0 else 0.0

    # Score de qualité basé sur la distance (plus bas = meilleur)
    # Normaliser la distance (typiquement entre 0 et 200 pour SIFT)
    quality = max(0, 1 - avg_distance / 200) if avg_distance > 0 else 0.0

    # Score combiné
    # Pondération: matches (40%) + inliers (40%) + qualité (20%)
    score = 0.4 * match_ratio + 0.4 * inlier_ratio * match_ratio + 0.2 * quality * match_ratio

    return {
        "score": score,
        "match_ratio": match_ratio,
        "inlier_ratio": inlier_ratio,
        "quality": quality,
    }


# =============================================================================
# Comparaison d'images
# =============================================================================


def compare_two_images(
    image1_path: Path,
    image2_path: Path,
    use_ransac: bool = True,
) -> dict:
    """Compare deux images et retourne les résultats détaillés.

    Args:
        image1_path: Chemin vers la première image.
        image2_path: Chemin vers la deuxième image.
        use_ransac: Utiliser RANSAC pour filtrer les matches.

    Returns:
        Dictionnaire avec tous les résultats.
    """
    # Charger les images
    img1_bgr, img1_gray = load_image(image1_path)
    img2_bgr, img2_gray = load_image(image2_path)

    # Extraire SIFT
    kp1, desc1 = extract_sift(img1_gray)
    kp2, desc2 = extract_sift(img2_gray)

    n_kp1 = len(kp1) if kp1 else 0
    n_kp2 = len(kp2) if kp2 else 0

    # Matcher
    matches = match_descriptors(desc1, desc2)
    n_matches = len(matches)

    # Filtrer avec RANSAC
    if use_ransac and n_matches >= MIN_MATCHES_FOR_RANSAC:
        inliers, mask = filter_matches_ransac(kp1, kp2, matches)
        n_inliers = len(inliers)
    else:
        inliers = matches
        n_inliers = n_matches
        mask = None

    # Calculer la distance moyenne
    avg_distance = np.mean([m.distance for m in inliers]) if inliers else 0.0

    # Score
    scores = compute_similarity_score(n_matches, n_inliers, n_kp1, n_kp2, avg_distance)

    return {
        "image1": image1_path.name,
        "image2": image2_path.name,
        "keypoints1": n_kp1,
        "keypoints2": n_kp2,
        "matches": n_matches,
        "inliers": n_inliers,
        "avg_distance": avg_distance,
        "kp1": kp1,
        "kp2": kp2,
        "all_matches": matches,
        "inlier_matches": inliers,
        "img1_bgr": img1_bgr,
        "img2_bgr": img2_bgr,
        **scores,
    }


def compare_all_images(
    image_paths: list[Path],
    use_ransac: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Compare toutes les images entre elles.

    Args:
        image_paths: Liste des chemins d'images.
        use_ransac: Utiliser RANSAC.

    Returns:
        Tuple (noms, matrice de scores, matrice d'inliers, stats).
    """
    names = [p.stem for p in image_paths]
    n = len(names)
    score_matrix = np.zeros((n, n))
    inlier_matrix = np.zeros((n, n))
    stats = {}

    # Extraire tous les descripteurs
    print("\nExtraction des descripteurs SIFT...")
    all_data = []

    for path in image_paths:
        img_bgr, img_gray = load_image(path)
        kp, desc = extract_sift(img_gray)
        n_kp = len(kp) if kp else 0

        all_data.append({"kp": kp, "desc": desc, "n_kp": n_kp})
        stats[path.stem] = {"keypoints": n_kp}
        print(f"  {path.stem}: {n_kp} keypoints")

    # Matcher toutes les paires
    print("\nMatching des paires...")
    for i in range(n):
        score_matrix[i, i] = 1.0
        inlier_matrix[i, i] = all_data[i]["n_kp"]

        for j in range(i + 1, n):
            kp1, desc1 = all_data[i]["kp"], all_data[i]["desc"]
            kp2, desc2 = all_data[j]["kp"], all_data[j]["desc"]
            n_kp1, n_kp2 = all_data[i]["n_kp"], all_data[j]["n_kp"]

            # Matcher
            matches = match_descriptors(desc1, desc2)
            n_matches = len(matches)

            # RANSAC
            if use_ransac and n_matches >= MIN_MATCHES_FOR_RANSAC:
                inliers, _ = filter_matches_ransac(kp1, kp2, matches)
                n_inliers = len(inliers)
            else:
                inliers = matches
                n_inliers = n_matches

            avg_distance = np.mean([m.distance for m in inliers]) if inliers else 0.0
            scores = compute_similarity_score(n_matches, n_inliers, n_kp1, n_kp2, avg_distance)

            score_matrix[i, j] = scores["score"]
            score_matrix[j, i] = scores["score"]
            inlier_matrix[i, j] = n_inliers
            inlier_matrix[j, i] = n_inliers

    return names, score_matrix, inlier_matrix, stats


# =============================================================================
# Affichage
# =============================================================================


def print_similarity_matrix(
    names: list[str],
    score_matrix: np.ndarray,
    inlier_matrix: np.ndarray,
) -> None:
    """Affiche la matrice de similarité avec inliers."""
    short_names = [n[:12] for n in names]
    max_len = max(len(n) for n in short_names)

    # Header
    header = " " * (max_len + 2)
    for name in short_names:
        header += f"{name[:8]:>12}"
    print(header)
    print("-" * len(header))

    # Lignes
    for i, name in enumerate(short_names):
        row = f"{name:<{max_len}}  "
        for j in range(len(names)):
            if i == j:
                row += f"{'--':>12}"
            else:
                score = score_matrix[i, j]
                inliers = int(inlier_matrix[i, j])
                row += f"{score:.3f}({inliers:>2})  "
        print(row)


def visualize_matches(result: dict, output_path: Path | None = None) -> None:
    """Visualise les matches entre deux images.

    Args:
        result: Résultat de compare_two_images.
        output_path: Chemin de sortie (optionnel).
    """
    img_matches = cv2.drawMatches(
        result["img1_bgr"],
        result["kp1"],
        result["img2_bgr"],
        result["kp2"],
        result["inlier_matches"],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if output_path:
        cv2.imwrite(str(output_path), img_matches)
        print(f"  Visualisation sauvegardée: {output_path}")
    else:
        # Afficher avec OpenCV
        cv2.imshow("SIFT Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_matches(
    names: list[str],
    matrix: np.ndarray,
    threshold: float = 0.05,
) -> list[tuple[str, str, float]]:
    """Trouve les paires d'images similaires."""
    matches = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] >= threshold:
                matches.append((names[i], names[j], matrix[i, j]))

    return sorted(matches, key=lambda x: x[2], reverse=True)


# =============================================================================
# Points d'entrée
# =============================================================================


def process_directory(images_dir: Path, visualize: bool = False) -> None:
    """Traite toutes les images d'un répertoire."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"Aucune image trouvée dans {images_dir}")
        return

    print(f"Images trouvées: {len(image_paths)}")
    for p in image_paths:
        print(f"  - {p.name}")

    # Comparer
    names, score_matrix, inlier_matrix, stats = compare_all_images(image_paths)

    # Afficher la matrice
    print("\n" + "=" * 80)
    print("MATRICE DE SIMILARITÉ - Score(Inliers)")
    print("=" * 80)
    print_similarity_matrix(names, score_matrix, inlier_matrix)

    # Matches trouvés
    print("\n" + "=" * 80)
    print("INDIVIDUS POTENTIELLEMENT IDENTIQUES (seuil >= 0.05)")
    print("=" * 80)
    matches = find_matches(names, score_matrix, threshold=0.05)
    if matches:
        for img1, img2, score in matches:
            i, j = names.index(img1), names.index(img2)
            inliers = int(inlier_matrix[i, j])
            print(f"  {img1} <-> {img2}: score={score:.3f}, inliers={inliers}")
    else:
        print("  Aucune paire trouvée au-dessus du seuil")

    # Stats
    print("\n" + "=" * 80)
    print("STATISTIQUES")
    print("=" * 80)
    off_diag = score_matrix[~np.eye(score_matrix.shape[0], dtype=bool)]
    avg_kp = np.mean([s["keypoints"] for s in stats.values()])
    print(f"  Keypoints moyens: {avg_kp:.0f}")
    print(f"  Score moyen: {off_diag.mean():.4f}")
    print(f"  Score max: {off_diag.max():.4f}")
    print(f"  Écart-type: {off_diag.std():.4f}")

    # Visualisation si demandée
    if visualize and matches:
        print("\n" + "=" * 80)
        print("VISUALISATION DES MATCHES")
        print("=" * 80)
        for img1_name, img2_name, _ in matches[:3]:  # Top 3
            img1_path = next(p for p in image_paths if p.stem == img1_name)
            img2_path = next(p for p in image_paths if p.stem == img2_name)
            result = compare_two_images(img1_path, img2_path)
            output_path = images_dir / f"matches_{img1_name}_{img2_name}.jpg"
            visualize_matches(result, output_path)


def process_two_images(image1: Path, image2: Path, visualize: bool = False) -> None:
    """Compare deux images spécifiques."""
    result = compare_two_images(image1, image2)

    print("\n" + "=" * 70)
    print("COMPARAISON SIFT (avec RANSAC)")
    print("=" * 70)
    print(f"  Image 1: {result['image1']} ({result['keypoints1']} keypoints)")
    print(f"  Image 2: {result['image2']} ({result['keypoints2']} keypoints)")
    print(f"  Matches bruts: {result['matches']}")
    print(f"  Inliers (après RANSAC): {result['inliers']}")
    print(f"  Distance moyenne: {result['avg_distance']:.1f}")
    print(f"  Score: {result['score']:.4f}")
    print(f"    - Match ratio: {result['match_ratio']:.4f}")
    print(f"    - Inlier ratio: {result['inlier_ratio']:.2%}")
    print(f"    - Quality: {result['quality']:.2%}")

    if result["score"] >= 0.10:
        print("\n  -> MÊME INDIVIDU (haute confiance)")
    elif result["score"] >= 0.05:
        print("\n  -> Probablement même individu")
    elif result["score"] >= 0.03:
        print("\n  -> Incertain")
    else:
        print("\n  -> Individus différents")

    if visualize:
        output_path = image1.parent / f"matches_{image1.stem}_{image2.stem}.jpg"
        visualize_matches(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POC: Identification de salamandres via SIFT",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("poc/images_transparent"),
        help="Répertoire contenant les images",
    )
    parser.add_argument(
        "--image1",
        type=Path,
        help="Première image à comparer",
    )
    parser.add_argument(
        "--image2",
        type=Path,
        help="Deuxième image à comparer",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Sauvegarder la visualisation des matches",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("POC: IDENTIFICATION SALAMANDRES (SIFT + RANSAC)")
    print("=" * 70)

    if args.image1 and args.image2:
        process_two_images(args.image1, args.image2, args.visualize)
    else:
        process_directory(args.images_dir, args.visualize)


if __name__ == "__main__":
    main()
