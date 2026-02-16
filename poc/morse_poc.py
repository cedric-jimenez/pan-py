"""POC: Salamander identification via yellow pattern 'morse code'.

Human method:
  1. Segment yellow spots on black body
  2. Read the pattern head-to-tail like morse code (segment, gap, segment, gap)
  3. Look for distinctive spot shapes (hooks, angles)
  4. Match: same morse code + 2-3 distinctive shapes = same individual

This POC validates steps 1-3: yellow segmentation, medial axis, 1D projection.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

IMAGE_DIR = Path(__file__).parent / "images"

# --- Step 1: Yellow segmentation ---

def segment_spots(img_bgr: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """Segment light spots (yellow/cream/white) on the dark salamander body.

    Uses brightness within the body mask — spots are significantly brighter
    than the black body regardless of their exact hue.

    Returns binary mask (255 = spot, 0 = not spot).
    """
    # Convert to LAB for perceptual lightness
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]  # 0-255

    # Only consider pixels inside the body
    body_pixels = lightness[body_mask > 0]
    if len(body_pixels) == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    # Adaptive threshold: spots are bright outliers on the dark body
    # Use Otsu within the body to find the dark/light split
    _, otsu_mask = cv2.threshold(body_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = _

    # Apply the threshold to the full image, masked to body
    mask = ((lightness > otsu_thresh) & (body_mask > 0)).astype(np.uint8) * 255

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def segment_body(img_bgr: np.ndarray) -> np.ndarray:
    """Segment the salamander body (non-background).

    Background is gray (~128). Body is black + yellow spots.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Background is ~128 gray. Body is darker (<80) or has yellow (bright in some channels)
    # Use distance from background gray
    bg_value = np.median(gray[gray > 100])  # estimate background
    diff = np.abs(gray.astype(float) - bg_value)
    body_mask = (diff > 30).astype(np.uint8) * 255

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Keep only largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(body_mask)
    if n_labels <= 1:
        return body_mask
    # Component 0 is background
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    body_mask = ((labels == largest) * 255).astype(np.uint8)
    return body_mask


# --- Step 2: Medial axis / skeleton ---

def compute_medial_axis(body_mask: np.ndarray, n_points: int = 150) -> np.ndarray:
    """Compute an ordered curved medial axis using the distance transform ridge.

    1. Compute distance transform of body mask
    2. Find ridge points (local maxima = centers of maximal inscribed circles)
    3. Order them by tracing the ridge from one end to the other
    4. Smooth and resample to n_points
    """
    from collections import deque

    dist = cv2.distanceTransform(body_mask, cv2.DIST_L2, 5)

    # Ridge = pixels where distance is a local maximum in at least one direction
    # Use a threshold: ridge pixels have dist > some fraction of local neighborhood max
    kernel_size = 7
    local_max = cv2.dilate(dist, np.ones((kernel_size, kernel_size)))
    # Ridge: close to local max AND inside body AND not too thin
    min_width = 3.0  # minimum body half-width to consider
    ridge_mask = ((dist > 0.7 * local_max) & (dist >= min_width) & (body_mask > 0))

    ridge_points = np.argwhere(ridge_mask)
    if len(ridge_points) < 5:
        # Fallback: use body centroid line
        return _fallback_pca_axis(body_mask, n_points)

    # Order ridge points by tracing from one end
    point_set = set(map(tuple, ridge_points))

    def bfs_farthest(start):
        visited = {start: None}
        queue = deque([start])
        last = start
        while queue:
            node = queue.popleft()
            last = node
            r, c = node
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    nbr = (r + dr, c + dc)
                    if nbr in point_set and nbr not in visited:
                        visited[nbr] = node
                        queue.append(nbr)
        return last, visited

    start = tuple(ridge_points[0])
    end_a, _ = bfs_farthest(start)
    end_b, parents = bfs_farthest(end_a)

    # Trace path
    path = []
    node = end_b
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()

    if len(path) < 5:
        return _fallback_pca_axis(body_mask, n_points)

    path_arr = np.array(path, dtype=float)

    # Smooth the path (remove zigzag from ridge detection)
    from scipy.ndimage import uniform_filter1d
    path_arr[:, 0] = uniform_filter1d(path_arr[:, 0], size=min(15, len(path_arr)//3))
    path_arr[:, 1] = uniform_filter1d(path_arr[:, 1], size=min(15, len(path_arr)//3))

    # Resample to n_points equidistant points
    diffs = np.diff(path_arr, axis=0)
    seg_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < 1:
        return _fallback_pca_axis(body_mask, n_points)

    from scipy.interpolate import interp1d
    target_dists = np.linspace(0, total_length, n_points)
    interp_r = interp1d(cumulative, path_arr[:, 0], kind='linear')
    interp_c = interp1d(cumulative, path_arr[:, 1], kind='linear')
    resampled = np.column_stack([interp_r(target_dists), interp_c(target_dists)])

    # Ensure head (topmost point) is first
    if resampled[0, 0] > resampled[-1, 0]:
        resampled = resampled[::-1]
    return resampled


def _fallback_pca_axis(body_mask: np.ndarray, n_points: int) -> np.ndarray:
    """Fallback: PCA-based straight axis when ridge detection fails."""
    body_points = np.argwhere(body_mask > 0)
    if len(body_points) < 10:
        return np.array([])

    mean = body_points.mean(axis=0)
    centered = body_points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    projections = centered @ principal
    proj_min, proj_max = projections.min(), projections.max()
    bin_edges = np.linspace(proj_min, proj_max, n_points + 1)

    medial_points = []
    for k in range(n_points):
        mask_bin = (projections >= bin_edges[k]) & (projections < bin_edges[k + 1])
        if mask_bin.sum() == 0:
            continue
        medial_points.append(body_points[mask_bin].mean(axis=0))

    if not medial_points:
        return np.array([])

    result = np.array(medial_points)
    if result[0, 0] > result[-1, 0]:
        result = result[::-1]
    return result


# --- Step 3: Project yellow onto axis → 1D morse signal ---

TAIL_WIDTH_THRESHOLD = 8  # body narrower than this = tail (pixels)


def project_to_morse(
    ordered_points: np.ndarray,
    spot_mask: np.ndarray,
    body_mask: np.ndarray,
) -> np.ndarray:
    """Project spot presence onto the ordered skeleton axis.

    For each skeleton point, samples a perpendicular cross-section
    across the body and computes the ratio of spot pixels to body pixels.
    Truncates the signal where the body becomes too thin (tail).
    """
    if len(ordered_points) < 3:
        return np.array([])

    h, w = spot_mask.shape
    signal = []
    widths = []
    half_width = 40  # max perpendicular reach in pixels

    for i in range(len(ordered_points)):
        r, c = ordered_points[i]

        # Compute local tangent direction
        i0 = max(0, i - 5)
        i1 = min(len(ordered_points) - 1, i + 5)
        dr = ordered_points[i1][0] - ordered_points[i0][0]
        dc = ordered_points[i1][1] - ordered_points[i0][1]
        length = max(np.sqrt(dr**2 + dc**2), 1e-6)
        nr, nc = -dc / length, dr / length

        spot_count = 0
        body_count = 0
        for t in range(-half_width, half_width + 1):
            sr = int(round(r + t * nr))
            sc = int(round(c + t * nc))
            if 0 <= sr < h and 0 <= sc < w:
                if body_mask[sr, sc] > 0:
                    body_count += 1
                    if spot_mask[sr, sc] > 0:
                        spot_count += 1

        widths.append(body_count)
        signal.append(spot_count / max(body_count, 1))

    # Truncate tail: find where body width drops below threshold persistently
    # Use a sliding window to avoid single narrow points cutting the signal
    widths_arr = np.array(widths)
    smoothed_widths = np.convolve(widths_arr, np.ones(5)/5, mode='same')

    # Find the last position where body is wide enough (from the tail end)
    body_region = smoothed_widths >= TAIL_WIDTH_THRESHOLD
    if not body_region.any():
        return np.array(signal)

    # Keep from first wide point to last wide point
    first = np.argmax(body_region)
    last = len(body_region) - 1 - np.argmax(body_region[::-1])
    truncated = np.array(signal[first:last+1])

    return truncated


def normalize_signal(signal: np.ndarray, target_len: int = 200) -> np.ndarray:
    """Normalize signal to fixed length. Minimal smoothing to preserve detail."""
    if len(signal) == 0:
        return signal
    from scipy.interpolate import interp1d
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, signal, kind='linear')
    resampled = f(x_new)
    # Light smoothing (window=3) just to remove single-pixel noise
    kernel = np.ones(3) / 3
    return np.convolve(resampled, kernel, mode='same')


# --- Step 4: Compare morse signals ---

def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """DTW distance between two 1D signals with Sakoe-Chiba band.

    Uses a warping window to prevent pathological alignments
    and speed up computation.
    """
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return float('inf')
    # Sakoe-Chiba band: allow at most 20% warping
    window = max(10, int(0.2 * max(n, m)))
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (s1[i-1] - s2[j-1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return np.sqrt(dtw[n, m] / (n + m))


def dtw_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """DTW-based similarity score (0 to 1).

    Tries both orientations and returns the best score.
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    d_fwd = dtw_distance(s1, s2)
    d_rev = dtw_distance(s1, s2[::-1])
    d = min(d_fwd, d_rev)
    # Convert distance to similarity: exp(-d) or 1/(1+d)
    return float(np.exp(-d * 5))  # scale factor to spread scores


def correlation_score(s1: np.ndarray, s2: np.ndarray) -> float:
    """Normalized cross-correlation between two signals.

    Tries both orientations (forward and reversed) and returns the
    maximum, since head/tail orientation may differ between images.
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    s1z = s1 - s1.mean()
    s2z = s2 - s2.mean()
    norm = np.linalg.norm(s1z) * np.linalg.norm(s2z)
    if norm < 1e-8:
        return 0.0
    corr_fwd = float(np.dot(s1z, s2z) / norm)
    # Try reversed orientation
    s2_rev = s2[::-1]
    s2rz = s2_rev - s2_rev.mean()
    corr_rev = float(np.dot(s1z, s2rz) / norm)
    return max(corr_fwd, corr_rev)


# --- Step 5: Blob shape matching ---

MIN_BLOB_AREA = 50  # minimum blob area in pixels to consider
N_FOURIER = 12  # number of Fourier descriptor coefficients
TOP_N_BLOBS = 7  # max blobs to consider for matching


def fourier_descriptors(contour: np.ndarray, n: int = N_FOURIER) -> np.ndarray:
    """Compute Fourier descriptors of contour boundary.

    Rotation-invariant (magnitudes), scale-invariant (normalized by 1st coeff).
    Better than Hu moments for concave shapes (hooks, angles).
    """
    pts = contour.squeeze()
    if len(pts) < 5:
        return np.zeros(n)
    z = pts[:, 0] + 1j * pts[:, 1]
    coeffs = np.fft.fft(z)
    magnitudes = np.abs(coeffs)
    # Normalize by 1st non-DC coefficient for scale invariance
    if magnitudes[1] > 1e-8:
        magnitudes = magnitudes / magnitudes[1]
    # Skip DC (index 0), take first n
    if len(magnitudes) < n + 1:
        result = np.zeros(n)
        result[:len(magnitudes) - 1] = magnitudes[1:]
        return result
    return magnitudes[1:n + 1]


def extract_blobs(spot_mask: np.ndarray, body_mask: np.ndarray, medial_axis: np.ndarray):
    """Extract yellow blobs with shape descriptors.

    Returns a list of dicts with:
      - hu_moments: 7 Hu moment invariants (rotation/scale invariant)
      - circularity: 4π × area / perimeter² (1.0 = perfect circle, <1 = irregular)
      - area_ratio: blob area / body area (size-invariant)
      - axis_position: normalized position along body axis (0=head, 1=tail)
      - contour: the raw contour for debug
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spot_mask)

    body_area = max(int((body_mask > 0).sum()), 1)
    blobs = []

    for i in range(1, n_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA:
            continue

        # Extract contour of this blob
        blob_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)

        # Hu moments (log-transformed, rotation/scale invariant)
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        # Log-transform for better numerical stability
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # Circularity
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        # Solidity: area / convex hull area (low = concave/hooked shape)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0

        # Convexity defects: captures hooks, angles, concavities
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        defects_raw = cv2.convexityDefects(contour, hull_idx) if len(hull_idx) > 2 else None
        if defects_raw is not None:
            depths = defects_raw[:, 0, 3] / 256.0  # convert fixed-point to float
            # Normalize depths by sqrt(area) for scale invariance
            scale = max(np.sqrt(area), 1.0)
            norm_depths = np.sort(depths / scale)[::-1]  # descending
            n_defects = len(norm_depths)
            # Keep top 5 deepest defects as descriptor
            defect_desc = np.zeros(5)
            defect_desc[:min(5, len(norm_depths))] = norm_depths[:5]
            mean_defect_depth = float(norm_depths.mean()) if len(norm_depths) > 0 else 0.0
        else:
            n_defects = 0
            defect_desc = np.zeros(5)
            mean_defect_depth = 0.0

        # Fourier descriptors of contour boundary
        fd = fourier_descriptors(contour)

        # Position along body axis
        cr, cc = centroids[i]  # centroid (row, col) — note: connectedComponents returns (x,y)
        blob_center = np.array([cc, cr])  # (row, col) format
        if len(medial_axis) > 0:
            dists = np.linalg.norm(medial_axis - blob_center, axis=1)
            nearest_idx = np.argmin(dists)
            axis_pos = nearest_idx / max(len(medial_axis) - 1, 1)
        else:
            axis_pos = 0.5

        blobs.append({
            'hu_moments': hu_log,
            'fourier': fd,
            'circularity': circularity,
            'solidity': solidity,
            'n_defects': n_defects,
            'defect_desc': defect_desc,
            'mean_defect_depth': mean_defect_depth,
            'area_ratio': area / body_area,
            'axis_position': axis_pos,
            'area': area,
            'contour': contour,
            'center': blob_center,
        })

    # Sort by area (largest first)
    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs


def blob_shape_distance(b1: dict, b2: dict) -> float:
    """Distance between two blob shape descriptors.

    Combines Hu moment distance + circularity difference.
    """
    hu_dist = np.linalg.norm(b1['hu_moments'] - b2['hu_moments'])
    circ_dist = abs(b1['circularity'] - b2['circularity'])
    return hu_dist + circ_dist * 2


def _match_blobs_directed(blobs_src: list, blobs_dst: list, max_position_diff: float) -> float:
    """Directed blob matching: for each blob in src, find best match in dst."""
    if not blobs_src or not blobs_dst:
        return 0.0

    matched_scores = []

    for b1 in blobs_src:
        best_dist = float('inf')
        for b2 in blobs_dst:
            pos_diff = min(
                abs(b1['axis_position'] - b2['axis_position']),
                abs(b1['axis_position'] - (1 - b2['axis_position'])),
            )
            if pos_diff > max_position_diff:
                continue
            size_ratio = max(b1['area_ratio'], 1e-6) / max(b2['area_ratio'], 1e-6)
            if size_ratio > 5 or size_ratio < 0.2:
                continue
            d = blob_shape_distance(b1, b2)
            best_dist = min(best_dist, d)

        if best_dist < float('inf'):
            matched_scores.append(np.exp(-best_dist * 0.3))

    if not matched_scores:
        return 0.0
    return float(np.mean(matched_scores))


def match_blobs(blobs1: list, blobs2: list, max_position_diff: float = 0.3) -> float:
    """Symmetric blob matching: average of both directions.

    score = (match(A→B) + match(B→A)) / 2
    This ensures score(A,B) == score(B,A).
    """
    fwd = _match_blobs_directed(blobs1, blobs2, max_position_diff)
    bwd = _match_blobs_directed(blobs2, blobs1, max_position_diff)
    return (fwd + bwd) / 2


# --- Piste 1: Convexity defects descriptor ---

def blob_shape_distance_defects(b1: dict, b2: dict) -> float:
    """Shape distance using Hu moments + circularity + convexity defects.

    Convexity defects directly capture hooks, angles, concavities.
    """
    hu_dist = np.linalg.norm(b1['hu_moments'] - b2['hu_moments'])
    circ_dist = abs(b1['circularity'] - b2['circularity'])
    defect_dist = np.linalg.norm(b1['defect_desc'] - b2['defect_desc'])
    solid_dist = abs(b1['solidity'] - b2['solidity'])
    return hu_dist + circ_dist * 2 + defect_dist * 3 + solid_dist * 2


def _match_blobs_directed_defects(
    blobs_src: list, blobs_dst: list, max_position_diff: float,
) -> float:
    """Piste 1: matching with convexity defects distance."""
    if not blobs_src or not blobs_dst:
        return 0.0
    matched_scores = []
    for b1 in blobs_src:
        best_dist = float('inf')
        for b2 in blobs_dst:
            pos_diff = min(
                abs(b1['axis_position'] - b2['axis_position']),
                abs(b1['axis_position'] - (1 - b2['axis_position'])),
            )
            if pos_diff > max_position_diff:
                continue
            size_ratio = max(b1['area_ratio'], 1e-6) / max(b2['area_ratio'], 1e-6)
            if size_ratio > 5 or size_ratio < 0.2:
                continue
            d = blob_shape_distance_defects(b1, b2)
            best_dist = min(best_dist, d)
        if best_dist < float('inf'):
            matched_scores.append(np.exp(-best_dist * 0.3))
    if not matched_scores:
        return 0.0
    return float(np.mean(matched_scores))


def match_blobs_defects(blobs1: list, blobs2: list, max_position_diff: float = 0.3) -> float:
    """Piste 1: Symmetric matching with convexity defects."""
    fwd = _match_blobs_directed_defects(blobs1, blobs2, max_position_diff)
    bwd = _match_blobs_directed_defects(blobs2, blobs1, max_position_diff)
    return (fwd + bwd) / 2


# --- Piste 2: Non-round blob filter ---

def _filter_non_round(blobs: list, max_circularity: float = 0.7) -> list:
    """Keep only non-round blobs (circularity below threshold)."""
    return [b for b in blobs if b['circularity'] < max_circularity]


def match_blobs_non_round(blobs1: list, blobs2: list, max_position_diff: float = 0.3) -> float:
    """Piste 2: Only match non-round (distinctive) blobs."""
    filtered1 = _filter_non_round(blobs1)
    filtered2 = _filter_non_round(blobs2)
    if not filtered1 or not filtered2:
        return 0.0
    fwd = _match_blobs_directed(filtered1, filtered2, max_position_diff)
    bwd = _match_blobs_directed(filtered2, filtered1, max_position_diff)
    return (fwd + bwd) / 2


# --- Piste 3: Hyperparameter grid search ---

def _match_blobs_param(
    blobs1: list, blobs2: list,
    decay: float, max_pos: float, size_lo: float, size_hi: float,
) -> float:
    """Parameterized blob matching for grid search."""
    def _directed(src, dst):
        if not src or not dst:
            return 0.0
        matched = []
        for b1 in src:
            best_dist = float('inf')
            for b2 in dst:
                pos_diff = min(
                    abs(b1['axis_position'] - b2['axis_position']),
                    abs(b1['axis_position'] - (1 - b2['axis_position'])),
                )
                if pos_diff > max_pos:
                    continue
                ratio = max(b1['area_ratio'], 1e-6) / max(b2['area_ratio'], 1e-6)
                if ratio > size_hi or ratio < size_lo:
                    continue
                d = blob_shape_distance(b1, b2)
                best_dist = min(best_dist, d)
            if best_dist < float('inf'):
                matched.append(np.exp(-best_dist * decay))
        return float(np.mean(matched)) if matched else 0.0

    return (_directed(blobs1, blobs2) + _directed(blobs2, blobs1)) / 2


# --- Combined: Piste 1 + 2 (defects on non-round blobs) ---

def match_blobs_defects_non_round(blobs1: list, blobs2: list, max_position_diff: float = 0.3) -> float:
    """Piste 1+2: Convexity defects distance on non-round blobs only."""
    filtered1 = _filter_non_round(blobs1)
    filtered2 = _filter_non_round(blobs2)
    if not filtered1 or not filtered2:
        return 0.0
    fwd = _match_blobs_directed_defects(filtered1, filtered2, max_position_diff)
    bwd = _match_blobs_directed_defects(filtered2, filtered1, max_position_diff)
    return (fwd + bwd) / 2


def combined_score(
    morse_dtw: float,
    blob_sim: float,
    morse_weight: float = 0.5,
) -> float:
    """Combine morse DTW similarity and blob matching into one score."""
    return morse_weight * morse_dtw + (1 - morse_weight) * blob_sim


# --- Visualization ---

def save_debug_image(
    img_bgr: np.ndarray,
    yellow_mask: np.ndarray,
    body_mask: np.ndarray,
    skeleton: np.ndarray,
    ordered_points: np.ndarray,
    output_path: Path,
) -> None:
    """Save a debug visualization."""
    vis = img_bgr.copy()

    # Yellow overlay in green
    yellow_overlay = np.zeros_like(vis)
    yellow_overlay[:, :, 1] = yellow_mask  # green channel
    vis = cv2.addWeighted(vis, 0.7, yellow_overlay, 0.3, 0)

    # Skeleton in red
    skel_mask = skeleton > 0
    vis[skel_mask] = [0, 0, 255]

    # Ordered points: color gradient blue→red along axis
    if len(ordered_points) > 0:
        for i, (r, c) in enumerate(ordered_points):
            t = i / max(len(ordered_points) - 1, 1)
            color = (int(255 * (1-t)), 0, int(255 * t))  # blue → red
            cv2.circle(vis, (c, r), 1, color, -1)

    cv2.imwrite(str(output_path), vis)


def save_morse_comparison(
    signals: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save a visual comparison of morse signals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, sig) in zip(axes, signals.items()):
        ax.fill_between(range(len(sig)), sig, alpha=0.7, color='gold')
        ax.set_ylim(0, 1)
        ax.set_ylabel(name, rotation=0, ha='right', fontsize=8)
        ax.set_yticks([])

    axes[-1].set_xlabel('Position along body axis (head → tail)')
    fig.suptitle('Morse Code Signals', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
    plt.close()


# --- Main ---

def process_image(img_path: Path, debug_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, list]:
    """Process one image: segment → axis → morse signal + blob descriptors.

    Returns (ordered_points, morse_signal, blobs).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read {img_path}")

    body = segment_body(img)
    yellow = segment_spots(img, body_mask=body)

    # Medial axis
    ordered = compute_medial_axis(body, n_points=150)
    morse = project_to_morse(ordered, yellow, body)
    normalized = normalize_signal(morse)

    # Blob extraction
    blobs = extract_blobs(yellow, body, ordered)

    if debug_dir:
        debug_dir.mkdir(exist_ok=True)
        stem = img_path.stem
        skel = np.zeros(body.shape, dtype=np.uint8)
        for r, c in ordered.astype(int):
            if 0 <= r < skel.shape[0] and 0 <= c < skel.shape[1]:
                skel[r, c] = 255
        save_debug_image(img, yellow, body, skel, ordered.astype(int), debug_dir / f"{stem}_debug.jpg")

    # Debug stats
    n_body = int((body > 0).sum())
    n_spots = int((yellow > 0).sum())
    spot_pct = n_spots / max(n_body, 1) * 100
    blob_areas = [f"{b['area']}" for b in blobs[:5]]
    print(f"body={n_body}px spots={n_spots}px ({spot_pct:.0f}%) "
          f"axis={len(ordered)} morse={len(morse)} "
          f"blobs={len(blobs)} top_areas=[{','.join(blob_areas)}]", end=" ")

    return ordered, normalized, blobs


def main():
    debug_dir = Path(__file__).parent / "morse_debug"

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    names = []
    signals = {}
    all_blobs = {}

    print("Processing images...")
    for p in image_paths:
        short = p.stem.replace("IMG_20251116_", "").replace("-segmented", "")
        print(f"  {short}...", end=" ", flush=True)
        try:
            _, signal, blobs = process_image(p, debug_dir)
            names.append(short)
            signals[short] = signal
            all_blobs[short] = blobs
            print(f"OK (signal length: {len(signal)})")
        except Exception as e:
            print(f"FAILED: {e}")

    # Save morse comparison chart
    save_morse_comparison(signals, debug_dir / "morse_comparison.png")
    print(f"\nDebug images saved to {debug_dir}/")

    # Compute pairwise scores
    name_list = list(signals.keys())
    n = len(name_list)

    for metric_name, metric_fn in [("CORRELATION", correlation_score), ("DTW SIMILARITY", dtw_similarity)]:
        print("\n" + "=" * 70)
        print(f"PAIRWISE {metric_name}")
        print("=" * 70)

        # Header
        print(f"{'':>12}", end="")
        for name in name_list:
            print(f"{name[:8]:>9}", end="")
        print()

        matrix = np.zeros((n, n))
        for i in range(n):
            print(f"{name_list[i][:12]:>12}", end="")
            for j in range(n):
                if i == j:
                    score = 1.0
                else:
                    score = metric_fn(signals[name_list[i]], signals[name_list[j]])
                matrix[i, j] = score
                print(f"{score:9.3f}", end="")
            print()

        # Analysis
        titine_indices = [i for i, nm in enumerate(name_list) if "titine" in nm]
        if len(titine_indices) == 2:
            ti, tj = titine_indices
            same_score = matrix[ti, tj]
            print(f"\n  Same pair (titine-1 vs titine-2): {same_score:.3f}")

        diff_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                both_titine = "titine" in name_list[i] and "titine" in name_list[j]
                if not both_titine:
                    diff_scores.append((name_list[i], name_list[j], matrix[i, j]))

        diff_vals = [s for _, _, s in diff_scores]
        print(f"  Different: min={min(diff_vals):.3f}, max={max(diff_vals):.3f}, mean={np.mean(diff_vals):.3f}")
        if len(titine_indices) == 2:
            gap = same_score - max(diff_vals)
            print(f"  Gap: {gap:+.3f} {'OK' if gap > 0 else 'OVERLAP'}")

        diff_scores.sort(key=lambda x: x[2], reverse=True)
        print(f"  Top 5 different: ", end="")
        print(", ".join(f"{n1[:6]}↔{n2[:6]}={s:.3f}" for n1, n2, s in diff_scores[:5]))

    # --- Blob matching + Combined scores ---
    def _print_matrix(title, score_fn):
        print("\n" + "=" * 70)
        print(f"PAIRWISE {title}")
        print("=" * 70)
        print(f"{'':>12}", end="")
        for name in name_list:
            print(f"{name[:8]:>9}", end="")
        print()

        matrix = np.zeros((n, n))
        for i in range(n):
            print(f"{name_list[i][:12]:>12}", end="")
            for j in range(n):
                if i == j:
                    score = 1.0
                else:
                    score = score_fn(name_list[i], name_list[j])
                matrix[i, j] = score
                print(f"{score:9.3f}", end="")
            print()

        titine_indices = [i for i, nm in enumerate(name_list) if "titine" in nm]
        if len(titine_indices) == 2:
            ti, tj = titine_indices
            same_score = matrix[ti, tj]
            print(f"\n  Same pair (titine-1 vs titine-2): {same_score:.3f}")

        diff_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                both_titine = "titine" in name_list[i] and "titine" in name_list[j]
                if not both_titine:
                    diff_scores.append((name_list[i], name_list[j], matrix[i, j]))
        diff_vals = [s for _, _, s in diff_scores]
        print(f"  Different: min={min(diff_vals):.3f}, max={max(diff_vals):.3f}, mean={np.mean(diff_vals):.3f}")
        if len(titine_indices) == 2:
            gap = same_score - max(diff_vals)
            print(f"  Gap: {gap:+.3f} {'OK' if gap > 0 else 'OVERLAP'}")
        diff_scores.sort(key=lambda x: x[2], reverse=True)
        print(f"  Top 5 different: ", end="")
        print(", ".join(f"{n1[:6]}↔{n2[:6]}={s:.3f}" for n1, n2, s in diff_scores[:5]))

    _print_matrix("BLOB v0 (baseline: Hu + circ)",
                  lambda a, b: match_blobs(all_blobs[a], all_blobs[b]))

    _print_matrix("PISTE 1: convexity defects",
                  lambda a, b: match_blobs_defects(all_blobs[a], all_blobs[b]))

    _print_matrix("PISTE 2: non-round only (circ<0.7)",
                  lambda a, b: match_blobs_non_round(all_blobs[a], all_blobs[b]))

    _print_matrix("PISTE 1+2: defects + non-round",
                  lambda a, b: match_blobs_defects_non_round(all_blobs[a], all_blobs[b]))

    # --- Combined grid search: circularity threshold + hyperparams + non-round filter ---
    print("\n" + "=" * 70)
    print("GRID SEARCH: non-round filter + hyperparams")
    print("=" * 70)
    print(f"{'circ':>5} {'decay':>6} {'max_pos':>8} {'size_lo':>8} {'size_hi':>8}  {'same':>6} {'max_diff':>9} {'gap':>7}")
    print("-" * 70)

    best_gap = -999
    best_params = None
    top_results = []

    for max_circ in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Pre-filter blobs for this circularity threshold
        filtered = {}
        for nm in name_list:
            if max_circ >= 1.0:
                filtered[nm] = all_blobs[nm]  # no filter
            else:
                filtered[nm] = [b for b in all_blobs[nm] if b['circularity'] < max_circ]

        for decay in [0.15, 0.2, 0.3, 0.4, 0.5]:
            for max_pos in [0.2, 0.3, 0.4, 0.5]:
                for size_lo, size_hi in [(0.1, 10), (0.2, 5), (0.3, 3)]:
                    scores_map = {}
                    for i in range(n):
                        for j in range(i + 1, n):
                            s = _match_blobs_param(
                                filtered[name_list[i]], filtered[name_list[j]],
                                decay, max_pos, size_lo, size_hi,
                            )
                            scores_map[(name_list[i], name_list[j])] = s

                    same_s = None
                    diff_vals = []
                    for (n1, n2), s in scores_map.items():
                        if "titine" in n1 and "titine" in n2:
                            same_s = s
                        else:
                            diff_vals.append(s)

                    if same_s is None or not diff_vals:
                        continue
                    gap = same_s - max(diff_vals)
                    top_results.append((gap, max_circ, decay, max_pos, size_lo, size_hi, same_s, max(diff_vals)))
                    if gap > best_gap:
                        best_gap = gap
                        best_params = (max_circ, decay, max_pos, size_lo, size_hi)

    # Print top 20 results
    top_results.sort(key=lambda x: x[0], reverse=True)
    for gap, mc, d, mp, sl, sh, ss, md in top_results[:20]:
        print(f"{mc:>5.1f} {d:>6.2f} {mp:>8.2f} {sl:>8.2f} {sh:>8.1f}  {ss:>6.3f} {md:>9.3f} {gap:>+7.3f} OK")

    mc, bd, bp, bsl, bsh = best_params
    print(f"\nBest: circ<{mc}, decay={bd}, max_pos={bp}, "
          f"size=({bsl},{bsh}) → gap={best_gap:+.3f}")

    # Show best params matrix
    best_filtered = {}
    for nm in name_list:
        if mc >= 1.0:
            best_filtered[nm] = all_blobs[nm]
        else:
            best_filtered[nm] = [b for b in all_blobs[nm] if b['circularity'] < mc]

    _print_matrix(
        f"BEST (circ<{mc}, d={bd}, pos={bp}, sz={bsl}-{bsh})",
        lambda a, b: _match_blobs_param(best_filtered[a], best_filtered[b], bd, bp, bsl, bsh),
    )


if __name__ == "__main__":
    main()
