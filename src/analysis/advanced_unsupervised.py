"""Two-dimensional K-means and local energy-adaptive thresholds."""
from typing import Any, Dict, Tuple
import numpy as np

def _validate_iterations(max_iter, tol):
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")

def cluster_multifeature_2d(
    periodicity_vals: np.ndarray,
    zcr_vals: np.ndarray,
    method: str = "acf",
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform 2D K-Means clustering on the joint space [Periodicity, ZCR].
    Automatically identifies which cluster is Voiced based on acoustic properties:
    - Voiced: High periodicity (ACF/Cepstrum) or Low dip (AMDF), and LOW ZCR.
    - Unvoiced: Low periodicity (or High dip) and HIGH ZCR.

    Both clustering and V/UV mapping use standardized features. Acoustic score
    ties prefer stronger periodicity; equal-distance observations are UV.
    Empty centers are reseeded at a distant point. Unidentifiable single groups
    return UV with status='degenerate' instead of manufacturing a second class.

    Returns:
        labels: 1D array of predicted labels ('v' or 'uv').
        model_info: Centroids and cluster statistics.
    """
    _validate_iterations(max_iter, tol)
    method = method.lower()
    if method not in ("acf", "amdf", "cepstrum"):
        raise ValueError("Unknown periodicity method")
    periodicity_vals = np.asarray(periodicity_vals, dtype=float)
    zcr_vals = np.asarray(zcr_vals, dtype=float)
    if periodicity_vals.ndim != 1 or periodicity_vals.shape != zcr_vals.shape:
        raise ValueError("Periodicity and ZCR must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(periodicity_vals)) or not np.all(np.isfinite(zcr_vals)):
        raise ValueError("Clustering features must be finite")
    n = len(periodicity_vals)
    if n == 0:
        return np.array([], dtype=object), {"status": "empty"}

    # Feature matrix: shape (N, 2)
    # Only constant features use scale 1; an absolute floor would break invariance
    # when e.g. ZCR is expressed in a very small unit.
    p_mean, p_std = np.mean(periodicity_vals), np.std(periodicity_vals)
    z_mean, z_std = np.mean(zcr_vals), np.std(zcr_vals)
    p_std = float(p_std) if p_std > 0 else 1.0
    z_std = float(z_std) if z_std > 0 else 1.0

    norm_p = (periodicity_vals - p_mean) / p_std
    norm_z = (zcr_vals - z_mean) / z_std
    X = np.column_stack([norm_p, norm_z])

    # Smart initialization of 2 centroids:
    # Voiced candidate (high periodicity / low ZCR), Unvoiced candidate (low periodicity / high ZCR)
    if method == "amdf":
        # Voiced has low D_min (low norm_p) and low ZCR (low norm_z)
        c_v_init = np.array([np.percentile(norm_p, 20), np.percentile(norm_z, 20)])
        c_uv_init = np.array([np.percentile(norm_p, 80), np.percentile(norm_z, 80)])
    else:
        # ACF & Cepstrum: Voiced has high periodicity (high norm_p) and low ZCR (low norm_z)
        c_v_init = np.array([np.percentile(norm_p, 80), np.percentile(norm_z, 20)])
        c_uv_init = np.array([np.percentile(norm_p, 20), np.percentile(norm_z, 80)])

    c0 = np.copy(c_uv_init)
    c1 = np.copy(c_v_init)

    converged = False
    for iteration in range(max_iter):
        dist0 = np.sum((X - c0) ** 2, axis=1)
        dist1 = np.sum((X - c1) ** 2, axis=1)
        cluster1_mask = dist1 < dist0

        pts0 = X[~cluster1_mask]
        pts1 = X[cluster1_mask]

        # Reseed an empty center at the point furthest from the occupied center.
        new_c0 = np.mean(pts0, axis=0) if len(pts0) > 0 else X[int(np.argmax(dist1))]
        new_c1 = np.mean(pts1, axis=0) if len(pts1) > 0 else X[int(np.argmax(dist0))]

        shift = max(np.linalg.norm(new_c0 - c0), np.linalg.norm(new_c1 - c1))
        c0, c1 = new_c0, new_c1

        if shift < tol:
            converged = True
            break

    # Determine which centroid is Voiced
    # Map centroids back to original scale
    c0_orig = np.array([c0[0] * p_std + p_mean, c0[1] * z_std + z_mean])
    c1_orig = np.array([c1[0] * p_std + p_mean, c1[1] * z_std + z_mean])

    # Acoustic scores must use the same standardized space as K-Means.
    # On a score tie, prefer stronger periodicity, then lower ZCR, then index 0.
    direction = -1.0 if method == "amdf" else 1.0
    def acoustic_key(center, index):
        periodicity = direction * center[0]
        return (round(float(periodicity - center[1]), 12),
                round(float(periodicity), 12), -float(center[1]), -index)
    v_idx = max(range(2), key=lambda index: acoustic_key((c0, c1)[index], index))

    dist0 = np.sum((X - c0) ** 2, axis=1)
    dist1 = np.sum((X - c1) ** 2, axis=1)
    # Equal-distance points are UV, independent of the voiced center's index.
    distance_ties = np.isclose(dist0, dist1, rtol=1e-12, atol=1e-12)
    is_v = ((dist0 < dist1) if v_idx == 0 else (dist1 < dist0)) & ~distance_ties
    occupied = np.bincount((dist1 < dist0).astype(int), minlength=2)
    degenerate = np.allclose(c0, c1, rtol=0, atol=1e-12) or np.any(occupied == 0)
    if degenerate:
        # A single distinguishable group gives no evidence for a V/UV split.
        is_v[:] = False

    pred_labels = np.where(is_v, "v", "uv")
    centroid_v = c0_orig if v_idx == 0 else c1_orig
    centroid_uv = c1_orig if v_idx == 0 else c0_orig

    return pred_labels, {
        "centroid_v": centroid_v,
        "centroid_uv": centroid_uv,
        "centroid_v_normalized": (c0 if v_idx == 0 else c1).copy(),
        "centroid_uv_normalized": (c1 if v_idx == 0 else c0).copy(),
        "status": "degenerate" if degenerate else "ok",
        "converged": converged,
        "num_iter": iteration + 1,
        "p_mean": p_mean,
        "p_std": p_std,
        "z_mean": z_mean,
        "z_std": z_std,
    }

def compute_sliding_window_adaptive_threshold(
    values: np.ndarray,
    ste: np.ndarray,
    global_threshold: float,
    window_frames: int = 50,
    alpha: float = 0.35,
) -> np.ndarray:
    """Compute frame-dependent dynamic threshold T(t) using a sliding temporal window.
    Interpolates between global threshold and local median/statistics:
        T(t) = (1 - alpha) * T_global + alpha * T_local(t)

    Even windows include one more past frame than future frame; window 50
    uses [i-25, i+25), clipped at file boundaries without padding. The blend
    weight is alpha modulated by relative STE, not an estimated SNR.

    Args:
        values: 1D array of extracted periodicity metrics.
        ste: 1D array of Short-Time Energy.
        global_threshold: Baseline global threshold (e.g. Otsu or Median 50%).
        window_frames: Length of sliding window in frames (50 frames ~ 500ms).
        alpha: Maximum weight given to local context, modulated by STE.

    Returns:
        np.ndarray: Array of dynamic thresholds T(t) for each frame.
    """
    values, ste = np.asarray(values, dtype=float), np.asarray(ste, dtype=float)
    if values.ndim != 1 or values.shape != ste.shape:
        raise ValueError("Periodicity and STE must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ste)) or np.any(ste < 0):
        raise ValueError("Features must be finite and STE must be non-negative")
    if not isinstance(window_frames, (int, np.integer)) or window_frames < 1:
        raise ValueError("window_frames must be a positive integer")
    if not np.isfinite(global_threshold) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Threshold must be finite and alpha must be between 0 and 1")
    n = len(values)
    if n == 0:
        return np.array([])

    t_dynamic = np.full(n, global_threshold, dtype=np.float64)
    left = window_frames // 2
    right = window_frames - left - 1

    # Relative energy weighting: frames with higher local energy adapt more strongly
    ste_clean = np.copy(ste)
    max_ste = np.max(ste_clean) if len(ste_clean) > 0 else 1.0
    norm_ste = ste_clean / (max_ste + 1e-12)

    for i in range(n):
        start = max(0, i - left)
        end = min(n, i + right + 1)
        local_vals = values[start:end]

        local_med = float(np.median(local_vals))
        local_alpha = alpha * min(1.0, float(np.mean(norm_ste[start:end])) * 2.0 + 0.2)
        t_dynamic[i] = (1.0 - local_alpha) * global_threshold + local_alpha * local_med

    return t_dynamic
