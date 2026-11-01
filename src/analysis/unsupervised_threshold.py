"""Unsupervised thresholding methods for Voiced/Unvoiced classification.
Implements:
1. Otsu's Thresholding (Maximizing between-class variance sigma_B^2)
2. Bimodal Histogram Valley Detection (Smoothed histogram local minimum)
3. Median / 50th Percentile Thresholding (Direct exploitation of P(V) ~ P(UV) ~ 0.5)
4. 1D K-Means Clustering (Mid-point boundary between two cluster centroids)
"""
from typing import Dict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks



def compute_otsu_threshold(values: np.ndarray, n_bins: int = 100) -> float:
    """Find optimal threshold using Otsu's method on the empirical histogram.
    Maximizes between-class variance: sigma_B^2(T) = w0 * w1 * (mu0 - mu1)^2.

    Args:
        values: 1D array of extracted periodicity metrics from active speech frames.
        n_bins: Number of histogram bins.

    Returns:
        float: Optimal decision threshold T.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Otsu values must be one-dimensional")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) < 2:
        return float(np.mean(clean_vals)) if len(clean_vals) == 1 else 0.5
    if np.min(clean_vals) == np.max(clean_vals):
        # No two distinguishable groups; retain the observed value instead of
        # selecting the first bin of NumPy's artificially expanded histogram.
        return float(clean_vals[0])

    hist, bin_edges = np.histogram(clean_vals, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = np.sum(hist)
    if total == 0:
        return float(np.mean(clean_vals))

    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_centers)
    mu_t = mu[-1]

    # Between-class variance
    denom = omega * (1.0 - omega)
    valid = denom > 1e-12

    sigma_b_sq = np.zeros_like(omega)
    sigma_b_sq[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    best_idx = int(np.argmax(sigma_b_sq))
    # omega[i] includes bin i. Its boundary with the other class is the right
    # edge, not the center: returning the center splits a bin already assigned
    # to the lower class while optimizing sigma_b_sq.
    return float(bin_edges[best_idx + 1])


def compute_histogram_valley_threshold(
    values: np.ndarray, n_bins: int = 100, smooth_sigma: float = 2.0
) -> float:
    """Find the valley (local minimum) between two bimodal peaks in the smoothed histogram.

    Args:
        values: 1D array of extracted periodicity metrics.
        n_bins: Number of histogram bins.
        smooth_sigma: Standard deviation of 1D Gaussian kernel for histogram smoothing.

    Returns:
        float: Optimal threshold corresponding to the histogram valley.
    """
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) < 2:
        return float(np.mean(clean_vals)) if len(clean_vals) == 1 else 0.5

    hist, bin_edges = np.histogram(clean_vals, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Smooth histogram using Gaussian filter to eliminate micro-ripples
    smoothed = gaussian_filter1d(hist.astype(np.float64), sigma=smooth_sigma)

    # Detect local peaks
    peaks, _ = find_peaks(smoothed)

    if len(peaks) >= 2:
        # Sort detected peaks by height (descending) and select the two most prominent
        sorted_by_height = sorted(peaks, key=lambda idx: smoothed[idx], reverse=True)
        p1, p2 = sorted_by_height[0], sorted_by_height[1]
        p_left, p_right = min(p1, p2), max(p1, p2)

        # Ensure there is a gap between the peaks
        if p_right - p_left > 1:
            valley_offset = int(np.argmin(smoothed[p_left : p_right + 1]))
            valley_idx = p_left + valley_offset
            return float(bin_centers[valley_idx])

    # Fallback to Otsu threshold if bimodal structure is not sharply separated
    return compute_otsu_threshold(clean_vals, n_bins=n_bins)


def compute_median_threshold(values: np.ndarray) -> float:
    """Find threshold using the 50th percentile (Median), directly exploiting
    the prior assumption P(V) ~ P(UV) ~ 0.5 on active speech frames.

    Args:
        values: 1D array of extracted periodicity metrics.

    Returns:
        float: Median threshold T.
    """
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) == 0:
        return 0.5
    return float(np.median(clean_vals))


def compute_kmeans_threshold(
    values: np.ndarray, k: int = 2, max_iter: int = 100, tol: float = 1e-5
) -> float:
    """Find decision boundary using 1D K-Means clustering (K=2) and mid-point threshold.

    Args:
        values: 1D array of extracted periodicity metrics.
        k: Number of clusters (fixed to 2 for Voiced vs Unvoiced).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance for centroid movement.

    Returns:
        float: Boundary threshold T = (c0 + c1) / 2.
    """
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) < 2:
        return float(np.mean(clean_vals)) if len(clean_vals) == 1 else 0.5

    # Initialize centroids at 25th and 75th percentiles
    c0 = float(np.percentile(clean_vals, 25))
    c1 = float(np.percentile(clean_vals, 75))

    if np.isclose(c0, c1):
        c0 = float(np.min(clean_vals))
        c1 = float(np.max(clean_vals))

    for _ in range(max_iter):
        dist0 = np.abs(clean_vals - c0)
        dist1 = np.abs(clean_vals - c1)
        cluster0_mask = dist0 <= dist1

        pts0 = clean_vals[cluster0_mask]
        pts1 = clean_vals[~cluster0_mask]

        new_c0 = float(np.mean(pts0)) if len(pts0) > 0 else c0
        new_c1 = float(np.mean(pts1)) if len(pts1) > 0 else c1

        shift = max(abs(new_c0 - c0), abs(new_c1 - c1))
        c0, c1 = new_c0, new_c1

        if shift < tol:
            break

    c_low, c_high = min(c0, c1), max(c0, c1)
    return float((c_low + c_high) / 2.0)


def compute_all_unsupervised_thresholds(
    values: np.ndarray, n_bins: int = 100, smooth_sigma: float = 2.0
) -> Dict[str, float]:
    """Compute all 4 unsupervised thresholds for a given metric array.

    Returns:
        dict: {'otsu': float, 'valley': float, 'median': float, 'kmeans': float}
    """
    return {
        "otsu": compute_otsu_threshold(values, n_bins=n_bins),
        "valley": compute_histogram_valley_threshold(
            values, n_bins=n_bins, smooth_sigma=smooth_sigma
        ),
        "median": compute_median_threshold(values),
        "kmeans": compute_kmeans_threshold(values),
    }
