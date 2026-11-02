"""Otsu and two-cluster one-dimensional threshold fitting."""
import numpy as np

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
    if k != 2:
        raise ValueError("Only two clusters are supported")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not isinstance(max_iter, (int, np.integer)) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Invalid K-means input or iteration settings")
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


def compute_all_unsupervised_thresholds(values, n_bins=100):
    return {'otsu': compute_otsu_threshold(values, n_bins=n_bins),
            'kmeans': compute_kmeans_threshold(values)}
