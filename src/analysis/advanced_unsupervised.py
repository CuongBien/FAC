"""Advanced unsupervised Voiced/Unvoiced classification methods overcoming 1D histogram limitations:
1. 1D GMM-EM with fixed component priors (default 50/50)
2. Multi-feature 2D Unsupervised Clustering (Periodicity Metric + Zero Crossing Rate ZCR)
3. 2-State HMM with learned Gaussian emissions and fixed A/pi, reset at silence
4. Context-Adaptive Dynamic Thresholding (Sliding-window median/STE adaptation)
"""
from typing import Any, Dict, Tuple
import numpy as np
from scipy.stats import norm
from src.core.processing import voiced_segments

# ==============================================================================
# 1. 1D GMM-EM WITH FIXED COMPONENT PRIORS (DEFAULT 50/50)
# ==============================================================================

def _validate_iterations(max_iter, tol):
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")


def _gaussian_intersection(mu0, std0, mu1, std1, weight0=0.5, weight1=0.5):
    """Return a weighted-density crossing and an explicit fallback status.

    Prefer a root between the means, then the root nearest their midpoint.
    Unequal variances can produce two crossings; a single threshold only models
    one of those boundaries. A fallback is never reported as an exact crossing.
    """
    parameters = np.array([mu0, std0, mu1, std1, weight0, weight1], dtype=float)
    if not np.all(np.isfinite(parameters)) or min(std0, std1, weight0, weight1) <= 0:
        raise ValueError("Gaussian scales and weights must be positive and finite")
    midpoint = mu0 / 2.0 + mu1 / 2.0
    # Center x at the midpoint to reduce cancellation when means have an offset.
    m0, m1 = mu0 - midpoint, mu1 - midpoint
    var0, var1 = std0 ** 2, std1 ** 2
    a = 0.5 / var1 - 0.5 / var0
    b = m0 / var0 - m1 / var1
    c = m1 ** 2 / (2.0 * var1) - m0 ** 2 / (2.0 * var0)
    c += np.log(std1 / std0) + np.log(weight0 / weight1)
    if abs(a) <= 1e-12 * max(0.5 / var0, 0.5 / var1):
        if b == 0:
            status = "identical_components" if abs(c) < 1e-12 else "no_real_intersection"
            return float(midpoint), status
        roots = [-c / b]
    else:
        disc = b * b - 4.0 * a * c
        roundoff = 1e-12 * (b * b + abs(4.0 * a * c))
        if disc < -roundoff:
            return float(midpoint), "no_real_intersection"
        disc = max(0.0, disc)
        q = -0.5 * (b + np.copysign(np.sqrt(disc), b))
        roots = [q / a, c / q] if q != 0 else [-b / (2.0 * a)]
    roots = [float(r + midpoint) for r in roots if np.isfinite(r)]
    between = [r for r in roots if min(mu0, mu1) <= r <= max(mu0, mu1)]
    if not roots:
        return float(midpoint), "no_finite_intersection"
    chosen = min(between or roots, key=lambda r: (abs(r - midpoint), r))
    return chosen, "between_means" if between else "outside_means"


def compute_gmm_em_1d(
    values: np.ndarray,
    prior: float = 0.5,
    max_iter: int = 150,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Fit two Gaussians with fixed component weights and a variance floor.

    prior is the weight of the initially lower-mean component; the other weight
    is 1-prior. Component indices are not V/UV labels (AMDF has the reverse
    direction). Returned weights follow their components if means are reordered.
    EM uses log probabilities; convergence requires both parameter and likelihood
    stability. threshold_status distinguishes a true crossing from a fallback.
    """
    _validate_iterations(max_iter, tol)
    if not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("prior must be strictly between 0 and 1")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("GMM values must be one-dimensional")
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) < 4:
        mid = float(np.median(clean_vals)) if len(clean_vals) > 0 else 0.5
        return {
            "threshold": mid,
            "mu0": mid, "std0": 0.1,
            "mu1": mid, "std1": 0.1,
            "weight0": float(prior), "weight1": float(1.0 - prior),
            "log_likelihood": float(np.sum(norm.logpdf(clean_vals, loc=mid, scale=0.1))),
            "converged": False,
            "num_iter": 0,
            "threshold_status": "insufficient_data",
        }

    # Initialize centroids using 25th and 75th percentiles
    p25 = float(np.percentile(clean_vals, 25))
    p75 = float(np.percentile(clean_vals, 75))
    mu0, mu1 = min(p25, p75), max(p25, p75)
    std_all = float(np.std(clean_vals))
    std0 = max(std_all * 0.5, 1e-3)
    std1 = max(std_all * 0.5, 1e-3)

    w0 = prior
    w1 = 1.0 - prior
    converged = False
    prev_ll = -np.inf

    def log_density(mean0, scale0, mean1, scale1):
        log0 = np.log(w0) + norm.logpdf(clean_vals, loc=mean0, scale=scale0)
        log1 = np.log(w1) + norm.logpdf(clean_vals, loc=mean1, scale=scale1)
        return log0, np.logaddexp(log0, log1)

    for it in range(max_iter):
        log0, log_total = log_density(mu0, std0, mu1, std1)
        gamma0 = np.exp(log0 - log_total)
        gamma1 = 1.0 - gamma0

        # M-step: Keep the supplied component weights fixed.
        n0 = np.sum(gamma0)
        n1 = np.sum(gamma1)

        if n0 < 1e-4 or n1 < 1e-4:
            break

        new_mu0 = float(np.sum(gamma0 * clean_vals) / n0)
        new_mu1 = float(np.sum(gamma1 * clean_vals) / n1)

        new_std0 = float(np.sqrt(max(np.sum(gamma0 * (clean_vals - new_mu0) ** 2) / n0, 1e-6)))
        new_std1 = float(np.sqrt(max(np.sum(gamma1 * (clean_vals - new_mu1) ** 2) / n1, 1e-6)))

        # Check convergence
        shift = max(abs(new_mu0 - mu0), abs(new_mu1 - mu1),
                    abs(new_std0 - std0), abs(new_std1 - std1))
        mu0, mu1 = new_mu0, new_mu1
        std0, std1 = new_std0, new_std1

        ll = float(np.sum(log_density(mu0, std0, mu1, std1)[1]))
        if np.isfinite(prev_ll) and shift < tol and abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
            converged = True
            break
        prev_ll = ll

    # Ensure mu0 < mu1
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
        std0, std1 = std1, std0
        w0, w1 = w1, w0

    t_optimal, threshold_status = _gaussian_intersection(mu0, std0, mu1, std1, w0, w1)
    final_ll = float(np.sum(log_density(mu0, std0, mu1, std1)[1]))

    return {
        "threshold": float(t_optimal),
        "mu0": float(mu0),
        "std0": float(std0),
        "mu1": float(mu1),
        "std1": float(std1),
        "weight0": float(w0),
        "weight1": float(w1),
        "log_likelihood": final_ll,
        "threshold_status": threshold_status,
        "converged": converged,
        "num_iter": it + 1,
    }


# ==============================================================================
# 2. MULTI-FEATURE 2D CLUSTERING (PERIODICITY + ZERO CROSSING RATE ZCR)
# ==============================================================================

def compute_zcr(frame: np.ndarray) -> float:
    """Compute normalized Zero Crossing Rate for a 1D audio frame."""
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    # Replace zeros with 1 to avoid zero-product artifacts
    signs[signs == 0] = 1.0
    crossings = np.sum(np.abs(signs[1:] - signs[:-1])) / 2.0
    return float(crossings / (len(frame) - 1))


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


# ==============================================================================
# 3. 2-STATE HMM (EMISSION-ONLY EM WITH FIXED A/PI + SEGMENT VITERBI)
# ==============================================================================

def _hmm_posteriors(log_emissions, A, pi, segments):
    """Forward/backward with a fresh pi at each half-open active segment."""
    gamma = np.zeros_like(log_emissions)
    log_A, log_pi = np.log(A), np.log(pi)
    total_ll = 0.0
    for start, end in segments:
        emissions = log_emissions[start:end]
        forward = np.zeros_like(emissions)
        backward = np.zeros_like(emissions)
        forward[0] = log_pi + emissions[0]
        for t in range(1, len(emissions)):
            forward[t] = emissions[t] + np.logaddexp.reduce(forward[t - 1, :, None] + log_A, axis=0)
        ll = float(np.logaddexp.reduce(forward[-1]))
        total_ll += ll
        for t in range(len(emissions) - 2, -1, -1):
            backward[t] = np.logaddexp.reduce(log_A + emissions[t + 1] + backward[t + 1], axis=1)
        posterior = np.exp(forward + backward - ll)
        gamma[start:end] = posterior / posterior.sum(axis=1, keepdims=True)
    return gamma, total_ll


def _hmm_viterbi(log_emissions, A, pi, segments):
    """Decode each segment independently; inactive positions retain state -1."""
    path = np.full(len(log_emissions), -1, dtype=int)
    log_A, log_pi = np.log(A), np.log(pi)
    for start, end in segments:
        emissions = log_emissions[start:end]
        scores = np.zeros_like(emissions)
        pointers = np.zeros_like(emissions, dtype=int)
        scores[0] = log_pi + emissions[0]
        for t in range(1, len(emissions)):
            totals = scores[t - 1, :, None] + log_A
            pointers[t] = np.argmax(totals, axis=0)
            scores[t] = np.max(totals, axis=0) + emissions[t]
        path[end - 1] = int(np.argmax(scores[-1]))
        for t in range(end - 2, start - 1, -1):
            path[t] = pointers[t + 1 - start, path[t + 1]]
    return path


def train_and_decode_2state_hmm(
    observations: np.ndarray,
    method: str = "acf",
    prior: float = 0.5,
    max_iter: int = 40,
    tol: float = 1e-4,
    active_mask: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Learn Gaussian emissions with fixed A/pi, then decode active segments.

    Pass the full frame timeline and active_mask so silence breaks both EM
    recurrences and Viterbi paths. Emissions are shared across active segments
    within the input file; A and pi are reset, never learned. prior weights the
    initially lower-mean state, not the semantic UV class for every method.
    Inactive frames return 'sil'; an unidentifiable constant input returns 'uv'.
    """
    _validate_iterations(max_iter, tol)
    if not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("prior must be strictly between 0 and 1")
    method = method.lower()
    if method not in ("acf", "amdf", "cepstrum"):
        raise ValueError("Unknown periodicity method")
    X = np.asarray(observations, dtype=float)
    if X.ndim != 1 or not np.all(np.isfinite(X)):
        raise ValueError("HMM observations must be a finite one-dimensional array")
    active = np.ones(len(X), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != X.shape:
        raise ValueError("active_mask must align with the full observation timeline")
    segments = list(voiced_segments(active))
    labels = np.full(len(X), "sil", dtype=object)
    if not segments:
        return labels, {"status": "no_active_frames", "segments": []}
    active_X = X[active]

    # Both A and pi stay fixed throughout emission-only EM.
    A = np.array([
        [0.85, 0.15],
        [0.15, 0.85]
    ], dtype=np.float64)

    pi = np.array([prior, 1.0 - prior], dtype=np.float64)

    p25, p75 = float(np.percentile(active_X, 25)), float(np.percentile(active_X, 75))
    mu = np.array([min(p25, p75), max(p25, p75)], dtype=np.float64)
    var = np.full(2, np.var(active_X) * 0.5 + 1e-4)
    def emissions(means, variances):
        return norm.logpdf(X[:, None], loc=means, scale=np.sqrt(variances))

    converged = False
    for it in range(max_iter):
        gamma, _ = _hmm_posteriors(emissions(mu, var), A, pi, segments)
        new_mu, new_var = mu.copy(), var.copy()
        for j in range(2):
            weights = gamma[active, j]
            g_sum = np.sum(weights)
            if g_sum > 1e-12:
                new_mu[j] = np.sum(weights * active_X) / g_sum
                new_var[j] = max(np.sum(weights * (active_X - new_mu[j]) ** 2) / g_sum, 1e-5)

        shift = max(np.max(np.abs(new_mu - mu)), np.max(np.abs(np.sqrt(new_var) - np.sqrt(var))))
        mu, var = new_mu, new_var

        if shift < tol:
            converged = True
            break

    # Determine which state corresponds to Voiced
    if method == "amdf":
        # Lower dip is Voiced
        v_state = 0 if mu[0] < mu[1] else 1
    else:
        # Higher peak is Voiced
        v_state = 1 if mu[1] > mu[0] else 0
    uv_state = 1 - v_state

    log_emissions = emissions(mu, var)
    path = _hmm_viterbi(log_emissions, A, pi, segments)
    identifiable = not np.isclose(mu[0], mu[1], rtol=0, atol=1e-12)
    labels[active] = np.where(path[active] == v_state, "v", "uv") if identifiable else "uv"
    _, final_ll = _hmm_posteriors(log_emissions, A, pi, segments)
    return labels, {
        "A": A,
        "pi": pi,
        "means": mu,
        "variances": var,
        "v_state": v_state,
        "segments": [(int(start), int(end)) for start, end in segments],
        "status": "ok" if identifiable else "degenerate",
        "converged": converged,
        "num_iter": it + 1,
        "log_likelihood": final_ll,
        "fixed_transitions": True,
        "mu_v": mu[v_state],
        "std_v": np.sqrt(var[v_state]),
        "mu_uv": mu[uv_state],
        "std_uv": np.sqrt(var[uv_state]),
    }


# ==============================================================================
# 4. CONTEXT-ADAPTIVE SLIDING-WINDOW DYNAMIC THRESHOLDING
# ==============================================================================

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
