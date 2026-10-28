"""Advanced unsupervised Voiced/Unvoiced classification methods overcoming 1D histogram limitations:
1. 1D GMM-EM with fixed 50/50 prior (Continuous Gaussian Mixture, eliminates histogram binning artifacts)
2. Multi-feature 2D Unsupervised Clustering (Periodicity Metric + Zero Crossing Rate ZCR)
3. 2-State Unsupervised Hidden Markov Model (Baum-Welch EM + Global Viterbi Decoding)
4. Context-Adaptive Dynamic Thresholding (Sliding-window local SNR/energy adaptation)
"""
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm

from src.core.audio import load_wav, frame_signal, compute_ste
from src.core.lab_parser import parse_lab_file
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum


# ==============================================================================
# 1. 1D GMM-EM WITH FIXED 50/50 PRIOR (CONTINUOUS BAYESIAN INTERSECTION)
# ==============================================================================

def compute_gmm_em_1d(
    values: np.ndarray,
    prior: float = 0.5,
    max_iter: int = 150,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    """Fit a 2-component 1D Gaussian Mixture Model using Expectation-Maximization (EM)
    under the fixed prior constraint P(V) = P(UV) = prior (default 0.5).

    Finds the exact continuous Bayesian decision threshold without histogram binning.

    Args:
        values: 1D array of extracted periodicity metrics.
        prior: Fixed prior probability for class 0 (Unvoiced) and class 1 (Voiced).
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on centroid shift.

    Returns:
        dict: {
            'threshold': float,
            'mu0': float, 'std0': float,
            'mu1': float, 'std1': float,
            'log_likelihood': float,
            'converged': bool,
            'num_iter': int
        }
    """
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) < 4:
        mid = float(np.median(clean_vals)) if len(clean_vals) > 0 else 0.5
        return {
            "threshold": mid,
            "mu0": mid * 0.8, "std0": 0.1,
            "mu1": mid * 1.2, "std1": 0.1,
            "log_likelihood": 0.0,
            "converged": False,
            "num_iter": 0,
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

    for it in range(max_iter):
        # E-step: Compute responsibilities
        pdf0 = norm.pdf(clean_vals, loc=mu0, scale=std0) + 1e-12
        pdf1 = norm.pdf(clean_vals, loc=mu1, scale=std1) + 1e-12

        total_pdf = w0 * pdf0 + w1 * pdf1
        gamma0 = (w0 * pdf0) / total_pdf
        gamma1 = 1.0 - gamma0

        # M-step: Update parameters (keeping w0 = w1 = 0.5 fixed!)
        n0 = np.sum(gamma0)
        n1 = np.sum(gamma1)

        if n0 < 1e-4 or n1 < 1e-4:
            break

        new_mu0 = float(np.sum(gamma0 * clean_vals) / n0)
        new_mu1 = float(np.sum(gamma1 * clean_vals) / n1)

        new_std0 = float(np.sqrt(np.sum(gamma0 * (clean_vals - new_mu0) ** 2) / n0 + 1e-6))
        new_std1 = float(np.sqrt(np.sum(gamma1 * (clean_vals - new_mu1) ** 2) / n1 + 1e-6))

        # Check convergence
        shift = max(abs(new_mu0 - mu0), abs(new_mu1 - mu1))
        mu0, mu1 = new_mu0, new_mu1
        std0, std1 = new_std0, new_std1

        ll = float(np.sum(np.log(total_pdf)))
        if shift < tol or abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    # Ensure mu0 < mu1
    if mu0 > mu1:
        mu0, mu1 = mu1, mu0
        std0, std1 = std1, std0

    # Solve exact continuous Bayesian intersection: N(mu0, std0) == N(mu1, std1)
    # Equation: A * x^2 + B * x + C = 0
    var0 = std0 ** 2
    var1 = std1 ** 2

    if abs(var0 - var1) < 1e-6:
        t_optimal = (mu0 + mu1) / 2.0
    else:
        a = 1.0 / (2.0 * var1) - 1.0 / (2.0 * var0)
        b = mu0 / var0 - mu1 / var1
        c = (mu1 ** 2) / (2.0 * var1) - (mu0 ** 2) / (2.0 * var0) + np.log(std0 / std1)

        disc = b ** 2 - 4.0 * a * c
        if disc >= 0:
            root1 = (-b - np.sqrt(disc)) / (2.0 * a)
            root2 = (-b + np.sqrt(disc)) / (2.0 * a)

            # Pick root strictly between mu0 and mu1
            in_range = [r for r in (root1, root2) if mu0 <= r <= mu1]
            if in_range:
                t_optimal = in_range[0]
            else:
                mid = (mu0 + mu1) / 2.0
                t_optimal = min((root1, root2), key=lambda r: abs(r - mid))
        else:
            t_optimal = (mu0 * std1 + mu1 * std0) / (std0 + std1)

    return {
        "threshold": float(t_optimal),
        "mu0": float(mu0),
        "std0": float(std0),
        "mu1": float(mu1),
        "std1": float(std1),
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

    Returns:
        labels: 1D array of predicted labels ('v' or 'uv').
        model_info: Centroids and cluster statistics.
    """
    method = method.lower()
    n = len(periodicity_vals)
    if n == 0:
        return np.array([]), {}

    # Feature matrix: shape (N, 2)
    p_mean, p_std = np.mean(periodicity_vals), max(np.std(periodicity_vals), 1e-6)
    z_mean, z_std = np.mean(zcr_vals), max(np.std(zcr_vals), 1e-6)

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

    for _ in range(max_iter):
        dist0 = np.sum((X - c0) ** 2, axis=1)
        dist1 = np.sum((X - c1) ** 2, axis=1)
        cluster1_mask = dist1 < dist0

        pts0 = X[~cluster1_mask]
        pts1 = X[cluster1_mask]

        new_c0 = np.mean(pts0, axis=0) if len(pts0) > 0 else c0
        new_c1 = np.mean(pts1, axis=0) if len(pts1) > 0 else c1

        shift = max(np.linalg.norm(new_c0 - c0), np.linalg.norm(new_c1 - c1))
        c0, c1 = new_c0, new_c1

        if shift < tol:
            break

    # Determine which centroid is Voiced
    # Map centroids back to original scale
    c0_orig = np.array([c0[0] * p_std + p_mean, c0[1] * z_std + z_mean])
    c1_orig = np.array([c1[0] * p_std + p_mean, c1[1] * z_std + z_mean])

    if method == "amdf":
        # Lower Dmin and lower ZCR is Voiced
        score0 = c0_orig[0] + c0_orig[1]
        score1 = c1_orig[0] + c1_orig[1]
        v_idx = 0 if score0 < score1 else 1
    else:
        # Higher periodicity and lower ZCR is Voiced: score = Periodicity - ZCR
        score0 = c0_orig[0] - c0_orig[1]
        score1 = c1_orig[0] - c1_orig[1]
        v_idx = 1 if score1 > score0 else 0

    dist0 = np.sum((X - c0) ** 2, axis=1)
    dist1 = np.sum((X - c1) ** 2, axis=1)
    is_v = (dist0 < dist1) if v_idx == 0 else (dist1 < dist0)

    pred_labels = np.where(is_v, "v", "uv")
    centroid_v = c0_orig if v_idx == 0 else c1_orig
    centroid_uv = c1_orig if v_idx == 0 else c0_orig

    return pred_labels, {
        "centroid_v": centroid_v,
        "centroid_uv": centroid_uv,
        "p_mean": p_mean,
        "p_std": p_std,
        "z_mean": z_mean,
        "z_std": z_std,
    }


# ==============================================================================
# 3. 2-STATE UNSUPERVISED HIDDEN MARKOV MODEL (BAUM-WELCH + VITERBI)
# ==============================================================================

def train_and_decode_2state_hmm(
    observations: np.ndarray,
    method: str = "acf",
    prior: float = 0.5,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit a 2-State continuous Gaussian HMM without labels using Baum-Welch (EM)
    and decode the globally optimal state sequence using the Viterbi algorithm.

    State 0: Unvoiced (UV), State 1: Voiced (V).
    Leverages speech temporal continuity: self-transition probabilities ~ 0.85.

    Returns:
        labels: 1D array of decoded states ('v' or 'uv').
        hmm_params: Final transition matrix, emission means & variances.
    """
    method = method.lower()
    T = len(observations)
    if T < 2:
        return np.full(T, "v", dtype=object), {}

    X = np.copy(observations)

    # Initial transition matrix A with high self-transitions (temporal continuity)
    A = np.array([
        [0.85, 0.15],
        [0.15, 0.85]
    ], dtype=np.float64)

    pi = np.array([prior, 1.0 - prior], dtype=np.float64)

    # Emission initialization
    p25, p75 = float(np.percentile(X, 25)), float(np.percentile(X, 75))
    mu = np.array([min(p25, p75), max(p25, p75)], dtype=np.float64)
    var = np.array([np.var(X) * 0.5 + 1e-4, np.var(X) * 0.5 + 1e-4], dtype=np.float64)

    # Baum-Welch training loop
    for it in range(max_iter):
        # Emission probabilities: B[t, j] = P(x_t | S_j)
        B = np.zeros((T, 2), dtype=np.float64)
        for j in range(2):
            B[:, j] = norm.pdf(X, loc=mu[j], scale=np.sqrt(var[j])) + 1e-12

        # Forward pass (scaled to prevent underflow)
        alpha = np.zeros((T, 2), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)

        alpha[0] = pi * B[0]
        scale[0] = np.sum(alpha[0]) + 1e-12
        alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            scale[t] = np.sum(alpha[t]) + 1e-12
            alpha[t] /= scale[t]

        # Backward pass
        beta = np.zeros((T, 2), dtype=np.float64)
        beta[-1] = 1.0 / scale[-1]

        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / scale[t]

        # Posterior state probabilities
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True) + 1e-12
        gamma /= gamma_sum

        # Parameter updates
        new_mu = np.zeros(2)
        new_var = np.zeros(2)
        for j in range(2):
            g_sum = np.sum(gamma[:, j]) + 1e-12
            new_mu[j] = np.sum(gamma[:, j] * X) / g_sum
            new_var[j] = np.sum(gamma[:, j] * (X - new_mu[j]) ** 2) / g_sum + 1e-5

        shift = np.max(np.abs(new_mu - mu))
        mu, var = new_mu, new_var

        if shift < tol:
            break

    # Determine which state corresponds to Voiced
    if method == "amdf":
        # Lower dip is Voiced
        v_state = 0 if mu[0] < mu[1] else 1
    else:
        # Higher peak is Voiced
        v_state = 1 if mu[1] > mu[0] else 0
    uv_state = 1 - v_state

    # Viterbi decoding for globally optimal state path
    B = np.zeros((T, 2), dtype=np.float64)
    for j in range(2):
        B[:, j] = norm.pdf(X, loc=mu[j], scale=np.sqrt(var[j])) + 1e-12

    V = np.zeros((T, 2), dtype=np.float64)
    ptr = np.zeros((T, 2), dtype=int)

    V[0] = np.log(pi + 1e-12) + np.log(B[0])
    log_A = np.log(A + 1e-12)

    for t in range(1, T):
        for j in range(2):
            trans_probs = V[t - 1] + log_A[:, j]
            best_prev = np.argmax(trans_probs)
            ptr[t, j] = best_prev
            V[t, j] = trans_probs[best_prev] + np.log(B[t, j])

    # Traceback
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(V[-1])
    for t in range(T - 2, -1, -1):
        best_path[t] = ptr[t + 1, best_path[t + 1]]

    decoded_labels = np.where(best_path == v_state, "v", "uv")

    return decoded_labels, {
        "A": A,
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

    Args:
        values: 1D array of extracted periodicity metrics.
        ste: 1D array of Short-Time Energy.
        global_threshold: Baseline global threshold (e.g. Otsu or Median 50%).
        window_frames: Length of sliding window in frames (50 frames ~ 500ms).
        alpha: Weight given to local context (0.0 = static, 1.0 = purely local).

    Returns:
        np.ndarray: Array of dynamic thresholds T(t) for each frame.
    """
    n = len(values)
    if n == 0:
        return np.array([])

    t_dynamic = np.full(n, global_threshold, dtype=np.float64)
    half_w = window_frames // 2

    # Relative energy weighting: frames with higher local energy adapt more strongly
    ste_clean = np.copy(ste)
    max_ste = np.max(ste_clean) if len(ste_clean) > 0 else 1.0
    norm_ste = ste_clean / (max_ste + 1e-12)

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        local_vals = values[start:end]

        if len(local_vals) > 5:
            local_med = float(np.median(local_vals))
            # Modulate alpha slightly by local speech activity
            local_alpha = alpha * min(1.0, float(np.mean(norm_ste[start:end])) * 2.0 + 0.2)
            t_dynamic[i] = (1.0 - local_alpha) * global_threshold + local_alpha * local_med
        else:
            t_dynamic[i] = global_threshold

    return t_dynamic
