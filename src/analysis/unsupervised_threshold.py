"""Unsupervised thresholding methods for Voiced/Unvoiced classification.
Implements:
1. Otsu's Thresholding (Maximizing between-class variance sigma_B^2)
2. Bimodal Histogram Valley Detection (Smoothed histogram local minimum)
3. Median / 50th Percentile Thresholding (Direct exploitation of P(V) ~ P(UV) ~ 0.5)
4. 1D K-Means Clustering (Mid-point boundary between two cluster centroids)
"""
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.core.audio import load_wav, frame_signal, compute_ste
from src.core.lab_parser import parse_lab_file, get_frame_labels
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum


def compute_otsu_threshold(values: np.ndarray, n_bins: int = 100) -> float:
    """Find optimal threshold using Otsu's method on the empirical histogram.
    Maximizes between-class variance: sigma_B^2(T) = w0 * w1 * (mu0 - mu1)^2.

    Args:
        values: 1D array of extracted periodicity metrics from active speech frames.
        n_bins: Number of histogram bins.

    Returns:
        float: Optimal decision threshold T.
    """
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) < 2:
        return float(np.mean(clean_vals)) if len(clean_vals) == 1 else 0.5

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
    return float(bin_centers[best_idx])


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


def extract_active_speech_frame_metrics(
    wav_paths: List[str],
    method: str = "acf",
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    ste_silence_ratio: float = 0.008,
    plugins: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Extract periodicity metric values from non-silence (active speech) frames
    across a set of WAV audio files.

    If ground-truth *.lab files exist, also matches each frame to 'v' or 'uv'
    for validation of unsupervised threshold accuracy.

    Returns:
        dict: {
            'all_values': np.ndarray,
            'voiced_values': np.ndarray,
            'unvoiced_values': np.ndarray,
            'ground_truth_labels': List[str],
            'frame_file_origins': List[str],
            'num_silence_dropped': int,
            'num_active_frames': int,
        }
    """
    method = method.lower()
    all_values = []
    v_values = []
    uv_values = []
    gt_labels = []
    file_origins = []
    num_silence_dropped = 0

    for wav_path in wav_paths:
        sr, signal, _ = load_wav(wav_path)
        base_name = os.path.basename(wav_path)

        if plugins:
            for p in plugins:
                if hasattr(p, "pre_process_signal"):
                    signal = p.pre_process_signal(signal, sr)

        frames, frame_times = frame_signal(
            signal,
            sr,
            frame_duration_ms=frame_duration_ms,
            hop_duration_ms=hop_duration_ms,
            window="rectangular",
        )
        ste, _ = compute_ste(
            signal,
            sr,
            frame_duration_ms=frame_duration_ms,
            hop_duration_ms=hop_duration_ms,
        )

        max_ste = np.max(ste) if len(ste) > 0 else 1.0
        ste_thresh = ste_silence_ratio * max_ste

        # Check for matching LAB file
        lab_path = wav_path.rsplit(".", 1)[0] + ".lab"
        labels = None
        if os.path.exists(lab_path):
            lab_data = parse_lab_file(lab_path)
            labels = get_frame_labels(frame_times, lab_data["segments"])

        for i, frame in enumerate(frames):
            # Silence gating: drop silence frames from V/UV distribution
            if ste[i] < ste_thresh:
                num_silence_dropped += 1
                continue

            if plugins:
                for p in plugins:
                    if hasattr(p, "pre_process_frame"):
                        frame = p.pre_process_frame(frame, sr)

            if method == "acf":
                _, metric_val, _ = find_f0_acf(
                    frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized"
                )
            elif method == "amdf":
                _, metric_val, _ = find_f0_amdf(
                    frame, sr, f0_min=f0_min, f0_max=f0_max, mode="normalized"
                )
            elif method == "cepstrum":
                _, metric_val, _ = find_f0_cepstrum(
                    frame,
                    sr,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    window="hamming",
                    n_fft=2048,
                    mode="peak",
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            all_values.append(metric_val)
            file_origins.append(base_name)

            if labels is not None:
                lbl = labels[i]
                gt_labels.append(lbl)
                if lbl == "v":
                    v_values.append(metric_val)
                elif lbl == "uv":
                    uv_values.append(metric_val)

    return {
        "all_values": np.array(all_values, dtype=np.float64),
        "voiced_values": np.array(v_values, dtype=np.float64),
        "unvoiced_values": np.array(uv_values, dtype=np.float64),
        "ground_truth_labels": gt_labels,
        "frame_file_origins": file_origins,
        "num_silence_dropped": num_silence_dropped,
        "num_active_frames": len(all_values),
    }
