"""Pitch-Shift Equivariance Consistency using ACF (preserved baseline).

Exploits geometric symmetry and transformation equivariance:
When a periodic signal frame is resampled by a scaling factor alpha, its
fundamental pitch transforms predictably as F0_new = alpha * F0_old.
Stochastic unvoiced frames lack this harmonic structure and violate the
scaling relationship. The equivariance consistency error directly defines
the self-supervised Voiced probability.
"""
from typing import Any, Dict, Tuple
import numpy as np
from scipy.signal import resample

from src.core.acf import find_f0_acf
from src.core.processing import (
    center_frame,
    prepare_analysis_signal,
    silence_energy_threshold,
    validate_pitch_range,
)


def compute_frame_pitch_shift_equivariance(
    frame: np.ndarray,
    sample_rate: int,
    alpha: float = 1.25,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    sigma: float = 0.12,
) -> Tuple[float, float, float]:
    """Compute pitch-shift equivariance error and self-supervised Voiced probability.

    Args:
        frame: 1D signal frame.
        sample_rate: Audio sampling frequency in Hz.
        alpha: Pitch scaling factor (default 1.25, shifting pitch up by 25%).
        f0_min: Pitch search minimum Hz.
        f0_max: Pitch search maximum Hz.
        sigma: Equivariance tolerance width.

    Returns:
        f0_hz: Original pitch frequency in Hz.
        rel_error: Relative equivariance discrepancy.
        prob_voiced: Self-supervised probability in [0.0, 1.0].
    """
    frame = center_frame(frame)
    n = len(frame)
    if n == 0 or np.mean(frame ** 2) <= 1e-12:
        return 0.0, 1.0, 0.0

    # 1. Estimate F0 of original frame
    f0_orig, r_orig, _ = find_f0_acf(frame, sample_rate, f0_min, f0_max)
    if f0_orig <= 0.0 or r_orig <= 0.10:
        return 0.0, 1.0, 0.0

    # 2. Resample frame to scale pitch by factor alpha
    # Resampling N samples to N_new = int(N / alpha) accelerates waveform, multiplying frequency by alpha
    new_len = max(16, int(round(n / alpha)))
    shifted_frame = resample(frame, new_len)

    # 3. Estimate F0 of transformed frame
    # Note: Shifted pitch can reach alpha * f0_max, bounded by Nyquist
    shifted_f0_min = max(20.0, f0_min * alpha * 0.7)
    shifted_f0_max = min(sample_rate / 2.0 - 50.0, f0_max * alpha * 1.3)

    f0_shifted, r_shifted, _ = find_f0_acf(
        shifted_frame, sample_rate, shifted_f0_min, shifted_f0_max
    )

    if f0_shifted <= 0.0:
        return f0_orig, 1.0, 0.0

    # 4. Measure relative equivariance error
    expected_f0 = f0_orig * alpha
    rel_error = float(abs(f0_shifted - expected_f0) / expected_f0)

    # 5. Self-supervised probability: joint probability of equivariance and periodicity strength
    equivariance_score = float(np.exp(- (rel_error ** 2) / (2.0 * sigma ** 2)))
    periodicity_strength = float(np.clip(r_orig, 0.0, 1.0))

    prob_voiced = float(np.clip(equivariance_score * periodicity_strength, 0.0, 1.0))
    return f0_orig, rel_error, prob_voiced


def analyze_signal_ssl_pitch_shift(
    signal: np.ndarray,
    sample_rate: int,
    frame_duration_ms: float = 30.0,
    hop_duration_ms: float = 10.0,
    alpha: float = 1.25,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    ste_ratio: float = 0.01,
) -> Dict[str, Any]:
    """Perform Pitch-Shift Equivariance SSL pitch and V/UV analysis on a signal.

    Args:
        signal: 1D audio waveform array.
        sample_rate: Audio sampling frequency in Hz.
        frame_duration_ms: Frame length in ms.
        hop_duration_ms: Hop shift in ms.
        alpha: Artificial pitch scaling ratio.
        f0_min: Minimum pitch frequency.
        f0_max: Maximum pitch frequency.
        ste_ratio: Relative silence floor.

    Returns:
        Dict containing 'f0_contour', 'labels', 'probabilities', 'rel_errors'.
    """
    validate_pitch_range(sample_rate, f0_min, f0_max)
    values = prepare_analysis_signal(signal, sample_rate, frame_duration_ms)

    frame_len = int(round(frame_duration_ms * 1e-3 * sample_rate))
    hop_len = int(round(hop_duration_ms * 1e-3 * sample_rate))
    num_frames = max(1, (len(values) - frame_len) // hop_len + 1)

    f0_list = []
    prob_list = []
    error_list = []
    ste_list = []

    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frame = values[start:end]

        f0, rel_err, prob = compute_frame_pitch_shift_equivariance(
            frame, sample_rate, alpha=alpha, f0_min=f0_min, f0_max=f0_max
        )
        f0_list.append(f0)
        error_list.append(rel_err)
        prob_list.append(prob)
        ste_list.append(float(np.mean(frame ** 2)) if len(frame) else 0.0)

    ste_arr = np.array(ste_list, dtype=np.float64)
    ste_thresh = silence_energy_threshold(ste_arr, ste_ratio)
    active_mask = ste_arr >= ste_thresh

    probabilities = np.array(prob_list, dtype=np.float64)
    probabilities[~active_mask] = 0.0

    # Natural decision threshold at P >= 0.5
    labels = np.where((probabilities >= 0.5) & active_mask, "v", "uv")

    f0_raw = np.array(f0_list, dtype=np.float64)
    f0_contour = np.where(labels == "v", f0_raw, 0.0)

    return {
        "f0_contour": f0_contour,
        "labels": labels,
        "probabilities": probabilities,
        "rel_errors": np.array(error_list, dtype=np.float64),
        "active_mask": active_mask,
    }
