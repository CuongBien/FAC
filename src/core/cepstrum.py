"""Cepstrum Pitch Determination (Homomorphic Speech Processing) implementation.

Based on Noll (1967) and Oppenheim & Schafer (1968).
Deconvolves speech into vocal tract envelope (low quefrency) and pitch impulses (high quefrency).
"""
from typing import Tuple
import numpy as np
from src.core.processing import center_frame


def compute_real_cepstrum(
    frame: np.ndarray,
    sample_rate: int,
    window: str = "hamming",
    n_fft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Real Cepstrum of a signal frame using Homomorphic Deconvolution.

    c[m] = IFFT(log|FFT(w[n] * x[n])| + eps)

    Args:
        frame: 1D numpy array representing a single audio frame.
        sample_rate: Sampling frequency in Hz.
        window: Windowing function ('hamming', 'hanning', 'rectangular').
        n_fft: Size of FFT (zero-padded for high quefrency interpolation resolution).

    Returns:
        tuple: (cepstrum, quefrencies_sec)
            - cepstrum: 1D numpy array of real cepstrum coefficients c[m].
            - quefrencies_sec: Quefrency axis in seconds (m / sample_rate).
    """
    frame = center_frame(frame)
    n = len(frame)
    if n == 0 or np.mean(frame ** 2) <= 1e-12:
        c_zero = np.zeros(n_fft, dtype=np.float64)
        q_zero = np.arange(n_fft, dtype=np.float64) / sample_rate
        return c_zero, q_zero

    # 1. Apply Windowing to minimize spectral leakage across pitch harmonics
    if window == "hamming":
        w = np.hamming(n)
    elif window == "hanning":
        w = np.hanning(n)
    else:
        w = np.ones(n, dtype=np.float64)

    windowed = (frame - np.mean(frame)) * w

    # 2. Forward FFT to compute Magnitude Spectrum
    n_fft = max(n_fft, 2 ** int(np.ceil(np.log2(n))))
    spectrum = np.fft.rfft(windowed, n=n_fft)
    mag_spectrum = np.abs(spectrum)

    # 3. Logarithm of Magnitude Spectrum: Transforms Convolution into Addition
    # log|S(omega)| = log|H(omega)| (formants) + log|E(omega)| (pitch harmonics)
    log_mag = np.log(mag_spectrum + 1e-12)

    # 4. Inverse FFT to Quefrency Domain (Real Cepstrum)
    # Since log_mag is real and even, IRFFT yields a real and even cepstrum sequence
    cepstrum = np.fft.irfft(log_mag, n=n_fft).astype(np.float64)
    quefrencies_sec = np.arange(len(cepstrum), dtype=np.float64) / sample_rate

    return cepstrum, quefrencies_sec


def find_f0_cepstrum(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    window: str = "hamming",
    n_fft: int = 2048,
    mode: str = "peak",
) -> Tuple[float, float, int]:
    """Estimate fundamental frequency F0, peak metric, and quefrency lag using Cepstrum.

    Separates vocal tract (low quefrency < 2ms) from excitation (high quefrency > 2.5ms).
    Locates peak in [q_min, q_max] with 3-point parabolic interpolation.

    Args:
        frame: 1D signal frame.
        sample_rate: Audio sampling frequency in Hz.
        f0_min: Minimum pitch frequency (default 70 Hz).
        f0_max: Maximum pitch frequency (default 400 Hz).
        window: Windowing function ('hamming', 'hanning', 'rectangular').
        n_fft: FFT length (default 2048).
        mode: Metric returned for Voiced/Unvoiced decision:
            - 'peak': Raw cepstral peak amplitude c[q0].
            - 'normalized': Peak normalized by total frame log-energy c[0] (peak / |c[0]|).
            - 'prominence': Peak Z-score relative to search region ((peak - mean) / std).
            - 'ratio': Peak relative to mean absolute quefrency in search range.

    Returns:
        tuple: (f0_hz: float, peak_metric: float, best_lag: int)
            If no valid peak is found, f0_hz = 0.0.
    """
    frame = center_frame(frame)
    n = len(frame)
    if n == 0 or np.mean(frame ** 2) <= 1e-12:
        return 0.0, 0.0, 0

    cepstrum, _ = compute_real_cepstrum(frame, sample_rate, window=window, n_fft=n_fft)

    # Pitch search region in quefrency samples
    lag_min = max(1, int(round(sample_rate / f0_max)))
    lag_max = min(len(cepstrum) // 2, int(round(sample_rate / f0_min)))

    if lag_min >= lag_max or lag_max >= len(cepstrum):
        return 0.0, 0.0, 0

    search_region = cepstrum[lag_min : lag_max + 1]
    if len(search_region) == 0:
        return 0.0, 0.0, 0

    # Find the maximum peak in the pitch quefrency range
    rel_peak_idx = int(np.argmax(search_region))
    best_lag = lag_min + rel_peak_idx
    peak_val = float(cepstrum[best_lag])

    # Parabolic (quadratic) interpolation around best_lag for sub-sample accuracy
    if 0 < best_lag < len(cepstrum) - 1:
        alpha = float(cepstrum[best_lag - 1])
        beta = float(cepstrum[best_lag])
        gamma = float(cepstrum[best_lag + 1])
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            # Guard against runaway delta outside [-1, 1]
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
    else:
        delta = 0.0

    refined_lag = float(best_lag + delta)
    f0_hz = float(sample_rate / refined_lag) if refined_lag > 0 else 0.0

    # Ensure F0 is strictly within [f0_min, f0_max]
    if f0_hz < f0_min or f0_hz > f0_max:
        f0_hz = float(np.clip(f0_hz, f0_min, f0_max))

    # Compute decision metric
    c0 = abs(float(cepstrum[0])) if abs(float(cepstrum[0])) > 1e-12 else 1.0
    mean_roi = float(np.mean(search_region))
    std_roi = float(np.std(search_region)) + 1e-12
    mean_abs_roi = float(np.mean(np.abs(search_region))) + 1e-12

    if mode == "normalized":
        metric_val = float(peak_val / c0)
    elif mode == "prominence":
        metric_val = float((peak_val - mean_roi) / std_roi)
    elif mode == "ratio":
        metric_val = float(peak_val / mean_abs_roi)
    else:
        # 'peak' mode (standard Noll 1967)
        metric_val = peak_val

    return f0_hz, metric_val, best_lag
