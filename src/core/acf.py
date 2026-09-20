"""Autocorrelation Function (ACF) implementation for F0 estimation and periodicity detection."""
from typing import Optional, Tuple
import numpy as np


def compute_acf(frame: np.ndarray, mode: str = "normalized") -> np.ndarray:
    """Compute Autocorrelation Function (ACF) for a signal frame.

    R(tau) = sum_{n=0}^{N - 1 - tau} x(n) * x(n + tau)

    Args:
        frame: 1D numpy array representing one signal frame.
        mode:
            - 'raw': Unnormalized R(tau).
            - 'normalized': Normalized by R(0), so R(0) = 1.0.
            - 'pearson': Fully normalized by local energy terms (NACF) in [-1.0, 1.0].

    Returns:
        np.ndarray: Autocorrelation values for lag tau = 0 .. N-1.
    """
    n = len(frame)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Compute correlation via numpy correlate (efficient time-domain or FFT)
    # Using 'full' mode:
    corr_full = np.correlate(frame, frame, mode="full")
    # Non-negative lags start from index n - 1
    r = corr_full[n - 1 :].astype(np.float64)

    r0 = r[0]
    if r0 <= 1e-12:
        return np.zeros_like(r)

    if mode == "normalized":
        return r / r0

    elif mode == "pearson":
        # Overlap-normalized (NACF)
        # nacf[tau] = sum(x[n]*x[n+tau]) / sqrt(sum(x[n]^2) * sum(x[n+tau]^2))
        nacf = np.zeros(n, dtype=np.float64)
        nacf[0] = 1.0
        cum_sq = np.cumsum(frame ** 2)
        total_sq = cum_sq[-1]

        for tau in range(1, n):
            # sum_{n=0}^{N-1-tau} x(n)^2
            e1 = cum_sq[n - 1 - tau]
            # sum_{n=0}^{N-1-tau} x(n+tau)^2 = sum_{k=tau}^{N-1} x(k)^2
            e2 = total_sq - (cum_sq[tau - 1] if tau > 0 else 0.0)
            denom = np.sqrt(e1 * e2)
            if denom > 1e-12:
                nacf[tau] = r[tau] / denom
            else:
                nacf[tau] = 0.0
        return nacf

    return r


def find_f0_acf(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    mode: str = "normalized",
) -> Tuple[float, float, int]:
    """Estimate fundamental frequency F0 and peak value using ACF.

    Restricted to human pitch range [f0_min, f0_max] Hz.

    Args:
        frame: 1D signal frame.
        sample_rate: Audio sampling frequency in Hz.
        f0_min: Minimum pitch frequency (default 70 Hz).
        f0_max: Maximum pitch frequency (default 400 Hz).
        mode: ACF normalization mode ('normalized', 'pearson', or 'raw').

    Returns:
        tuple: (f0_hz: float, peak_value: float, best_lag: int)
            If no valid peak is found, f0_hz = 0.0.
    """
    n = len(frame)
    r = compute_acf(frame, mode=mode)

    lag_min = max(1, int(round(sample_rate / f0_max)))
    lag_max = min(n - 1, int(round(sample_rate / f0_min)))

    if lag_min >= lag_max or lag_max >= len(r):
        return 0.0, 0.0, 0

    search_region = r[lag_min : lag_max + 1]
    if len(search_region) == 0:
        return 0.0, 0.0, 0

    # Look for local maxima strictly inside the search region
    local_peaks = []
    for lag in range(lag_min, lag_max + 1):
        left = r[lag - 1] if lag > 0 else -np.inf
        right = r[lag + 1] if lag + 1 < len(r) else -np.inf
        if r[lag] >= left and r[lag] >= right:
            local_peaks.append(lag)

    if local_peaks:
        # Pick the highest local peak
        best_lag = max(local_peaks, key=lambda l: r[l])
        peak_value = float(r[best_lag])
    else:
        # Fallback to argmax in region if no strict local peak
        rel_idx = int(np.argmax(search_region))
        best_lag = lag_min + rel_idx
        peak_value = float(r[best_lag])

    f0_hz = float(sample_rate / best_lag) if best_lag > 0 else 0.0
    return f0_hz, peak_value, best_lag
