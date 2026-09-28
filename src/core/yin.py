"""YIN Pitch Tracking Algorithm implementation.

Reference:
    Alain de Cheveigné and Hideki Kawahara (2002).
    "YIN, a fundamental frequency estimator for speech and music."
    Journal of the Acoustical Society of America, 111(4), 1917-1930.
"""
from typing import Tuple, Optional
import numpy as np


def compute_difference_function(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """Step 1: Compute the squared difference function d(tau) for tau = 0 .. max_lag.

    d(tau) = sum_{j=0}^{W-1} (x[j] - x[j + tau])^2
    where W = len(frame) - max_lag (constant integration window).
    """
    n = len(frame)
    if n <= max_lag:
        max_lag = n - 1

    W = n - max_lag
    if W < 30:
        W = n // 2
        max_lag = n - W

    d = np.zeros(max_lag + 1, dtype=np.float64)
    ref = frame[:W]

    for tau in range(max_lag + 1):
        diff = ref - frame[tau : W + tau]
        d[tau] = np.dot(diff, diff)

    return d


def cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    """Step 2: Cumulative Mean Normalized Difference Function (CMNDF).

    d'(0) = 1
    d'(tau) = d(tau) / [ (1 / tau) * sum_{j=1}^{tau} d(j) ], for tau >= 1
    """
    d_prime = np.zeros_like(d, dtype=np.float64)
    d_prime[0] = 1.0

    running_sum = 0.0
    for tau in range(1, len(d)):
        running_sum += d[tau]
        if running_sum > 1e-12:
            d_prime[tau] = (d[tau] * tau) / running_sum
        else:
            d_prime[tau] = 1.0

    return d_prime


def parabolic_interpolation(d_prime: np.ndarray, tau: int) -> Tuple[float, float]:
    """Step 4: Sub-sample parabolic interpolation around a local dip.

    Fits a parabola through [tau-1, tau, tau+1] to estimate continuous peak/dip position:
        delta = (y_{-1} - y_{+1}) / [ 2 * (y_{+1} + y_{-1} - 2*y_0) ]
        tau_fine = tau + delta
        y_fine = y_0 - (delta / 4) * (y_{-1} - y_{+1})

    Returns:
        tuple: (tau_fine: float, dip_depth: float)
    """
    if tau <= 0 or tau >= len(d_prime) - 1:
        return float(tau), float(d_prime[tau])

    y_prev = d_prime[tau - 1]
    y_curr = d_prime[tau]
    y_next = d_prime[tau + 1]

    denom = 2.0 * (y_prev + y_next - 2.0 * y_curr)
    if abs(denom) < 1e-12:
        return float(tau), float(y_curr)

    delta = (y_prev - y_next) / denom
    if abs(delta) > 1.0:
        return float(tau), float(y_curr)

    tau_fine = float(tau) + delta
    val_fine = float(y_curr - 0.25 * delta * (y_prev - y_next))
    return tau_fine, max(0.0, val_fine)


def find_f0_yin(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    harmonic_threshold: float = 0.15,
) -> Tuple[float, float, float]:
    """Estimate fundamental frequency F0 and dip depth using the YIN algorithm.

    Args:
        frame: 1D signal frame (array of samples).
        sample_rate: Audio sampling frequency in Hz.
        f0_min: Minimum expected pitch in Hz (default: 70 Hz).
        f0_max: Maximum expected pitch in Hz (default: 400 Hz).
        harmonic_threshold: Threshold on CMNDF for selecting the first prominent dip (default: 0.15).

    Returns:
        tuple: (f0_hz: float, dip_depth: float, best_lag: float)
            If frame is unvoiced, f0_hz = 0.0.
    """
    lag_min = max(1, int(round(sample_rate / f0_max)))
    lag_max = min(len(frame) - 1, int(round(sample_rate / f0_min)))

    if lag_min >= lag_max:
        return 0.0, 1.0, 0.0

    # Step 1: Difference Function
    d = compute_difference_function(frame, max_lag=lag_max)

    # Step 2: Cumulative Mean Normalized Difference Function
    d_prime = cumulative_mean_normalized_difference(d)

    # Step 3: Absolute Thresholding
    # Search for the first local minimum below harmonic_threshold
    tau_selected: Optional[int] = None
    for tau in range(lag_min, min(lag_max, len(d_prime) - 1)):
        if d_prime[tau] < harmonic_threshold:
            # Confirm it's a local minimum
            if d_prime[tau] <= d_prime[tau - 1] and d_prime[tau] <= d_prime[tau + 1]:
                tau_selected = tau
                break

    # If no dip falls below threshold, fallback to the global minimum within search range
    is_voiced = True
    if tau_selected is None:
        search_slice = d_prime[lag_min : lag_max + 1]
        if len(search_slice) == 0:
            return 0.0, 1.0, 0.0
        rel_min = int(np.argmin(search_slice))
        tau_selected = lag_min + rel_min
        # If even global minimum is too high, declare unvoiced
        if d_prime[tau_selected] >= harmonic_threshold * 1.5:
            is_voiced = False

    # Step 4: Parabolic sub-sample interpolation
    tau_fine, dip_val = parabolic_interpolation(d_prime, tau_selected)

    if not is_voiced or tau_fine <= 0.0:
        return 0.0, dip_val, tau_fine

    f0_hz = float(sample_rate / tau_fine)
    # Sanity check frequency bounds
    if f0_hz < f0_min * 0.9 or f0_hz > f0_max * 1.1:
        return 0.0, dip_val, tau_fine

    return f0_hz, dip_val, tau_fine
