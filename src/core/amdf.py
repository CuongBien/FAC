"""Average Magnitude Difference Function (AMDF) implementation for F0 estimation."""
from typing import Tuple
import numpy as np


def compute_amdf(frame: np.ndarray, mode: str = "normalized") -> np.ndarray:
    """Compute standard or normalized AMDF for a signal frame.

    D(tau) = sum_{n=0}^{N - 1 - tau} |x(n) - x(n + tau)|

    Args:
        frame: 1D numpy array representing one signal frame.
        mode:
            - 'raw': Standard unnormalized AMDF.
            - 'normalized': Normalized by sum(|x(n)| + |x(n+tau)|) so D(tau) in [0.0, 1.0].

    Returns:
        np.ndarray: AMDF values for lag tau = 0 .. N-1.
    """
    n = len(frame)
    if n == 0:
        return np.array([], dtype=np.float64)

    d = np.zeros(n, dtype=np.float64)

    if mode == "normalized":
        abs_frame = np.abs(frame)
        for tau in range(n):
            diff = np.abs(frame[: n - tau] - frame[tau:])
            denom = np.sum(abs_frame[: n - tau] + abs_frame[tau:])
            if denom > 1e-12:
                d[tau] = np.sum(diff) / denom
            else:
                d[tau] = 0.0
    else:
        for tau in range(n):
            diff = np.abs(frame[: n - tau] - frame[tau:])
            d[tau] = np.sum(diff)

    return d


def find_f0_amdf(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
    mode: str = "normalized",
) -> Tuple[float, float, int]:
    """Estimate fundamental frequency F0 and dip (minimum valley) value using AMDF.

    Restricted to human pitch range [f0_min, f0_max] Hz.
    In AMDF, periodic (Voiced) frames produce a deep local minimum (dip) at tau_0 = T0 * fs.

    Args:
        frame: 1D signal frame.
        sample_rate: Audio sampling frequency in Hz.
        f0_min: Minimum pitch frequency (default 70 Hz).
        f0_max: Maximum pitch frequency (default 400 Hz).
        mode: AMDF normalization mode ('normalized' or 'raw').

    Returns:
        tuple: (f0_hz: float, dip_value: float, best_lag: int)
            If no valid dip is found, f0_hz = 0.0.
    """
    n = len(frame)
    d = compute_amdf(frame, mode=mode)

    lag_min = max(1, int(round(sample_rate / f0_max)))
    lag_max = min(n - 1, int(round(sample_rate / f0_min)))

    if lag_min >= lag_max or lag_max >= len(d):
        return 0.0, 1.0, 0

    search_region = d[lag_min : lag_max + 1]
    if len(search_region) == 0:
        return 0.0, 1.0, 0

    # Look for local minima (dips) strictly inside the search region
    local_dips = []
    for lag in range(lag_min, lag_max + 1):
        left = d[lag - 1] if lag > 0 else np.inf
        right = d[lag + 1] if lag + 1 < len(d) else np.inf
        if d[lag] <= left and d[lag] <= right:
            local_dips.append(lag)

    if local_dips:
        min_dip = min(d[l] for l in local_dips)
        # Check for first significant dip to avoid pitch halving (selecting 2*tau_0)
        best_lag = None
        for lag in local_dips:
            # If this dip is sufficiently deep and close to global minimum, it represents tau_0
            if d[lag] <= min_dip + 0.05 or d[lag] <= min_dip * 1.25:
                best_lag = lag
                break
        if best_lag is None:
            best_lag = min(local_dips, key=lambda l: d[l])
        dip_value = float(d[best_lag])
    else:
        # Fallback to argmin in region if no strict local dip
        rel_idx = int(np.argmin(search_region))
        best_lag = lag_min + rel_idx
        dip_value = float(d[best_lag])

    f0_hz = float(sample_rate / best_lag) if best_lag > 0 else 0.0
    return f0_hz, dip_value, best_lag
