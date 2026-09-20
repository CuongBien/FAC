"""Average Magnitude Difference Function (AMDF) implementation for F0 estimation."""
import numpy as np


def compute_amdf(frame: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Compute standard or normalized AMDF for a signal frame.

    D(tau) = sum_{n=0}^{N-1-tau} |x(n) - x(n+tau)|

    Args:
        frame: 1D numpy array of signal frame
        normalized: Whether to normalize by frame amplitude

    Returns:
        np.ndarray: AMDF values for lag tau = 0 .. N-1
    """
    pass


def find_f0_amdf(frame: np.ndarray, sample_rate: int, f0_min: float = 70.0, f0_max: float = 400.0) -> tuple:
    """Find pitch period lag and minimum dip value using AMDF within [70, 400] Hz.

    Returns:
        tuple: (f0_hz: float, dip_value: float, lag: int)
    """
    pass
