"""Autocorrelation Function (ACF) implementation for F0 estimation and periodicity detection."""
import numpy as np


def compute_acf(frame: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Compute standard or normalized autocorrelation function for a signal frame.

    R(tau) = sum_{n=0}^{N-1-tau} x(n) * x(n+tau)

    Args:
        frame: 1D numpy array of signal frame
        normalized: Whether to normalize by R(0) so R(0) = 1.0

    Returns:
        np.ndarray: Autocorrelation values for lag tau = 0 .. N-1
    """
    pass


def find_f0_acf(frame: np.ndarray, sample_rate: int, f0_min: float = 70.0, f0_max: float = 400.0) -> tuple:
    """Find pitch period lag and peak value using ACF within the human F0 range [70, 400] Hz.

    Returns:
        tuple: (f0_hz: float, peak_value: float, lag: int)
    """
    pass
