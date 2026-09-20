"""Plotting routines using Matplotlib for speech signals, F0 contours, and ACF/AMDF."""
import matplotlib.pyplot as plt
import numpy as np


def plot_signal_and_f0_contour(
    time_sig: np.ndarray,
    signal: np.ndarray,
    time_f0: np.ndarray,
    f0_contour: np.ndarray,
    title: str = "Signal and Estimated F0 Contour",
    save_path: str = None,
):
    """Plot waveform in top subplot and F0 contour in bottom subplot, aligned on timeline."""
    pass


def plot_frame_analysis(
    voiced_frame: np.ndarray,
    voiced_corr: np.ndarray,
    unvoiced_frame: np.ndarray,
    unvoiced_corr: np.ndarray,
    sample_rate: int,
    method: str = "acf",
    save_path: str = None,
):
    """Plot side-by-side comparison of 1 periodic (voiced) frame and 1 non-periodic (unvoiced) frame."""
    pass


def plot_distributions_and_threshold(
    values_v: np.ndarray,
    values_u: np.ndarray,
    threshold: float,
    method: str = "acf",
    save_path: str = None,
):
    """Plot Voiced and Unvoiced peak/dip distributions with the separation threshold T."""
    pass


def plot_parameter_comparison(
    results_20ms: dict,
    results_30ms: dict,
    title: str = "Impact of Frame Length (20ms vs 30ms)",
    save_path: str = None,
):
    """Plot comparison between 20ms and 30ms frame lengths."""
    pass
