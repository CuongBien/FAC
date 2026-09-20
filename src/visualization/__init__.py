"""Visualization utilities for waveform, pitch contour, frame analysis, and distributions."""
from .plotter import (
    plot_signal_and_f0_contour,
    plot_frame_acf_comparison,
    plot_distributions_and_threshold,
    plot_parameter_comparison,
)

# Alias for generic frame analysis
plot_frame_analysis = plot_frame_acf_comparison

__all__ = [
    "plot_signal_and_f0_contour",
    "plot_frame_acf_comparison",
    "plot_frame_analysis",
    "plot_distributions_and_threshold",
    "plot_parameter_comparison",
]
