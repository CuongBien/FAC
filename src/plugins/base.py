"""Base class and interface for pitch detection plugins."""
from typing import Any, Dict, Tuple
import numpy as np


class BasePlugin:
    """Abstract plugin interface allowing pre-processing, decision adjustment, and post-processing."""

    def __init__(self, name: str = "BasePlugin"):
        self.name = name

    def reset(self) -> None:
        """Reset internal state before processing a new audio file."""
        pass

    def pre_process_signal(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Hook executed before framing (e.g., bandpass filtering).

        Args:
            signal: 1D audio waveform.
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            np.ndarray: Modified 1D audio waveform.
        """
        return signal

    def adjust_frame_decision(
        self,
        frame_idx: int,
        peak_val: float,
        f0_val: float,
        is_voiced: bool,
        context: Dict[str, Any],
    ) -> Tuple[bool, float]:
        """Hook executed during frame-by-frame V/UV decision (e.g., hysteresis thresholding).

        Args:
            frame_idx: Index of current frame.
            peak_val: ACF peak amplitude for current frame.
            f0_val: Estimated F0 in Hz for current frame.
            is_voiced: Current decision (True for voiced, False for unvoiced).
            context: Shared processing context (sample_rate, ste, frames, etc.).

        Returns:
            tuple: (new_is_voiced: bool, new_f0_val: float)
        """
        return is_voiced, f0_val

    def post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Hook executed after framing and initial contour generation (e.g., energy boundary extension).

        Args:
            result: Result dictionary from pitch detector.

        Returns:
            dict: Modified result dictionary.
        """
        return result
