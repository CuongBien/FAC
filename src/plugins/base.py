"""Base classes and interfaces for pitch detection plugins categorized by pipeline stage."""
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np


class PluginStage(str, Enum):
    """Pipeline stages where enhancement plugins operate."""
    PRE_PROCESSING = "pre_processing"
    DECISION = "decision"
    POST_PROCESSING = "post_processing"


class BasePlugin:
    """Abstract plugin interface allowing pre-processing, decision adjustment, and post-processing."""

    stage: PluginStage = PluginStage.PRE_PROCESSING

    def __init__(self, name: str = "BasePlugin", stage: Optional[PluginStage] = None):
        self.name = name
        if stage is not None:
            self.stage = stage

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

    def pre_process_frame(self, frame: np.ndarray, sample_rate: int) -> np.ndarray:
        """Hook executed on each frame before pitch extraction (e.g., center clipping).

        Args:
            frame: 1D windowed signal frame.
            sample_rate: Audio sampling frequency in Hz.

        Returns:
            np.ndarray: Modified 1D frame.
        """
        return frame

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
            peak_val: ACF peak amplitude or AMDF dip depth for current frame.
            f0_val: Estimated F0 in Hz for current frame.
            is_voiced: Current decision (True for voiced, False for unvoiced).
            context: Shared processing context (sample_rate, ste, frames, etc.).

        Returns:
            tuple: (new_is_voiced: bool, new_f0_val: float)
        """
        return is_voiced, f0_val

    def post_process_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Hook executed after framing and initial contour generation (e.g., energy boundary extension, Viterbi tracking).

        Args:
            result: Result dictionary from pitch detector.

        Returns:
            dict: Modified result dictionary.
        """
        return result


class PreProcessingPlugin(BasePlugin):
    """Base class for plugins operating in the Pre-processing Stage (signal filtering, frame clipping)."""
    stage: PluginStage = PluginStage.PRE_PROCESSING

    def __init__(self, name: str = "PreProcessingPlugin"):
        super().__init__(name=name, stage=PluginStage.PRE_PROCESSING)


class DecisionPlugin(BasePlugin):
    """Base class for plugins operating in the Decision Stage (hysteresis, adaptive thresholds)."""
    stage: PluginStage = PluginStage.DECISION

    def __init__(self, name: str = "DecisionPlugin"):
        super().__init__(name=name, stage=PluginStage.DECISION)


class PostProcessingPlugin(BasePlugin):
    """Base class for plugins operating in the Post-processing Stage (edge extension, Viterbi tracking)."""
    stage: PluginStage = PluginStage.POST_PROCESSING

    def __init__(self, name: str = "PostProcessingPlugin"):
        super().__init__(name=name, stage=PluginStage.POST_PROCESSING)
