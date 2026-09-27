"""Stage 1 Pre-processing Plugin: Cắt gọt trung tâm từng khung (Center Clipping / Sondhi 1968)."""
from typing import Literal
import numpy as np

from src.plugins.base import PreProcessingPlugin


class CenterClippingPlugin(PreProcessingPlugin):
    """Applies Center Clipping to each frame before ACF/AMDF computation.

    Reference:
        Sondhi, M. M. (1968). "New methods of pitch extraction."
        IEEE Transactions on Audio and Electroacoustics, 16(2), 262-266.

    Eliminates vocal tract formant structure from speech frames (Spectral Flattening),
    leaving only pitch-synchronous pulses at Glottal Closure Instants (GCI).

    Two modes available:
        - "standard": y[n] = sign(x[n]) * max(0, |x[n]| - C_L)
        - "3level":   y[n] = +1 if x > C_L, -1 if x < -C_L, 0 otherwise.
    """

    def __init__(
        self,
        clipping_ratio: float = 0.40,
        mode: Literal["standard", "3level"] = "standard",
    ):
        super().__init__(name="CenterClipping")
        if not (0.0 < clipping_ratio < 1.0):
            raise ValueError(f"clipping_ratio must be between 0 and 1, got {clipping_ratio}")
        self.clipping_ratio = clipping_ratio
        self.mode = mode

    def pre_process_frame(self, frame: np.ndarray, sample_rate: int) -> np.ndarray:
        max_val = np.max(np.abs(frame))
        if max_val == 0.0:
            return frame

        c_l = self.clipping_ratio * max_val
        clipped_frame = np.zeros_like(frame, dtype=np.float64)

        if self.mode == "3level":
            clipped_frame[frame > c_l] = 1.0
            clipped_frame[frame < -c_l] = -1.0
        else:
            pos_mask = frame > c_l
            neg_mask = frame < -c_l
            clipped_frame[pos_mask] = frame[pos_mask] - c_l
            clipped_frame[neg_mask] = frame[neg_mask] + c_l

        return clipped_frame
