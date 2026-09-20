"""Plugin 1: Ngưỡng trễ kép (Hysteresis Thresholding / Schmitt Trigger)."""
from typing import Any, Dict, Tuple
from .base import BasePlugin


class HysteresisPlugin(BasePlugin):
    """Applies a dual-threshold state machine to reduce boundary dropouts at syllable edges.

    - Turn-on threshold (T_high): High bar required to enter Voiced state (avoids false alarms).
    - Hold-off threshold (T_low): Lower bar required to maintain Voiced state once triggered.
    - Pitch continuity: Ensures pitch does not jump drastically (> max_pitch_jump Hz) during hold-off.
    """

    def __init__(
        self,
        t_high: float = 0.462,
        t_low: float = 0.400,
        max_pitch_jump: float = 40.0,
    ):
        super().__init__(name="HysteresisThresholding")
        self.t_high = t_high
        self.t_low = t_low
        self.max_pitch_jump = max_pitch_jump
        self._in_voiced = False
        self._prev_f0 = 0.0

    def reset(self) -> None:
        self._in_voiced = False
        self._prev_f0 = 0.0

    def adjust_frame_decision(
        self,
        frame_idx: int,
        peak_val: float,
        f0_val: float,
        is_voiced: bool,
        context: Dict[str, Any],
    ) -> Tuple[bool, float]:
        # Silence frames reset state immediately
        is_silence = context.get("is_silence", False)
        if is_silence:
            self._in_voiced = False
            self._prev_f0 = 0.0
            return False, 0.0

        if not self._in_voiced:
            # Entering Voiced requires peak >= t_high
            if peak_val >= self.t_high:
                self._in_voiced = True
                self._prev_f0 = f0_val
                return True, f0_val
            else:
                self._in_voiced = False
                self._prev_f0 = 0.0
                return False, 0.0
        else:
            # Maintaining Voiced requires peak >= t_low and pitch continuity
            continuity_ok = (self._prev_f0 == 0.0) or (abs(f0_val - self._prev_f0) <= self.max_pitch_jump)
            if peak_val >= self.t_low and continuity_ok:
                self._in_voiced = True
                self._prev_f0 = f0_val
                return True, f0_val
            else:
                self._in_voiced = False
                self._prev_f0 = 0.0
                return False, 0.0
