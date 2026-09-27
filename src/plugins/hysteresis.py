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
        metric_val: float,
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

        method = context.get("method", "acf").lower()

        if method == "amdf":
            # For AMDF: smaller dip means stronger periodicity
            # Enter threshold (strict): lower value (e.g. 0.38 - 0.40)
            # Exit threshold (relaxed): higher value (e.g. 0.44 - 0.48)
            t_enter = min(self.t_high, self.t_low)
            t_exit = max(self.t_high, self.t_low)

            if not self._in_voiced:
                if metric_val <= t_enter:
                    self._in_voiced = True
                    self._prev_f0 = f0_val
                    return True, f0_val
                else:
                    self._in_voiced = False
                    self._prev_f0 = 0.0
                    return False, 0.0
            else:
                continuity_ok = (self._prev_f0 == 0.0) or (abs(f0_val - self._prev_f0) <= self.max_pitch_jump)
                if metric_val <= t_exit and continuity_ok:
                    self._in_voiced = True
                    self._prev_f0 = f0_val
                    return True, f0_val
                else:
                    self._in_voiced = False
                    self._prev_f0 = 0.0
                    return False, 0.0
        else:
            # For ACF: larger peak means stronger periodicity
            t_enter = max(self.t_high, self.t_low)
            t_exit = min(self.t_high, self.t_low)

            if not self._in_voiced:
                if metric_val >= t_enter:
                    self._in_voiced = True
                    self._prev_f0 = f0_val
                    return True, f0_val
                else:
                    self._in_voiced = False
                    self._prev_f0 = 0.0
                    return False, 0.0
            else:
                continuity_ok = (self._prev_f0 == 0.0) or (abs(f0_val - self._prev_f0) <= self.max_pitch_jump)
                if metric_val >= t_exit and continuity_ok:
                    self._in_voiced = True
                    self._prev_f0 = f0_val
                    return True, f0_val
                else:
                    self._in_voiced = False
                    self._prev_f0 = 0.0
                    return False, 0.0
