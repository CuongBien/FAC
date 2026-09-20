"""Plugin 3: Lọc thông dải tiền xử lý (Bandpass Pre-filtering)."""
import numpy as np
from scipy.signal import butter, filtfilt

from .base import BasePlugin


class BandpassFilterPlugin(BasePlugin):
    """Applies a zero-phase digital bandpass filter to suppress out-of-band noise and formants before ACF computation.

    - Passband: [low_cutoff, high_cutoff] Hz (e.g. 70 Hz to 900 Hz, encompassing human pitch and lower harmonics).
    - Filter: 2nd-order Butterworth.
    - Zero-phase filtering: uses filtfilt to prevent time delays or phase distortion.
    """

    def __init__(
        self,
        low_cutoff: float = 70.0,
        high_cutoff: float = 900.0,
        order: int = 2,
    ):
        super().__init__(name="BandpassPreFilter")
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order

    def pre_process_signal(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        nyquist = 0.5 * sample_rate
        if nyquist <= self.high_cutoff:
            return signal

        low = max(10.0, self.low_cutoff) / nyquist
        high = min(nyquist - 10.0, self.high_cutoff) / nyquist

        b, a = butter(self.order, [low, high], btype="band")
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal.astype(np.float64)
