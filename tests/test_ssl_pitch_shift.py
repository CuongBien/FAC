"""Unit tests for Pitch-Shift Equivariance SSL."""
import unittest
import numpy as np

from src.analysis.ssl_pitch_shift import (
    compute_frame_pitch_shift_equivariance,
    analyze_signal_ssl_pitch_shift,
)


class TestPitchShiftSSL(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.f0 = 175.0
        t = np.arange(self.sample_rate) / self.sample_rate

        # 1. Harmonic voiced signal
        voiced = 0.6 * np.sin(2 * np.pi * self.f0 * t) + 0.3 * np.sin(2 * np.pi * 350 * t)
        signal = np.zeros(self.sample_rate, dtype=np.float64)
        signal[int(0.2 * self.sample_rate) : int(0.8 * self.sample_rate)] = voiced[
            int(0.2 * self.sample_rate) : int(0.8 * self.sample_rate)
        ]
        self.voiced_signal = signal

        # 2. Pure random noise
        rng = np.random.default_rng(777)
        self.noise_signal = rng.normal(0, 0.1, self.sample_rate)

    def test_compute_frame_pitch_shift_equivariance_voiced(self):
        frame = self.voiced_signal[int(0.4 * self.sample_rate) : int(0.43 * self.sample_rate)]
        f0, rel_err, prob = compute_frame_pitch_shift_equivariance(
            frame, self.sample_rate, alpha=1.25
        )

        # Harmonic signal should obey scaling relationship alpha=1.25 closely
        self.assertAlmostEqual(f0, 175.0, delta=2.0)
        self.assertLess(rel_err, 0.08)  # Relative error under 8%
        self.assertGreater(prob, 0.65)   # P(V) high

    def test_compute_frame_pitch_shift_equivariance_noise(self):
        frame = self.noise_signal[1000:1480]
        f0, rel_err, prob = compute_frame_pitch_shift_equivariance(
            frame, self.sample_rate, alpha=1.25
        )
        # Noise should have high equivariance discrepancy or very low periodicity
        self.assertLess(prob, 0.35)

    def test_analyze_signal_ssl_pitch_shift(self):
        res = analyze_signal_ssl_pitch_shift(
            self.voiced_signal, self.sample_rate, frame_duration_ms=30.0, hop_duration_ms=10.0
        )
        labels = res["labels"]
        f0_contour = res["f0_contour"]

        # Middle active frames should be labeled as voiced
        voiced_ratio_middle = np.mean(labels[30:70] == "v")
        self.assertGreater(voiced_ratio_middle, 0.85)

        # Silence regions should be unvoiced
        self.assertTrue(np.all(labels[:15] == "uv"))
        self.assertTrue(np.all(labels[85:] == "uv"))

        # Estimated pitch in voiced frames should be around 175 Hz
        valid_pitches = f0_contour[30:70]
        valid_pitches = valid_pitches[valid_pitches > 0]
        self.assertAlmostEqual(float(np.mean(valid_pitches)), 175.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
