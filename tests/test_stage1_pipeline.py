"""Regression tests for stage 1 input and pipeline correctness (unittest only)."""
import importlib.util
import io
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from src.core.audio import load_wav, compute_ste, frame_signal
from src.core.acf import find_f0_acf
from src.core.amdf import find_f0_amdf
from src.core.cepstrum import find_f0_cepstrum
from src.core.pitch_detector import PitchDetector
from src.core.processing import center_frame, smooth_voiced_segments
from src.plugins import (
    BandpassFilterPlugin, CenterClippingPlugin, EnergyExtensionPlugin,
    PluginPitchDetector, ViterbiTrackingPlugin,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"acf": find_f0_acf, "amdf": find_f0_amdf, "cepstrum": find_f0_cepstrum}


def read_pcm(samples):
    buffer = io.BytesIO()
    wavfile.write(buffer, 16000, samples)
    buffer.seek(0)
    return load_wav(buffer)[1]


def speech_like(sr=16000, seconds=.3, frequency=180):
    t = np.arange(round(sr * seconds)) / sr
    rng = np.random.default_rng(23)
    return .3 * np.sin(2 * np.pi * frequency * t) + .08 * np.sin(4 * np.pi * frequency * t) + rng.normal(0, .001, len(t))


class AudioTests(unittest.TestCase):
    def test_unsigned_pcm_silence_is_zero_in_mono_and_stereo(self):
        for shape in [(1600,), (1600, 2)]:
            with self.subTest(shape=shape):
                np.testing.assert_array_equal(read_pcm(np.full(shape, 128, dtype=np.uint8)), 0.)

    def test_unsigned_pcm_is_centered_before_channel_averaging(self):
        mono = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        np.testing.assert_allclose(read_pcm(mono), (mono.astype(float) - 128) / 128)
        stereo = np.array([[0, 255], [128, 128], [192, 128]], dtype=np.uint8)
        np.testing.assert_allclose(read_pcm(stereo), np.mean((stereo.astype(float) - 128) / 128, axis=1))

    def test_signed_pcm_stereo_keeps_the_same_scale_as_mono(self):
        mono = np.array([-32768, -8192, 0, 8192, 32767], dtype=np.int16)
        np.testing.assert_allclose(read_pcm(mono), mono.astype(float) / 32768)
        np.testing.assert_array_equal(read_pcm(np.column_stack([mono, mono])), read_pcm(mono))

    def test_ste_ignores_constant_dc(self):
        x = speech_like()
        np.testing.assert_allclose(compute_ste(x, 16000)[0], compute_ste(x + 3, 16000)[0], atol=1e-14)
        np.testing.assert_array_equal(compute_ste(np.ones(1600), 16000)[0], 0.)

    def test_bandpass_short_input_has_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Audio too short for bandpass"):
            BandpassFilterPlugin().pre_process_signal(np.ones(10), 16000)

    def test_cli_short_wav_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "short.wav"
            wavfile.write(source, 16000, np.zeros(10, dtype=np.int16))
            process = subprocess.run(
                [sys.executable, "-B", str(ROOT / "predict.py"), "-i", str(source), "-o", str(Path(directory) / "out")],
                capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
            )
            self.assertEqual(process.returncode, 2)
            self.assertIn("Audio too short for analysis", process.stderr)
            self.assertNotIn("Traceback", process.stderr)
            self.assertFalse((Path(directory) / "out").exists())


class DetectorTests(unittest.TestCase):
    def test_zero_and_dc_are_silence_in_both_detectors(self):
        for method, offset, ratio, plugin in itertools.product(METHODS, [0., .5], [0., .008], [False, True]):
            with self.subTest(method=method, offset=offset, ratio=ratio, plugin=plugin):
                base = PitchDetector(method=method, ste_silence_ratio=ratio)
                detector = PluginPitchDetector(base, plugins=[BandpassFilterPlugin(), EnergyExtensionPlugin(), ViterbiTrackingPlugin()]) if plugin else base
                result = detector.process_signal(np.full(1600, offset), 16000)
                self.assertEqual(result["num_silence"], result["num_frames"])
                self.assertEqual(result["num_voiced"], 0)
                np.testing.assert_array_equal(result["f0_contour"], 0.)

    def test_low_level_extractors_do_not_assign_pitch_to_dc(self):
        for name, extract in METHODS.items():
            for value in [0., .7]:
                with self.subTest(method=name, value=value):
                    self.assertEqual(extract(np.full(400, value), 16000)[0], 0.)

    def test_dc_shift_does_not_change_pitch_or_labels(self):
        x = speech_like()
        for method in METHODS:
            for plugin in [False, True]:
                with self.subTest(method=method, plugin=plugin):
                    base = PitchDetector(method=method)
                    detector = PluginPitchDetector(base, plugins=[CenterClippingPlugin(), ViterbiTrackingPlugin()]) if plugin else base
                    first = detector.process_signal(x, 16000)
                    shifted = detector.process_signal(x + 2, 16000)
                    np.testing.assert_array_equal(first["labels"], shifted["labels"])
                    np.testing.assert_allclose(first["f0_contour"], shifted["f0_contour"], atol=1e-5)

    def test_empty_or_short_detector_input_is_rejected(self):
        for plugin in [False, True]:
            detector = PluginPitchDetector() if plugin else PitchDetector()
            for length in [0, 10, 399]:
                with self.subTest(plugin=plugin, length=length):
                    with self.assertRaisesRegex(ValueError, "Audio too short for analysis"):
                        detector.process_signal(np.zeros(length), 16000)

    def test_baseline_and_empty_plugin_pipeline_agree(self):
        x = speech_like()
        for method in METHODS:
            with self.subTest(method=method):
                base = PitchDetector(method=method)
                baseline = base.process_signal(x, 16000)
                wrapped = PluginPitchDetector(base).process_signal(x, 16000)
                np.testing.assert_array_equal(baseline["labels"], wrapped["labels"])
                np.testing.assert_allclose(baseline["f0_contour"], wrapped["f0_contour"], atol=1e-6)

    def test_every_extractor_stays_in_the_requested_range_after_refinement(self):
        for sr, frequency, (method, extract) in itertools.product(
            [8000, 16000, 44100, 48000], [70., 70.1, 399., 400.], METHODS.items()
        ):
            with self.subTest(sr=sr, frequency=frequency, method=method):
                frame = speech_like(sr, .025, frequency)
                f0, _, _ = extract(frame, sr)
                self.assertTrue(f0 == 0 or 70 <= f0 <= 400, f0)


class MedianTests(unittest.TestCase):
    def test_filter_preserves_gap_and_cannot_read_across_it(self):
        contour = np.array([100., 100., 300., 0., 0., 0., 100.])
        np.testing.assert_array_equal(smooth_voiced_segments(contour), contour)
        changed = contour.copy()
        changed[-1] = 350
        np.testing.assert_array_equal(smooth_voiced_segments(changed)[:3], contour[:3])

    def test_filter_still_removes_an_interior_spike(self):
        np.testing.assert_array_equal(smooth_voiced_segments([100, 100, 300, 100, 100]), [100] * 5)

    def test_actual_detector_paths_preserve_segment_boundaries(self):
        expected = np.array([100., 100., 300., 0., 0., 0., 100.])
        x = speech_like(seconds=(400 + 6 * 160) / 16000)
        for plugin, module in [(False, "src.core.pitch_detector"), (True, "src.plugins.plugin_detector")]:
            values = [(value, .9 if value else .1, 100) for value in expected]
            base = PitchDetector(ste_silence_ratio=0)
            detector = PluginPitchDetector(base, plugins=[EnergyExtensionPlugin(min_voiced_len=100)]) if plugin else base
            with self.subTest(plugin=plugin), patch(module + ".find_f0_acf", side_effect=values):
                result = detector.process_signal(x, 16000)
                np.testing.assert_array_equal(result["f0_contour"], expected)


class ViterbiTests(unittest.TestCase):
    def test_tracker_receives_exact_clipped_frames_and_detector_configuration(self):
        x = speech_like(seconds=.4, frequency=230)
        tracker = ViterbiTrackingPlugin()
        clipper = CenterClippingPlugin()
        base = PitchDetector(frame_duration_ms=30, hop_duration_ms=20,
                             f0_min=200, f0_max=300, threshold=-1, use_median_filter=False)
        with patch.object(tracker, "_extract_acf_candidates", wraps=tracker._extract_acf_candidates) as extract:
            result = PluginPitchDetector(base, plugins=[clipper, tracker]).process_signal(x, 16000)
        self.assertGreater(extract.call_count, 1)
        self.assertEqual(result["pitch_frames"].shape[1], 480)
        np.testing.assert_allclose(np.diff(result["frame_times"]), .020)
        raw_frames, _ = frame_signal(x - x.mean(), 16000, 30, 20)
        for index, call in enumerate(extract.call_args_list):
            expected = clipper.pre_process_frame(center_frame(raw_frames[index]), 16000)
            np.testing.assert_array_equal(call.args[0], expected)
            self.assertEqual(call.kwargs, {"f0_min": 200, "f0_max": 300})
        positive = result["f0_contour"][result["f0_contour"] > 0]
        self.assertTrue(np.all((positive >= 200) & (positive <= 300)))

    def test_tracker_cannot_restore_pitch_outside_a_custom_range(self):
        base = PitchDetector(method="amdf", f0_min=200, f0_max=300, threshold=1, use_median_filter=False)
        result = PluginPitchDetector(base, plugins=[ViterbiTrackingPlugin()]).process_signal(
            speech_like(seconds=.2, frequency=150), 16000
        )
        voiced = result["f0_contour"][result["labels"] == "v"]
        self.assertGreater(len(voiced), 1)
        self.assertTrue(np.all((voiced >= 200) & (voiced <= 300)))

    def test_tracker_matches_brute_force_path_cost(self):
        tracker = ViterbiTrackingPlugin()
        candidates = [[{"f0": 100 + t * 5, "cost": .2}, {"f0": 200 + t * 10, "cost": .1}] for t in range(3)]
        def cost(path):
            return sum(c["cost"] for c in path) + sum(
                tracker._transition_cost(path[t - 1]["f0"], path[t]["f0"]) for t in range(1, len(path))
            )
        best = min(itertools.product(*candidates), key=cost)
        self.assertEqual(tracker._viterbi_segment(candidates), [c["f0"] for c in best])

    def test_tracker_requires_aligned_preprocessed_frames(self):
        result = {"labels": np.array(["v", "v"]), "f0_contour": np.array([150., 150.])}
        with self.assertRaisesRegex(ValueError, "preprocessed pitch_frames"):
            ViterbiTrackingPlugin().post_process_results(result)


class EnergyTests(unittest.TestCase):
    def result(self, method, gap_length=2, kind="uv", supported=True):
        count = 6 + gap_length
        threshold = .05 if method == "cepstrum" else .44
        metric = (.2 if supported else 1.) if method == "amdf" else (.8 if supported else 0.)
        return {
            "labels": np.array(["v"] * 3 + [kind] * gap_length + ["v"] * 3, dtype=object),
            "f0_contour": np.array([200.] * 3 + [0.] * gap_length + [203.] * 3),
            "candidate_f0": np.array([200.] * 3 + [201., 202.][:gap_length] + [203.] * 3),
            "peak_values": np.full(count, metric), "ste": np.ones(count),
            "ste_thresh": .01, "threshold": threshold, "method": method,
            "f0_min": 70, "f0_max": 400, "use_median_filter": False,
        }

    def apply(self, result):
        # Isolate gap filling from boundary expansion.
        return EnergyExtensionPlugin(min_voiced_len=100).post_process_results(result)

    def test_gap_fill_requires_periodicity_for_every_method(self):
        for method, length in itertools.product(METHODS, [1, 2]):
            with self.subTest(method=method, length=length):
                result = self.apply(self.result(method, length, supported=False))
                self.assertTrue(np.all(result["labels"][3:3 + length] == "uv"))

    def test_gap_fill_never_crosses_silence(self):
        result = self.apply(self.result("acf", kind="sil"))
        self.assertTrue(np.all(result["labels"][3:5] == "sil"))

    def test_gap_fill_requires_pitch_continuity(self):
        result = self.result("acf")
        result["f0_contour"][:3] = 100
        result["f0_contour"][5:] = 300
        result["candidate_f0"][3:5] = [166, 233]
        self.assertTrue(np.all(self.apply(result)["labels"][3:5] == "uv"))

    def test_gap_fill_rejects_out_of_range_candidates(self):
        result = self.result("acf")
        result["candidate_f0"][3:5] = 500
        self.assertTrue(np.all(self.apply(result)["labels"][3:5] == "uv"))

    def test_supported_one_and_two_frame_gaps_are_filled(self):
        for method, length in itertools.product(METHODS, [1, 2]):
            with self.subTest(method=method, length=length):
                result = self.apply(self.result(method, length))
                self.assertTrue(np.all(result["labels"] == "v"))
                np.testing.assert_array_equal(result["f0_contour"][3:3 + length], [201., 202.][:length])


class BenchmarkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {}
        for number, filename in [(15, "15_evaluate_unsupervised_thresholds.py"), (16, "16_evaluate_advanced_unsupervised.py")]:
            spec = importlib.util.spec_from_file_location(f"stage1_benchmark_{number}", ROOT / "scripts" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[number] = module

    def test_benchmark_16_uses_the_same_positive_silence_floor(self):
        module = self.modules[16]
        with patch.object(module, "load_wav", return_value=(16000, np.zeros(1600), .1)):
            result = module.extract_features_for_file("memory.wav", "amdf")
        self.assertFalse(np.any(result["is_active"]))

    def test_benchmark_15_uses_the_same_positive_silence_floor(self):
        module = self.modules[15]
        with patch.object(module, "load_wav", return_value=(16000, np.zeros(1600), .1)):
            result = module.extract_all_and_active_metrics(["memory.wav"], "amdf")
        self.assertEqual(len(result["active_values"]), 0)

    def test_benchmark_16_median_preserves_voiced_segments(self):
        module = self.modules[16]
        expected = np.array([100., 100., 300., 0., 0., 0., 100.])
        labels = np.where(expected > 0, "v", "uv")
        file_data = {"wav_path": "memory.wav", "frame_times": np.arange(7) * .01}
        with patch.object(module, "evaluate_against_ground_truth", side_effect=lambda result, path: result):
            result = module.evaluate_custom_prediction(file_data, labels, expected)
        np.testing.assert_array_equal(result["f0_contour"], expected)


if __name__ == "__main__":
    unittest.main()
