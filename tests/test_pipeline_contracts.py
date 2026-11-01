"""Feature-space boundaries and cache isolation across the shared engine."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import subprocess
import sys
import unittest

import numpy as np

from src.analysis.champion_unsupervised import (
    experiment_configuration, extract_signal_features, fit_champion_models, predict_decisions,
)
from src.analysis.protocols import fit_models, predict_strategies
from src.core import PitchDetector
from src.core.audio import frame_signal
from src.pipeline.cache import PitchCache, cached_call, pitch_cache
from src.pipeline.contracts import FeatureSpec, FrameFeatures, PredictionResult
from src.pipeline.engine import extract_signal
from src.pipeline.features import extract_raw_features
from src.pipelines import build_champion
from src.plugins import PluginPitchDetector
from src.plugins.base import PreProcessingPlugin, DecisionPlugin
from src.plugins.post_processing.viterbi_tracking import ViterbiTrackingPlugin


def signal(rate=16000):
    time = np.arange(round(rate*.24))/rate
    values = .3*np.sin(2*np.pi*180*time)+.02*np.cos(2*np.pi*311*time)
    values[:round(rate*.07)] *= .002
    return values


class FeatureContractTests(unittest.TestCase):
    def test_contract_roundtrip_is_immutable_and_rejects_inconsistent_policy(self):
        spec = FeatureSpec('acf', 30., 20., 100., 300., .01, 'plugins',
                           '[{"name":"clip","parameters":{"ratio":0.4}}]')
        self.assertEqual(FeatureSpec.from_dict(json.loads(json.dumps(spec.to_dict()))), spec)
        document = spec.to_dict()
        document['pre_processing'][0]['parameters']['ratio'] = .2
        self.assertEqual(spec.to_dict()['pre_processing'][0]['parameters']['ratio'], .4)
        for key, value in [('pitch_population', 'all'), ('schema_version', True), ('extra', 1)]:
            bad = spec.to_dict()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                FeatureSpec.from_dict(bad)
        for field, value in [('f0_min', 400.), ('hop_duration_ms', 0.),
                             ('frame_duration_ms', True), ('ste_silence_ratio', float('nan'))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(spec, **{field: value})

    def test_native_skips_inactive_pitch_but_historical_profiles_measure_all(self):
        x = signal()
        raw = extract_raw_features(x, 16000, 'acf')
        detector = PluginPitchDetector(PitchDetector())
        native = detector.process_signal(x, 16000)
        inactive = ~native['is_active']
        self.assertGreater(int(inactive.sum()), 0)
        self.assertTrue(np.all(native['candidate_f0'][inactive] == 0))
        self.assertTrue(np.all(native['periodicity_vals'][inactive] == 0))
        self.assertTrue(np.all(raw['candidate_f0'][~raw['is_active']] > 0))
        self.assertEqual(raw['feature_contract']['pitch_population'], 'all')
        self.assertEqual(native['feature_contract']['pitch_population'], 'active')
        self.assertNotIn('labels', raw)
        self.assertNotIn('f0_contour', raw)

    def test_raw_profiles_keep_padding_and_native_rejects_short_input(self):
        short = signal()[:80]
        raw = extract_raw_features(short, 16000, 'amdf')
        self.assertEqual(raw['num_frames'], 1)
        self.assertEqual(raw['pitch_frames'].shape, (1, 400))
        with self.assertRaisesRegex(ValueError, 'too short'):
            PitchDetector('amdf').process_signal(short, 16000)

    def test_raw_and_centered_metric_inputs_are_explicit(self):
        x = signal()+.7
        seen = []
        def measure(frame, rate, **options):
            seen.append(frame.copy())
            return 180., .8, 89
        for profile in ('raw_all', 'centered_all', 'baseline', 'plugins'):
            spec = FeatureSpec('acf', 25., 10., 70., 400., .008, profile)
            seen.clear()
            result = extract_signal(x, 16000, spec, threshold=.4, extractors={'acf': measure})
            frames, _ = frame_signal(x-x.mean() if spec.native else x, 16000)
            indices = np.flatnonzero(result['is_active']) if spec.native else range(len(frames))
            for actual, index in zip(seen, indices):
                expected = frames[index]
                if spec.center_pitch_frame:
                    expected = expected-expected.mean()
                np.testing.assert_array_equal(actual, expected)

    def test_array_and_prediction_contracts_reject_misalignment_and_invalid_counts(self):
        fd = extract_raw_features(signal(), 16000, 'acf')
        spec = FeatureSpec.from_dict(fd['feature_contract'])
        for key in ('frame_times', 'candidate_f0', 'pitch_frames'):
            bad = {**fd, key: fd[key][:-1]}
            with self.subTest(key=key), self.assertRaises(ValueError):
                FrameFeatures(spec, bad)
        for update in ({'is_active': ~fd['is_active']}, {'num_frames': 1}):
            with self.assertRaises(ValueError):
                FrameFeatures(spec, {**fd, **update})
        valid = {'labels': np.array(['v', 'uv']), 'f0_contour': np.array([180., 0.]),
                 'frame_times': np.array([.01, .02]), 'num_voiced': 1}
        self.assertEqual(PredictionResult(valid, 70., 400.).to_dict()['num_voiced'], 1)
        for update in ({'f0_contour': np.array([180., 1.])}, {'num_voiced': 2},
                       {'labels': np.array(['bad', 'uv'])}):
            with self.assertRaises(ValueError):
                PredictionResult({**valid, **update}, 70., 400.)

    def test_model_records_actual_framing_and_rejects_raw_processed_mix(self):
        fd = extract_raw_features(signal(), 16000, 'acf', 30., 20., 100., 300.)
        model = fit_models([fd], 'acf')
        self.assertEqual((model['frame_duration_ms'], model['hop_duration_ms']), (30., 20.))
        self.assertEqual((model['f0_min'], model['f0_max']), (100., 300.))
        self.assertEqual(model['fit_populations'], ['train_all', 'train_active'])
        bad = deepcopy(fd)
        bad['feature_contract']['profile'] = 'raw_all'
        for action in (lambda: fit_models([fd, bad], 'acf'), lambda: fit_models([fd], 'amdf'),
                       lambda: fit_models([{}], 'acf'), lambda: predict_strategies(bad, model)):
            with self.assertRaises(ValueError):
                action()
        self.assertEqual(len(predict_strategies(fd, model)), 17)

    def test_new_raw_models_enforce_declared_pitch_bounds(self):
        fd = extract_raw_features(signal(), 16000, 'acf', 30., 10., 120., 250.)
        model = fit_models([fd], 'acf')
        index = np.flatnonzero(fd['is_active'])[0]
        fd['candidate_f0'][index] = 110.
        for labels in predict_strategies(fd, model).values():
            self.assertNotEqual(labels[index], 'v')

    def test_legacy_raw_model_adapter_requires_its_recorded_framing(self):
        fd = extract_raw_features(signal(), 16000, 'acf')
        model = fit_models([fd], 'acf')
        expected = predict_strategies(fd, model)
        for key in ('feature_contract', 'fit_populations', 'f0_min', 'f0_max'):
            del model[key]
        actual = predict_strategies(fd, model)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        wrong = extract_raw_features(signal(), 16000, 'acf', 30.)
        with self.assertRaisesRegex(ValueError, 'feature space'):
            predict_strategies(wrong, model)

    def test_pipeline_imports_do_not_load_analysis_or_reporting(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; import src.pipeline.engine; import src.pipeline.champion; '
            'from src.core import PitchDetector; '
            'assert not any(n == "src.analysis" or n.startswith("src.analysis.") '
            'or n.startswith("matplotlib") for n in sys.modules)'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_champion_rejects_changed_gate_or_zcr_space(self):
        settings = experiment_configuration()
        fd = extract_signal_features(signal(), 16000, 'acf', settings)
        model = fit_champion_models([fd], 'acf', settings)
        for key, value in [('ste_silence_ratio', .1), ('zcr_source', 'centered_frame_all')]:
            bad = deepcopy(fd)
            bad['feature_contract'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                predict_decisions(bad, model, settings)
            with self.assertRaises(ValueError):
                build_champion('acf').process_feature_labels(bad, bad['labels'])
        wrong_settings = deepcopy(settings)
        wrong_settings['f0_min'] = 100.
        with self.assertRaisesRegex(ValueError, 'feature space'):
            predict_decisions(fd, model, wrong_settings)

    def test_decision_hooks_interleave_with_frame_processing_and_reset(self):
        events = []
        class Pre(PreProcessingPlugin):
            def reset(self):
                events.append('reset_pre')
            def pre_process_frame(self, frame, rate):
                events.append('pre')
                return frame
        class Decision(DecisionPlugin):
            def reset(self):
                events.append('reset_decision')
            def adjust_frame_decision(self, index, metric, f0, voiced, context):
                events.append('silence' if context['is_silence'] else 'decision')
                return voiced, f0
        detector = PluginPitchDetector(PitchDetector(), plugins=[Pre(), Decision()])
        for _ in range(2):
            events.clear()
            result = detector.process_signal(signal(), 16000)
            expected = ['reset_pre', 'reset_decision']
            for active in result['is_active']:
                expected.extend(['pre', 'decision'] if active else ['silence'])
            self.assertEqual(events, expected)
        with self.assertRaisesRegex(ValueError, 'pre-processing'):
            extract_signal(signal(), 16000, FeatureSpec('acf', 25., 10., 70., 400., .008, 'plugins'),
                           pre_processors=[Pre()])


class ExplicitCacheTests(unittest.TestCase):
    def test_explicit_cache_matches_uncached_at_both_rates_and_keeps_functions(self):
        import src.core.pitch_detector as core
        import src.plugins.plugin_detector as plugin
        original = (core.find_f0_acf, plugin.find_f0_acf, ViterbiTrackingPlugin._extract_acf_candidates)
        for rate in (16000, 44100):
            for method in ('acf', 'amdf', 'cepstrum'):
                cache = PitchCache()
                expected = build_champion(method).process_signal(signal(rate), rate)
                for _ in range(2):
                    actual = build_champion(method, cache=cache).process_signal(signal(rate), rate)
                    for key in ('labels', 'frame_times', 'candidate_f0', 'f0_contour', 'pitch_frames'):
                        np.testing.assert_array_equal(actual[key], expected[key])
                self.assertGreater(cache['hits'], 0)
        with pitch_cache():
            self.assertEqual(original, (core.find_f0_acf, plugin.find_f0_acf,
                                        ViterbiTrackingPlugin._extract_acf_candidates))

    def test_cache_keys_include_frame_dtype_options_and_candidate_configuration(self):
        cache = PitchCache()
        def value(frame, option=1):
            return [{'value': float(frame.sum())*option}]
        frame = np.arange(8, dtype=np.float64)
        first = cached_call(cache, value, frame)
        first[0]['value'] = -1
        self.assertEqual(cached_call(cache, value, frame)[0]['value'], 28.)
        self.assertEqual(dict(cache), {'hits': 1, 'misses': 1})
        cached_call(cache, value, frame.astype(np.float32))
        cached_call(cache, value, frame, option=2)
        cached_call(cache, value, frame+1)
        self.assertEqual(cache['misses'], 4)
        tracker = ViterbiTrackingPlugin(cache=cache)
        frame = signal()[:480]
        tracker._extract_acf_candidates(frame, 16000)
        misses = cache['misses']
        tracker.max_candidates = 2
        tracker._extract_acf_candidates(frame, 16000)
        self.assertEqual(cache['misses'], misses+1)

    def test_explicit_caches_and_nested_legacy_contexts_do_not_share_state(self):
        def value(frame):
            return frame.sum()
        frame = np.arange(4.)
        first, second = PitchCache(), PitchCache()
        cached_call(first, value, frame)
        cached_call(second, value, frame)
        self.assertEqual(dict(first), dict(second))
        with pitch_cache() as outer:
            cached_call(None, value, frame)
            with pitch_cache() as inner:
                cached_call(None, value, frame)
            cached_call(None, value, frame)
            self.assertEqual(dict(outer), {'hits': 1, 'misses': 1})
            self.assertEqual(dict(inner), {'hits': 0, 'misses': 1})

    def test_legacy_cache_contexts_are_isolated_between_interleaved_tasks(self):
        async def run():
            first_ready, second_ready = asyncio.Event(), asyncio.Event()
            def value(frame):
                return frame.sum()
            async def worker(ready, wait):
                with pitch_cache() as cache:
                    cached_call(None, value, np.arange(4.))
                    ready.set()
                    await wait.wait()
                    cached_call(None, value, np.arange(4.))
                    return dict(cache)
            return await asyncio.gather(worker(first_ready, second_ready), worker(second_ready, first_ready))
        self.assertEqual(asyncio.run(run()), [{'hits': 1, 'misses': 1}]*2)


if __name__ == '__main__':
    unittest.main()
