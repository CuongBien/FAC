"""Champion-space fitting, native post-processing and frozen-run integration."""
import contextlib
from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from src.analysis.champion_unsupervised import (
    PRIMARY, ADAPTIVE, experiment_configuration, feature_spec, feature_contract,
    extract_signal_features, fit_champion_models, predict_decisions,
)
from src.analysis.protocols import threshold_labels
from src.analysis.validation import json_default, read_frozen, sha256
from src.pipelines import build_champion

ROOT = Path(__file__).resolve().parents[1]


def signal(rate=16000):
    t = np.arange(round(rate*.32))/rate
    rng = np.random.default_rng(8)
    x = .3*np.sin(2*np.pi*180*t)+.07*np.sin(2*np.pi*360*t)+rng.normal(0, .003, len(t))
    x[:round(.05*rate)] = x[-round(.05*rate):] = 0
    return x


def simple_features(method, settings):
    return {'feature_spec': feature_spec(method, settings),
            'feature_contract': feature_contract(method, settings).to_dict(),
            'periodicity_vals': np.r_[np.linspace(.1, .2, 6), np.linspace(.7, .8, 6)],
            'zcr_vals': np.r_[np.linspace(.8, .7, 6), np.linspace(.2, .1, 6)],
            'is_active': np.ones(12, dtype=bool), 'candidate_f0': np.full(12, 180.), 'ste': np.ones(12)}


class ChampionFeatureTests(unittest.TestCase):
    def test_external_control_matches_native_champion_at_both_sample_rates(self):
        settings = experiment_configuration()
        for rate in (16000, 44100):
            for method in ('acf', 'amdf', 'cepstrum'):
                with self.subTest(rate=rate, method=method):
                    x = signal(rate)
                    fd = extract_signal_features(x, rate, method, settings)
                    detector = build_champion(method)
                    expected = detector.process_signal(x, rate)
                    labels = threshold_labels(fd, method, detector.base_detector.threshold)
                    actual = detector.process_feature_labels(fd, labels)
                    for key in ('labels', 'frame_times', 'f0_contour', 'pitch_frames', 'peak_values'):
                        np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(fd['feature_spec']['frame_duration_ms'], 30.)
                    self.assertEqual(actual['applied_plugins'], expected['applied_plugins'])

    def test_raw_or_wrong_frame_features_cannot_use_a_frozen_model(self):
        settings = experiment_configuration()
        fd = simple_features('acf', settings)
        model = fit_champion_models([fd], 'acf', settings)
        wrong = deepcopy(fd)
        wrong['feature_spec']['frame_duration_ms'] = 25.
        with self.assertRaisesRegex(ValueError, 'feature space'):
            predict_decisions(wrong, model, settings)
        with self.assertRaisesRegex(ValueError, 'feature space'):
            fit_champion_models([wrong], 'acf', settings)

    def test_post_processing_rejects_different_clipping_parameters_or_bad_labels(self):
        settings = experiment_configuration()
        wrong = deepcopy(settings)
        wrong['champions']['cepstrum']['clipping_ratio'] = .2
        fd = extract_signal_features(signal(), 16000, 'cepstrum', wrong)
        with self.assertRaisesRegex(ValueError, 'pre-processing'):
            build_champion('cepstrum').process_feature_labels(fd, fd['labels'])
        fd = extract_signal_features(signal(), 16000, 'acf', settings)
        for labels in (['v'], ['bad']*fd['num_frames']):
            with self.assertRaisesRegex(ValueError, 'External labels'):
                build_champion('acf').process_feature_labels(fd, labels)

    def test_frozen_predictions_never_fit_and_do_not_mutate_the_model(self):
        settings = experiment_configuration()
        fd = simple_features('acf', settings)
        model = fit_champion_models([fd], 'acf', settings)
        before = json.dumps(model, sort_keys=True, default=json_default)
        with patch('src.analysis.champion_unsupervised.cluster_multifeature_2d', side_effect=AssertionError('refit')), \
             patch('src.analysis.champion_unsupervised.train_and_decode_2state_hmm', side_effect=AssertionError('refit')), \
             patch('src.analysis.champion_unsupervised.compute_gmm_em_1d', side_effect=AssertionError('refit')):
            labels, thresholds, adaptive = predict_decisions(fd, model, settings)
        self.assertEqual(tuple(labels), PRIMARY)
        self.assertEqual(set(thresholds), {'median', 'otsu', 'valley', 'kmeans', 'gmm', 'dynamic'})
        self.assertEqual(adaptive, {})
        self.assertEqual(json.dumps(model, sort_keys=True, default=json_default), before)

    def test_hmm_fitting_keeps_recording_boundaries(self):
        settings = experiment_configuration()
        fd = simple_features('acf', settings)
        model = fit_champion_models([fd, fd], 'acf', settings)
        self.assertEqual(model['hmm']['segments'], [(0, 12), (13, 25)])
        self.assertEqual(model['num_train_active'], 24)

    def test_dynamic_context_cannot_cross_a_silence_gap(self):
        settings = experiment_configuration()
        fd = simple_features('acf', settings)
        model = fit_champion_models([fd], 'acf', settings)
        fd['is_active'][5:7] = False
        _, first, _ = predict_decisions(fd, model, settings)
        fd['periodicity_vals'][7:] = 1000.
        fd['ste'][7:] = 1000.
        _, second, _ = predict_decisions(fd, model, settings)
        np.testing.assert_array_equal(first['dynamic'][:5], second['dynamic'][:5])

    def test_all_strategies_preserve_gate_range_and_fixed_energy_guard(self):
        settings = experiment_configuration()
        for method in ('acf', 'amdf', 'cepstrum'):
            fd = extract_signal_features(signal(), 16000, method, settings)
            model = fit_champion_models([fd], method, settings)
            index = np.flatnonzero(fd['is_active'])[2]
            fd['candidate_f0'][index] = 401.
            before = {key: value.copy() for key, value in fd.items() if isinstance(value, np.ndarray)}
            decisions, _, adaptive = predict_decisions(fd, model, settings, include_input_adaptation=True)
            self.assertEqual(set(decisions), set(PRIMARY+ADAPTIVE))
            self.assertEqual(set(adaptive), {'kmeans2d', 'hmm'})
            for name, labels in decisions.items():
                with self.subTest(method=method, strategy=name):
                    result = build_champion(method).process_feature_labels(fd, labels)
                    self.assertTrue(np.all(result['labels'][~fd['is_active']] == 'sil'))
                    self.assertNotEqual(result['labels'][index], 'v')
                    self.assertEqual(result['threshold'], settings['champions'][method]['threshold'])
                    voiced = result['f0_contour'][result['labels'] == 'v']
                    self.assertTrue(np.all((voiced >= 70) & (voiced <= 400)))
                    self.assertEqual(len(voiced), result['num_voiced'])
            for key, value in before.items():
                np.testing.assert_array_equal(fd[key], value)


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_train_freeze_evaluate_report_and_preserve_frozen_models(self):
        spec = importlib.util.spec_from_file_location('champion_runner', ROOT/'scripts/21_benchmark_champion_unsupervised.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            train, test, output = root/'train', root/'not_yet_test', root/'out'
            train.mkdir()
            wavfile.write(train/'a.wav', 16000, (signal()*32767).astype(np.int16))
            (train/'a.lab').write_text('F0mean 180\nF0std 10\nF0num 20\n', encoding='utf-8')
            args = SimpleNamespace(train_dir=str(train), test_dir=str(test), output_dir=str(output))
            # Fitting is permitted to hash LAB for provenance, but never evaluate it.
            from src.experiments import champion as backend
            with patch.object(backend, 'evaluate_result', side_effect=AssertionError('LAB statistics used during fit')):
                runner.train(args)
            self.assertFalse(test.exists())
            frozen_hash = sha256(output/'frozen_models.json')
            frozen = read_frozen(output/'frozen_config.json')
            self.assertEqual(frozen['experiment']['fit_population'], 'train_active_after_champion_preprocessing')
            with self.assertRaises(FileExistsError):
                runner.train(args)
            with patch.object(backend, 'experiment_configuration', return_value={}):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    backend.checked_freezes(output, str(train))
            test.mkdir()
            wavfile.write(test/'b.wav', 16000, (signal()*30000).astype(np.int16))
            (test/'b.lab').write_text('F0mean 181\nF0std 11\nF0num 19\n', encoding='utf-8')
            runner.evaluate(args)
            runner.report(args)
            self.assertEqual(sha256(output/'frozen_models.json'), frozen_hash)
            result = json.loads((output/'evaluation.json').read_text(encoding='utf-8'))
            from src.metrics.comparison import read_series, compare_series
            from src.experiments.metric_artifacts import verify_metric_artifact
            from src.experiments.store import verify_run
            metric_metadata = json.loads((output/'metrics.json').read_bytes())
            self.assertEqual(metric_metadata['contract']['metric_version'], 'legacy-v1')
            self.assertEqual(len(metric_metadata['series']), 72)
            gmm = read_series(output, 'test/acf/gmm')
            self.assertEqual(gmm.to_dict()['context']['fit']['population'], 'train_active')
            self.assertTrue(all(value['delta'] == 0 for value in compare_series(gmm, gmm)['aggregate'].values()))
            baseline = read_series(output, 'test/acf/baseline')
            with self.assertRaisesRegex(ValueError, 'feature_contract'):
                compare_series(baseline, gmm)
            command = [sys.executable, '-B', '-m', 'src.metrics', str(output), 'test/acf/gmm', str(output)]
            cli = subprocess.run(command+['test/acf/hmm'], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertEqual(json.loads(cli.stdout)['metric_version'], 'legacy-v1')
            refused = subprocess.run(command+['test/acf/baseline'], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(refused.returncode, 2)
            self.assertIn('feature_contract', refused.stderr)
            self.assertNotIn('Traceback', refused.stderr)
            saved_metrics = (output/'metrics.json').read_bytes()
            pinned = verify_run(output)
            metric_metadata['series'][0]['context']['feature_contract']['frame_duration_ms'] = 99.
            (output/'metrics.json').write_text(json.dumps(metric_metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'context/values'):
                verify_metric_artifact(output, pinned)
            (output/'metrics.json').write_bytes(saved_metrics)
            self.assertEqual(result['native_champion_replays_matched'], 6)
            self.assertEqual(len(result['pipelines']), 12)
            self.assertEqual(len(result['protocols']), 60)
            self.assertEqual(sum(item['group'] == 'train_fitted' for item in result['protocols']), 48)
            with (output/'per_file.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 72)
            for row in rows:
                self.assertTrue((output/row['contour_path']).is_file())
            self.assertTrue((output/'figures/score_matrix.png').is_file())
            self.assertEqual(len(list((output/'input_models').rglob('*.json'))), 6)
            with self.assertRaises(FileExistsError):
                runner.evaluate(args)


if __name__ == '__main__':
    unittest.main()
