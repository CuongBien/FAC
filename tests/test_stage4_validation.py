"""Checks for train-only selection, freeze integrity, caching and timing."""
import contextlib
from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

import predict
from src.pipelines import build_detector, build_champion
from src.pipeline.contracts import FeatureSpec
from src.core.pitch_detector import PitchDetector
from src.analysis.validation import (
    pitch_cache, freeze_json, read_frozen, select_trial, dataset_manifest, verify_dataset, json_default,
    verify_source_snapshot,
)
from src.analysis.protocols import (
    fit_models, predict_strategies, predict_frozen_cluster, predict_frozen_hmm, result_from_labels,
)
from src.plugins.post_processing.viterbi_tracking import ViterbiTrackingPlugin


ROOT = Path(__file__).resolve().parents[1]


def features(method='acf'):
    p = np.r_[np.linspace(.1, .2, 6), np.linspace(.7, .8, 6)]
    return {'feature_contract': FeatureSpec(method, 25., 10., 70., 400., .008, 'centered_all').to_dict(), 'periodicity_vals': p, 'zcr_vals': 1-p, 'is_active': np.ones(12, dtype=bool),
            'candidate_f0': np.full(12, 180.), 'ste': np.ones(12), 'frame_times': np.arange(12)*.01}


def candidate(score, plugins=(), test_score=100):
    return {'spec': {'plugins': list(plugins), 'frame_duration_ms': 25., 'threshold': .4},
            'metrics': {'final_score': score}, 'files': [{'final_score': score}], 'test_score': test_score}


class FreezeAndSelectionTests(unittest.TestCase):
    def test_freeze_detects_tampering_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'frozen.json'
            freeze_json(path, {'threshold': .4})
            self.assertEqual(read_frozen(path)['threshold'], .4)
            with self.assertRaises(FileExistsError):
                freeze_json(path, {'threshold': .5})
            path.write_text('{"threshold": 0.5}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_frozen(path)

    def test_selection_ignores_test_scores_and_excludes_invalid_train_scores(self):
        chosen = candidate(2., test_score=99.)
        self.assertIs(select_trial([candidate(None), candidate(np.nan), candidate(3., test_score=0.), chosen]), chosen)
        with self.assertRaises(ValueError):
            select_trial([candidate(None), candidate(np.inf)])

    def test_train_ties_prefer_simpler_configuration(self):
        simple = candidate(2.)
        self.assertIs(select_trial([candidate(2., ['BP']), simple]), simple)

    def test_changed_training_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'a.wav').write_bytes(b'original audio')
            (Path(folder)/'a.lab').write_text('F0mean 180', encoding='utf-8')
            manifest = dataset_manifest(folder)
            verify_dataset(folder, manifest)
            (Path(folder)/'a.lab').write_text('F0mean 181', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_dataset(folder, manifest)

    def test_train_phase_does_not_access_test_directory(self):
        spec = importlib.util.spec_from_file_location('stage4_runner', ROOT/'scripts/17_validate_and_calibrate.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            train = Path(folder)/'train'; train.mkdir()
            (train/'a.wav').write_bytes(b'only hashes read by this orchestration test')
            (train/'a.lab').write_text('F0mean 180', encoding='utf-8')
            test = Path(folder)/'nonexistent_test'
            accessed = []
            def extract(path, method):
                accessed.append(Path(path).parent)
                return features(method)
            def fake_trial(spec, paths, tag, *, cache=None):
                self.assertTrue(all(path.parent == train for path in paths))
                score = 1 + abs(spec['threshold']-.4) + len(spec['plugins'])*.01
                return {'id': tag, 'spec': spec, 'metrics': {'final_score': score},
                        'files': [{'file': 'a.wav', 'final_score': score}]}
            output = Path(folder)/'run'
            with patch.object(runner, 'trial', side_effect=fake_trial), \
                 patch.object(runner, 'extract_file_features', side_effect=extract):
                runner.train(SimpleNamespace(train_dir=str(train), test_dir=str(test), output_dir=str(output)))
            self.assertFalse(test.exists())
            self.assertEqual(accessed, [train]*3)
            frozen = read_frozen(output/'frozen_config.json')
            self.assertEqual(set(frozen['selected']), {'acf', 'amdf', 'cepstrum'})
            self.assertTrue((output/'frozen_models.sha256').exists())
            self.assertGreater(verify_source_snapshot(output/'training_source_snapshot.tar.gz', frozen['provenance']), 0)
            self.assertTrue((output/'runtime.json').exists())


class ComputationCacheTests(unittest.TestCase):
    def test_cache_preserves_results_and_restores_functions(self):
        import src.core.pitch_detector as core
        original = core.find_f0_acf
        t = np.arange(2560)/16000
        signal = .4*np.sin(2*np.pi*180*t)
        for method in ('acf', 'amdf', 'cepstrum'):
            expected = build_champion(method).process_signal(signal, 16000)
            with pitch_cache() as stats:
                first = build_champion(method).process_signal(signal, 16000)
                second = build_champion(method).process_signal(signal, 16000)
                np.testing.assert_array_equal(first['f0_contour'], expected['f0_contour'])
                np.testing.assert_array_equal(second['labels'], expected['labels'])
                self.assertGreater(stats['hits'], 0)
        self.assertIs(core.find_f0_acf, original)

    def test_cached_candidates_are_copied_and_include_pitch_bounds(self):
        frame = np.sin(2*np.pi*180*np.arange(480)/16000)
        tracker = ViterbiTrackingPlugin()
        with pitch_cache() as stats:
            first = tracker._extract_acf_candidates(frame, 16000, f0_min=100, f0_max=300)
            self.assertTrue(first)
            first[0]['f0'] = 999.
            second = tracker._extract_acf_candidates(frame, 16000, f0_min=100, f0_max=300)
            self.assertNotEqual(second[0]['f0'], 999.)
            misses = stats['misses']
            tracker._extract_acf_candidates(frame, 16000, f0_min=200, f0_max=300)
            self.assertEqual(stats['misses'], misses+1)


class FrozenProtocolTests(unittest.TestCase):
    def test_models_keep_recording_boundaries(self):
        model = fit_models([features(), features()], 'acf')
        self.assertEqual(model['hmm']['segments'], [(0, 12), (13, 25)])
        self.assertEqual(model['num_train_active'], 24)

    def test_frozen_prediction_never_refits(self):
        fd = features()
        model = fit_models([fd], 'acf')
        with patch('src.analysis.protocols.cluster_multifeature_2d', side_effect=AssertionError('must not refit')), \
             patch('src.analysis.protocols.train_and_decode_2state_hmm', side_effect=AssertionError('must not refit')):
            self.assertEqual(len(predict_frozen_cluster(fd, model['cluster_2d'])), 12)
            self.assertEqual(len(predict_frozen_hmm(fd, model['hmm'])), 12)

    def test_all_protocols_preserve_silence_range_and_frozen_parameters(self):
        for method in ('acf', 'amdf', 'cepstrum'):
            fd = features(method)
            model = fit_models([deepcopy(fd)], method)
            before = json.dumps(model, default=json_default, sort_keys=True)
            fd['is_active'][3] = False
            fd['candidate_f0'][7:9] = [69., 401.]
            strategies = predict_strategies(fd, model)
            self.assertEqual(len(strategies), 17)
            for name, labels in strategies.items():
                with self.subTest(method=method, strategy=name):
                    self.assertEqual(labels[3], 'sil')
                    self.assertTrue(np.all(labels[7:9] != 'v'))
                    result = result_from_labels(fd, labels)
                    self.assertEqual(result['num_voiced'], int(np.sum(labels == 'v')))
                    self.assertEqual(result['f0_contour'][3], 0.)
            self.assertEqual(before, json.dumps(model, default=json_default, sort_keys=True))


class MixedFrameOutputTests(unittest.TestCase):
    def test_cli_keeps_each_methods_frame_times_and_lengths(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder)/'tone.wav'
            time = np.arange(6480)/16000
            wavfile.write(path, 16000, (.4*np.sin(2*np.pi*180*time)*32767).astype(np.int16))
            detectors = {'frame25': PitchDetector('acf'), 'frame30': build_detector('acf', frame_duration_ms=30)}
            results = predict.process_single_file(str(path), detectors, folder)
            with (Path(folder)/'tone_pitch_prediction.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            first_times = {}
            for name in detectors:
                own = [row for row in rows if row['Method'] == name]
                self.assertEqual(len(own), results[name]['total_frames'])
                first_times[name] = float(own[0]['Time_Sec'])
            self.assertNotEqual(results['frame25']['total_frames'], results['frame30']['total_frames'])
            self.assertEqual(first_times, {'frame25': .0125, 'frame30': .015})


if __name__ == '__main__':
    unittest.main()
