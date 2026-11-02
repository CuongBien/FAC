"""Scientific checks for the retained decisions and their shared feature space."""
from copy import deepcopy
import unittest
import numpy as np

from src.analysis.unsupervised_threshold import compute_otsu_threshold, compute_kmeans_threshold
from src.analysis.advanced_unsupervised import cluster_multifeature_2d, compute_sliding_window_adaptive_threshold
from src.analysis.champion_unsupervised import (
    PRIMARY, ADAPTIVE, experiment_configuration, fit_champion_models, predict_decisions,
)
from src.pipeline.champion import extract_signal_features
from src.pipelines import build_champion


class ThresholdTests(unittest.TestCase):
    def test_otsu_separates_two_groups_and_ignores_nonfinite_values(self):
        values = np.array([.1, .12, .15, .8, .85, .9, np.nan, np.inf])
        threshold = compute_otsu_threshold(values)
        self.assertTrue(.15 <= threshold < .8)
        self.assertEqual(compute_otsu_threshold(np.ones(10)), 1.)

    def test_kmeans_midpoint_and_degenerate_data(self):
        self.assertAlmostEqual(compute_kmeans_threshold(np.array([0., 0., 1., 1.])), .5)
        self.assertEqual(compute_kmeans_threshold(np.ones(10)), 1.)
        with self.assertRaises(ValueError):
            compute_kmeans_threshold(np.arange(10.), k=3)

    def test_2d_mapping_matches_acoustic_direction_and_feature_units(self):
        p = np.array([.1, .12, .15, .8, .85, .9])
        z = np.array([.4, .45, .5, .03, .04, .05])
        labels, model = cluster_multifeature_2d(p, z)
        np.testing.assert_array_equal(labels, ['uv']*3+['v']*3)
        scaled, _ = cluster_multifeature_2d(p*100, z*1e-6)
        np.testing.assert_array_equal(labels, scaled)
        amdf, _ = cluster_multifeature_2d(1-p, z, 'amdf')
        np.testing.assert_array_equal(labels, amdf)
        self.assertEqual(model['status'], 'ok')

    def test_2d_constant_input_does_not_invent_a_voiced_cluster(self):
        labels, model = cluster_multifeature_2d(np.ones(8), np.ones(8))
        self.assertEqual(model['status'], 'degenerate')
        self.assertTrue(np.all(labels == 'uv'))

    def test_dynamic_weight_and_window_match_definition(self):
        values = np.arange(6., dtype=float)
        ste = np.ones(6)
        result = compute_sliding_window_adaptive_threshold(values, ste, 2., window_frames=4, alpha=.3)
        # Even window uses two past frames, current frame and one future frame.
        self.assertAlmostEqual(result[3], .7*2+.3*np.median(values[1:5]))
        np.testing.assert_array_equal(
            compute_sliding_window_adaptive_threshold(values, ste, 2., alpha=0), np.full(6, 2.))


class SelectedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fs = 16000
        t = np.arange(4800)/fs
        cls.signal = .3*np.sin(2*np.pi*180*t)
        cls.signal[:640] = 0
        cls.settings = experiment_configuration()

    def test_each_method_fits_only_selected_models_and_replays_control(self):
        self.assertEqual(PRIMARY, ('otsu', 'kmeans', 'kmeans2d', 'dynamic'))
        self.assertEqual(ADAPTIVE, ('kmeans2d_input_active',))
        for method in ('acf', 'amdf', 'cepstrum'):
            with self.subTest(method=method):
                features = extract_signal_features(self.signal, 16000, method, self.settings)
                model = fit_champion_models([features], method, self.settings)
                self.assertEqual(set(model['thresholds']), {'otsu', 'kmeans'})
                decisions, thresholds, adaptive = predict_decisions(features, model, self.settings, True)
                self.assertEqual(set(decisions), set(PRIMARY+ADAPTIVE))
                self.assertEqual(set(adaptive), {'kmeans2d'})
                detector = build_champion(method)
                native = detector.process_signal(self.signal, 16000)
                replay = detector.process_feature_labels(features, native['labels'])
                np.testing.assert_array_equal(native['f0_contour'], replay['f0_contour'])
                np.testing.assert_array_equal(native['frame_times'], replay['frame_times'])
                for labels in decisions.values():
                    result = detector.process_feature_labels(features, labels)
                    self.assertTrue(np.all(result['f0_contour'][result['labels'] != 'v'] == 0))

    def test_dynamic_uses_frozen_otsu_and_resets_at_inactive_gaps(self):
        features = extract_signal_features(self.signal, 16000, 'acf', self.settings)
        model = fit_champion_models([features], 'acf', self.settings)
        before = deepcopy(model)
        features = deepcopy(features)
        features['is_active'][10] = False
        _, thresholds, _ = predict_decisions(features, model, self.settings)
        # The inactive frame retains the global Otsu value.
        self.assertEqual(thresholds['dynamic'][10], model['thresholds']['otsu'])
        expected = compute_sliding_window_adaptive_threshold(
            features['periodicity_vals'][11:], features['ste'][11:], model['thresholds']['otsu'],
            window_frames=50, alpha=.3)
        np.testing.assert_allclose(thresholds['dynamic'][11:], expected)
        self.assertEqual(model['thresholds'], before['thresholds'])

    def test_incompatible_training_and_prediction_spaces_are_rejected(self):
        features = extract_signal_features(self.signal, 16000, 'acf', self.settings)
        model = fit_champion_models([features], 'acf', self.settings)
        changed = deepcopy(features)
        changed['feature_contract']['frame_duration_ms'] = 25.
        with self.assertRaises(ValueError):
            predict_decisions(changed, model, self.settings)
        with self.assertRaises(ValueError):
            build_champion('acf').process_feature_labels(changed, features['is_active'])


if __name__ == '__main__':
    unittest.main()
