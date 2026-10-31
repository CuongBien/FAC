"""Independent numerical checks for stage 2 threshold and sequence methods."""
import itertools
import unittest

import numpy as np
from scipy.stats import norm

from src.analysis.advanced_unsupervised import (
    _gaussian_intersection, _hmm_posteriors, _hmm_viterbi,
    compute_gmm_em_1d, cluster_multifeature_2d,
    train_and_decode_2state_hmm, compute_sliding_window_adaptive_threshold,
)


def enumerate_hmm(log_emissions, transition, initial):
    """Enumerate all state paths, without using forward/backward or Viterbi."""
    paths = list(itertools.product(range(2), repeat=len(log_emissions)))
    logs = []
    for path in paths:
        score = np.log(initial[path[0]]) + log_emissions[0, path[0]]
        for t in range(1, len(path)):
            score += np.log(transition[path[t - 1], path[t]]) + log_emissions[t, path[t]]
        logs.append(score)
    log_total = float(np.logaddexp.reduce(logs))
    posterior = np.zeros_like(log_emissions)
    for path, log_score in zip(paths, logs):
        for t, state in enumerate(path):
            posterior[t, state] += np.exp(log_score - log_total)
    return np.array(paths[int(np.argmax(logs))]), posterior, log_total


class GmmTests(unittest.TestCase):
    def assert_crossing(self, parameters):
        threshold, status = _gaussian_intersection(*parameters)
        self.assertIn(status, ("between_means", "outside_means"))
        mu0, std0, mu1, std1, w0, w1 = parameters
        log0 = np.log(w0) + norm.logpdf(threshold, mu0, std0)
        log1 = np.log(w1) + norm.logpdf(threshold, mu1, std1)
        self.assertAlmostEqual(log0, log1, places=8)
        return threshold

    def test_unequal_variances_cross_weighted_densities(self):
        for parameters in [(0.2, .08, .7, .2, .5, .5),
                           (.12, .003, .16, .008, .7, .3),
                           (100.2, .08, 100.7, .2, .2, .8)]:
            with self.subTest(parameters=parameters):
                self.assert_crossing(parameters)

    def test_equal_variances_include_prior(self):
        threshold = self.assert_crossing((0., .2, 1., .2, .8, .2))
        self.assertAlmostEqual(threshold, .5 + .04 * np.log(4), places=12)

    def test_random_density_crossings(self):
        rng = np.random.default_rng(315)
        for _ in range(100):
            mu0, mu1 = sorted(rng.uniform(-2, 2, 2))
            std0, std1 = rng.uniform(.05, 1, 2)
            w0 = rng.uniform(.1, .9)
            parameters = (mu0, std0, mu1, std1, w0, 1-w0)
            threshold, status = _gaussian_intersection(*parameters)
            if status == "no_real_intersection":
                self.assertTrue(np.isfinite(threshold))
            else:
                self.assert_crossing(parameters)

    def test_no_crossing_is_an_explicit_fallback(self):
        self.assertEqual(_gaussian_intersection(.3, .2, .3, .2, .9, .1),
                         (.3, "no_real_intersection"))
        self.assertEqual(_gaussian_intersection(.3, .2, .3, .2),
                         (.3, "identical_components"))

    def test_em_returns_final_likelihood_and_prior_consistent_threshold(self):
        rng = np.random.default_rng(8)
        values = np.r_[rng.normal(.2, .04, 300), rng.normal(.7, .1, 700)]
        fit = compute_gmm_em_1d(values, prior=.3)
        self.assertTrue(fit['converged'])
        self.assertLess(fit['mu0'], fit['mu1'])
        self.assertEqual((fit['weight0'], fit['weight1']), (.3, .7))
        self.assert_crossing(tuple(fit[k] for k in ('mu0', 'std0', 'mu1', 'std1', 'weight0', 'weight1')))
        total = np.logaddexp(
            np.log(fit['weight0']) + norm.logpdf(values, fit['mu0'], fit['std0']),
            np.log(fit['weight1']) + norm.logpdf(values, fit['mu1'], fit['std1']))
        self.assertAlmostEqual(fit['log_likelihood'], float(total.sum()), places=9)

    def test_iteration_limit_does_not_claim_convergence(self):
        fit = compute_gmm_em_1d(np.linspace(0, 1, 50), max_iter=1)
        self.assertFalse(fit['converged'])
        self.assertEqual(fit['num_iter'], 1)

    def test_constant_sparse_and_nonfinite_input(self):
        fit = compute_gmm_em_1d(np.ones(20) * .4)
        self.assertEqual(fit['threshold_status'], 'identical_components')
        self.assertTrue(np.isfinite(fit['log_likelihood']))
        for values in [[], [np.nan, np.inf], [.1, .2, np.nan]]:
            with self.subTest(values=values):
                fit = compute_gmm_em_1d(values)
                self.assertEqual(fit['threshold_status'], 'insufficient_data')
                self.assertFalse(fit['converged'])
                self.assertTrue(np.isfinite(fit['threshold']))

    def test_rejects_invalid_fit_parameters(self):
        for arguments in [{'prior': 0}, {'prior': 1}, {'prior': np.nan},
                          {'max_iter': 0}, {'max_iter': 1.2}, {'tol': 0}]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compute_gmm_em_1d(np.arange(10), **arguments)


class MultifeatureTests(unittest.TestCase):
    def test_positive_unit_changes_preserve_labels(self):
        periodicity = np.r_[np.linspace(.12, .4, 30), np.linspace(.55, .8, 30)]
        zcr = np.r_[np.linspace(.1, .2, 30), np.linspace(.5, .9, 30)]
        for method in ('acf', 'amdf', 'cepstrum'):
            expected, model = cluster_multifeature_2d(periodicity, zcr, method)
            self.assertEqual(set(expected), {'v', 'uv'})
            for p_scale, z_scale in [(1000, .001), (.01, 10000), (1e-9, 1e-8)]:
                with self.subTest(method=method, scales=(p_scale, z_scale)):
                    actual, _ = cluster_multifeature_2d(periodicity*p_scale, zcr*z_scale, method)
                    np.testing.assert_array_equal(actual, expected)
            direction = -1 if method == 'amdf' else 1
            v, uv = model['centroid_v_normalized'], model['centroid_uv_normalized']
            self.assertGreaterEqual(direction*v[0]-v[1] + 1e-12, direction*uv[0]-uv[1])

    def test_translation_preserves_labels(self):
        p = np.r_[np.linspace(.1, .2, 20), np.linspace(.7, .8, 20)]
        z = 1-p
        labels, _ = cluster_multifeature_2d(p, z)
        shifted, _ = cluster_multifeature_2d(p+13, z+3)
        np.testing.assert_array_equal(shifted, labels)

    def test_collapsed_initial_centers_are_reseeded(self):
        p = np.r_[np.zeros(90), np.ones(10)]
        labels, model = cluster_multifeature_2d(p, np.zeros(100))
        np.testing.assert_array_equal(labels[:90], 'uv')
        np.testing.assert_array_equal(labels[90:], 'v')
        self.assertEqual(model['status'], 'ok')

    def test_equal_acoustic_scores_use_periodicity_tiebreak(self):
        p = np.r_[np.zeros(10), np.ones(10)]
        labels, _ = cluster_multifeature_2d(p, p)
        np.testing.assert_array_equal(labels[:10], 'uv')
        np.testing.assert_array_equal(labels[10:], 'v')

    def test_constant_single_and_empty_data(self):
        for n in (1, 10):
            labels, model = cluster_multifeature_2d(np.ones(n), np.ones(n))
            np.testing.assert_array_equal(labels, 'uv')
            self.assertEqual(model['status'], 'degenerate')
        labels, model = cluster_multifeature_2d([], [])
        self.assertEqual(len(labels), 0)
        self.assertEqual(model['status'], 'empty')

    def test_invalid_features_fail_clearly(self):
        for p, z in [([1], [1, 2]), ([np.nan], [0]), ([[1]], [[0]])]:
            with self.subTest(p=p, z=z), self.assertRaises(ValueError):
                cluster_multifeature_2d(p, z)


class HmmTests(unittest.TestCase):
    def setUp(self):
        self.transition = np.array([[.8, .2], [.1, .9]])
        self.initial = np.array([.7, .3])
        self.emissions = np.log(np.array([[.9, .1], [.4, .6], [.99, .01], [.5, .5], [.1, .9]]))
        self.segments = [(0, 2), (3, 5)]

    def test_forward_backward_matches_exhaustive_segment_paths(self):
        posterior, likelihood = _hmm_posteriors(self.emissions, self.transition, self.initial, self.segments)
        expected_ll = 0
        for start, end in self.segments:
            _, expected, ll = enumerate_hmm(self.emissions[start:end], self.transition, self.initial)
            np.testing.assert_allclose(posterior[start:end], expected, atol=1e-12)
            expected_ll += ll
        self.assertAlmostEqual(likelihood, expected_ll, places=12)
        np.testing.assert_array_equal(posterior[2], 0)

    def test_viterbi_matches_exhaustive_segment_paths(self):
        path = _hmm_viterbi(self.emissions, self.transition, self.initial, self.segments)
        for start, end in self.segments:
            expected, _, _ = enumerate_hmm(self.emissions[start:end], self.transition, self.initial)
            np.testing.assert_array_equal(path[start:end], expected)
        self.assertEqual(path[2], -1)

    def test_silence_resets_decoding_instead_of_connecting_active_frames(self):
        # Strong UV evidence at t=0 cannot drag an independent ambiguous segment
        # to UV when pi favors V at its fresh start.
        emissions = np.log(np.array([[.99, .01], [.5, .5], [.5, .5]]))
        transition = np.array([[.99, .01], [.01, .99]])
        pi = np.array([.1, .9])
        separate = _hmm_viterbi(emissions, transition, pi, [(0, 1), (2, 3)])
        compressed = _hmm_viterbi(emissions[[0, 2]], transition, pi, [(0, 2)])
        self.assertEqual(separate[2], 1)
        self.assertEqual(compressed[1], 0)

    def test_em_step_uses_independent_segment_posteriors(self):
        values = np.array([.1, .15, .99, .8, .85])
        active = np.array([True, True, False, True, True])
        active_values = values[active]
        mu = np.percentile(active_values, [25, 75])
        var = np.full(2, np.var(active_values)*.5 + 1e-4)
        transition = np.array([[.85, .15], [.15, .85]])
        pi = np.array([.5, .5])
        log_emissions = norm.logpdf(values[:, None], loc=mu, scale=np.sqrt(var))
        expected_gamma = np.zeros((5, 2))
        for start, end in self.segments:
            _, expected_gamma[start:end], _ = enumerate_hmm(log_emissions[start:end], transition, pi)
        expected_mu = np.sum(expected_gamma[active]*active_values[:, None], axis=0) / expected_gamma[active].sum(axis=0)
        labels, model = train_and_decode_2state_hmm(values, max_iter=1, active_mask=active)
        np.testing.assert_allclose(model['means'], expected_mu, atol=1e-12)
        np.testing.assert_array_equal(model['A'], transition)
        np.testing.assert_array_equal(model['pi'], pi)
        self.assertEqual(labels[2], 'sil')
        self.assertTrue(model['fixed_transitions'])

    def test_gap_length_does_not_change_emissions_or_labels(self):
        values = np.r_[np.linspace(.1, .2, 10), 0., np.linspace(.7, .8, 10)]
        mask = values != 0
        labels, model = train_and_decode_2state_hmm(values, active_mask=mask)
        longer = np.r_[values[:10], np.full(17, 999.), values[11:]]
        longer_mask = np.r_[np.ones(10, bool), np.zeros(17, bool), np.ones(10, bool)]
        longer_labels, longer_model = train_and_decode_2state_hmm(longer, active_mask=longer_mask)
        np.testing.assert_allclose(model['means'], longer_model['means'])
        np.testing.assert_array_equal(labels[mask], longer_labels[longer_mask])
        np.testing.assert_array_equal(longer_labels[~longer_mask], 'sil')

    def test_method_direction_and_short_sequences(self):
        values = np.r_[np.linspace(.1, .2, 10), np.linspace(.7, .8, 10)]
        acf_labels, _ = train_and_decode_2state_hmm(values)
        amdf_labels, _ = train_and_decode_2state_hmm(values, method='amdf')
        np.testing.assert_array_equal(acf_labels == 'v', amdf_labels == 'uv')
        for values in ([], [.4], [.4]*10):
            labels, _ = train_and_decode_2state_hmm(values)
            self.assertFalse(np.any(labels == 'v'))
        labels, model = train_and_decode_2state_hmm([.1, .9], active_mask=[False, False])
        np.testing.assert_array_equal(labels, 'sil')
        self.assertEqual(model['status'], 'no_active_frames')

    def test_no_mask_means_one_full_segment(self):
        values = np.linspace(0, 1, 20)
        labels, _ = train_and_decode_2state_hmm(values)
        explicit, _ = train_and_decode_2state_hmm(values, active_mask=np.ones(20, bool))
        np.testing.assert_array_equal(labels, explicit)

    def test_invalid_mask_or_observations(self):
        for values, mask in [([.1, .2], [True]), ([np.nan], [True]), ([[.1]], None)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                train_and_decode_2state_hmm(values, active_mask=mask)


class DynamicWindowTests(unittest.TestCase):
    def test_even_window_has_exactly_fifty_frames(self):
        values = np.arange(120, dtype=float)
        thresholds = compute_sliding_window_adaptive_threshold(values, np.ones(120), 0., 50, 1.)
        self.assertAlmostEqual(thresholds[60], np.median(values[35:85]))
        self.assertNotEqual(thresholds[60], np.median(values[35:86]))
        changed = values.copy()
        changed[85] = -10000
        actual = compute_sliding_window_adaptive_threshold(changed, np.ones(120), 0., 50, 1.)
        self.assertEqual(thresholds[60], actual[60])

    def test_odd_windows_and_clipped_boundaries(self):
        values = np.arange(10, dtype=float)
        for window in (1, 3, 4, 50):
            actual = compute_sliding_window_adaptive_threshold(values, np.ones(10), 0., window, 1.)
            left, right = window//2, window-window//2-1
            expected = [np.median(values[max(0, i-left):min(10, i+right+1)]) for i in range(10)]
            np.testing.assert_allclose(actual, expected)

    def test_alpha_zero_empty_and_energy_modulation(self):
        values = np.arange(10, dtype=float)
        np.testing.assert_array_equal(compute_sliding_window_adaptive_threshold(values, np.ones(10), .3, alpha=0), .3)
        self.assertEqual(len(compute_sliding_window_adaptive_threshold([], [], .3)), 0)
        actual = compute_sliding_window_adaptive_threshold([1.], [0.], .5, 1, 1.)
        self.assertAlmostEqual(actual[0], .8*.5 + .2*1.)

    def test_invalid_window_or_features(self):
        for kwargs in [{'window_frames': 0}, {'window_frames': 1.5}, {'alpha': 2}, {'global_threshold': np.nan}]:
            arguments = dict(values=[.1], ste=[1.], global_threshold=.5)
            arguments.update(kwargs)
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compute_sliding_window_adaptive_threshold(**arguments)
        with self.assertRaises(ValueError):
            compute_sliding_window_adaptive_threshold([1., 2.], [1.], .5)


if __name__ == '__main__':
    unittest.main()
