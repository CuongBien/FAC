"""Regression checks for undefined metrics, labeled fitting and shared config."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile
from scipy.stats import norm

from src import config
from src.analysis.evaluation import (
    compute_error_metrics, evaluate_against_ground_truth,
    mean_metric, format_metric, round_metric,
)
from src.analysis.threshold import extract_training_distributions, find_gaussian_threshold
from src.core.pitch_detector import PitchDetector
from src.pipelines import build_baseline, build_champion
from src.visualization.plotter import plot_distributions_and_threshold
import predict


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audio_pair(folder, name='zero', stats='F0mean 0\nF0std 0\nF0num 0\n', timed=''):
    path = Path(folder) / (name + '.wav')
    time = np.arange(6400) / 16000
    samples = np.zeros(len(time)) if name == 'zero' else .4*np.sin(2*np.pi*180*time)
    wavfile.write(path, 16000, (samples*32767).astype(np.int16))
    path.with_suffix('.lab').write_text(stats + timed, encoding='utf-8')
    return path


class ErrorMetricTests(unittest.TestCase):
    def test_nonzero_references_retain_three_component_formula(self):
        result = compute_error_metrics(110., 12., 100., 10., 48, 40)
        self.assertEqual((result['mape_f0'], result['mape_std'], result['mape_num']), (10., 20., 20.))
        self.assertEqual(result['final_score'], round(50/3, 2))

    def test_each_zero_reference_makes_only_its_mape_undefined(self):
        for index, key, absolute in [(0, 'mape_f0', 'abs_error_mean'),
                                     (1, 'mape_std', 'abs_error_std'),
                                     (2, 'mape_num', 'abs_error_num')]:
            reference = [100., 10., 20.]
            reference[index] = 0.
            result = compute_error_metrics(110., 12., reference[0], reference[1], 25, reference[2])
            with self.subTest(key=key):
                self.assertIsNone(result[key])
                self.assertGreater(result[absolute], 0)
                self.assertIsNone(result['final_score'])
                self.assertIsNone(result['composite_score'])

    def test_perfect_zero_predictions_still_have_undefined_mape(self):
        result = compute_error_metrics(0., 0., 0., 0., 0, 0)
        self.assertEqual(result['abs_error_mean'], 0.)
        self.assertEqual(result['abs_error_std'], 0.)
        self.assertEqual(result['abs_error_num'], 0)
        for key in ('mape_f0', 'mape_mean', 'rel_error_mean_pct', 'mape_f0_std', 'mape_num', 'final_score'):
            self.assertIsNone(result[key])
        self.assertNotIn('NaN', json.dumps(result, allow_nan=False))

    def test_missing_count_never_creates_a_two_component_score(self):
        result = compute_error_metrics(110., 12., 100., 10.)
        self.assertEqual(result['mape_f0'], 10.)
        self.assertEqual(result['mape_std'], 20.)
        self.assertIsNone(result['mape_num'])
        self.assertIsNone(result['final_score'])

    def test_missing_statistics_are_preserved(self):
        result = compute_error_metrics(100., 10., None, None, 20, 20)
        self.assertIsNone(result['abs_error_mean'])
        self.assertEqual(result['mape_num'], 0.)
        self.assertIsNone(result['final_score'])

    def test_aggregate_does_not_drop_unmeasurable_files(self):
        self.assertIsNone(mean_metric([10., None, 20.]))
        self.assertIsNone(mean_metric([]))
        self.assertEqual(mean_metric([10., 20.]), 15.)
        self.assertEqual(format_metric(None), 'N/A')
        self.assertEqual(format_metric(None, '<8.2f'), 'N/A     ')
        self.assertIsNone(round_metric(np.nan))

    def test_summary_only_lab_has_no_classification_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = audio_pair(folder)
            result = evaluate_against_ground_truth(PitchDetector().process_file(str(path)), str(path.with_suffix('.lab')))
        self.assertIsNone(result['classification_accuracy'])
        self.assertIsNone(result['voiced_f1'])
        self.assertIsNone(result['final_score'])

    def test_timed_labels_still_enable_classification_without_f0num(self):
        with tempfile.TemporaryDirectory() as folder:
            path = audio_pair(folder, stats='F0mean 180\nF0std 10\n', timed='0 .2 v\n.2 .4 uv\n')
            result = evaluate_against_ground_truth({
                'frame_times': np.array([.1, .3]), 'labels': np.array(['v', 'uv']),
                'f0_mean': 180., 'f0_std': 10., 'num_voiced': 1,
            }, str(path.with_suffix('.lab')))
        self.assertEqual(result['classification_accuracy'], 100.)
        self.assertEqual(result['voiced_f1'], 100.)
        self.assertEqual(result['ref_f0_num'], 1)
        self.assertEqual(result['final_score'], 0.)


class GaussianTrainingTests(unittest.TestCase):
    def test_summary_labels_cannot_generate_fake_gaussian_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            audio_pair(folder)
            with self.assertRaisesRegex(ValueError, r'V=0, UV=0.*timed'):
                extract_training_distributions(folder)

    def test_one_missing_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            audio_pair(folder, name='tone', timed='0 .4 v\n')
            with self.assertRaisesRegex(ValueError, r'UV=0'):
                extract_training_distributions(folder)

    def test_labeled_classes_produce_real_distributions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = audio_pair(folder, name='tone', timed='0 .2 v\n.2 .4 uv\n')
            # V and UV must come from labels, not from the current detector's decisions.
            stats = extract_training_distributions(folder)
        self.assertGreaterEqual(stats['num_voiced_frames'], 2)
        self.assertGreaterEqual(stats['num_unvoiced_frames'], 2)
        self.assertEqual(len(stats['values_v']), stats['num_voiced_frames'])
        self.assertEqual(stats['files_processed'], [path.name])

    def test_invalid_or_degenerate_gaussian_parameters_are_rejected(self):
        for parameters in [(0., 0., 0., 0.), (.7, 0., .2, .1),
                           (.7, .1, .2, np.nan), (.4, .1, .4, .1)]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                find_gaussian_threshold(*parameters)

    def test_valid_threshold_is_a_density_crossing(self):
        threshold = find_gaussian_threshold(.7, .08, .2, .13)
        self.assertAlmostEqual(norm.logpdf(threshold, .7, .08), norm.logpdf(threshold, .2, .13), places=10)

    def test_plot_rejects_missing_samples_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'not_created.png'
            with self.assertRaisesRegex(ValueError, 'labeled'):
                plot_distributions_and_threshold(np.array([]), np.array([]), 0., 0., 0., 0., 0., save_path=str(output))
            self.assertFalse(output.exists())

    def test_training_cli_exits_cleanly_without_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            audio_pair(folder)
            output = Path(folder) / 'outputs'
            run = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/02_train_threshold.py'),
                                  '--train_dir', folder, '--out_fig', str(output/'fit.png'),
                                  '--out_json', str(output/'fit.json')], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn('V=0, UV=0', run.stderr)
            self.assertNotIn('Traceback', run.stderr)
            self.assertFalse(output.exists())


class SharedConfigurationTests(unittest.TestCase):
    def test_api_defaults_match_saved_baseline_contract(self):
        for method, threshold in [('acf', .4408), ('amdf', .4380), ('cepstrum', .06)]:
            self.assertEqual(PitchDetector(method).threshold, threshold)
        self.assertEqual(PitchDetector('amdf', threshold=.123).threshold, .123)

    def test_api_cli_and_benchmark_produce_identical_contours(self):
        benchmark = script('14_compare_acf_amdf_cepstrum')
        factories, _ = benchmark.build_bt2_detectors()
        cli = {**predict.get_baseline_detectors(), **predict.get_champion_detectors()}
        time = np.arange(6400)/16000
        signal = .4*np.sin(2*np.pi*180*time) + .1*np.sin(2*np.pi*360*time)
        signal[:800] = signal[-800:] = 0
        for index, method in enumerate(('acf', 'amdf', 'cepstrum')):
            label = 'Cepstrum' if method == 'cepstrum' else method.upper()
            api = [build_baseline(method), build_champion(method)]
            for offset, kind in enumerate(('Baseline', 'Champion')):
                key = list(factories)[index*2+offset]
                expected = api[offset].process_signal(signal, 16000)
                for candidate in (cli[f'{label} {kind}'], factories[key]()):
                    actual = candidate.process_signal(signal, 16000)
                    np.testing.assert_array_equal(actual['f0_contour'], expected['f0_contour'])
                    np.testing.assert_array_equal(actual['labels'], expected['labels'])

    def test_factories_never_share_plugin_state(self):
        first, second = build_champion('cepstrum'), build_champion('cepstrum')
        self.assertIsNot(first.base_detector, second.base_detector)
        for a, b in zip(first.plugins, second.plugins):
            self.assertIsNot(a, b)

    def test_plugin_sweep_uses_the_same_thresholds(self):
        sweep = script('05_compare_plugins')
        for method in ('acf', 'amdf', 'cepstrum'):
            for combo in ((), ('BP',), ('Clip',), ('BP', 'Clip'), config.CHAMPION_PLUGINS[method]):
                self.assertEqual(sweep.get_threshold(method, combo), config.threshold_for_plugins(method, combo))


class ScriptIntegrationTests(unittest.TestCase):
    def test_script03_runs_with_unavailable_accuracy_and_f1(self):
        comparison = script('03_compare_params')
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            report = Path(folder)/'comparison.json'
            figure = Path(folder)/'comparison.png'
            comparison.run_comparison(output_fig=str(figure), output_json=str(report))
            data = json.loads(report.read_text(encoding='utf-8'))
            self.assertTrue(figure.exists())
        self.assertIn('N/A', stdout.getvalue())
        for key in ('frame_20ms', 'frame_25ms_standard', 'frame_30ms'):
            self.assertIsNone(data[key]['accuracy'])
            self.assertIsNone(data[key]['f1_score'])

    def test_benchmark15_csv_matches_json_ids_and_full_precision(self):
        benchmark = script('15_evaluate_unsupervised_thresholds')
        values = {key: None for key in ('diff_f0', 'mape_f0', 'diff_std', 'mape_std', 'diff_num', 'mape_num', 'final_score')}
        data = {'acf': {'thresholds': {key: .123456789+i*.01 for i, key in enumerate(benchmark.STRATEGY_LABELS)},
                        'pipeline_eval': {key: dict(values) for key in reversed(benchmark.STRATEGY_LABELS)}}}
        with tempfile.TemporaryDirectory() as folder:
            json_path, csv_path = benchmark.write_benchmark_results(data, folder)
            saved = json.loads(Path(json_path).read_text(encoding='utf-8'))
            with open(csv_path, encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(float(row['Threshold_T']), saved['acf']['thresholds'][row['Strategy']])
            self.assertEqual(row['Strategy_Label'], benchmark.STRATEGY_LABELS[row['Strategy']])
            self.assertEqual(row['Final_Score_Pct'], '')

    def test_benchmark15_and16_propagate_undefined_scores(self):
        histogram = script('15_evaluate_unsupervised_thresholds')
        advanced = script('16_evaluate_advanced_unsupervised')
        with tempfile.TemporaryDirectory() as folder:
            path = audio_pair(folder)
            result = histogram.evaluate_pipeline_across_files([str(path)], 'acf', .4408)
        self.assertIsNone(result['mape_f0'])
        self.assertIsNone(result['final_score'])
        self.assertEqual(result['diff_f0'], 0.)
        valid = compute_error_metrics(100., 10., 100., 10., 20, 20)
        invalid = compute_error_metrics(100., 10., 0., 10., 20, 20)
        mixed = advanced.aggregate_pipeline_metrics([valid, invalid])
        self.assertIsNone(mixed['mape_f0'])
        self.assertIsNone(mixed['final_score'])
        self.assertEqual(mixed['diff_f0'], 50.)

    def test_cli_zero_and_valid_files_export_blank_undefined_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            audio_pair(folder)
            audio_pair(folder, name='tone', stats='F0mean 180\nF0std 20\nF0num 30\n')
            output = Path(folder)/'out'
            run = subprocess.run([sys.executable, '-B', str(ROOT/'predict.py'), '-i', folder,
                                  '-m', 'baseline', '-o', str(output)], cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(run.returncode, 0, run.stderr)
            with (output/'evaluation_summary_metrics.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            zero_rows = [row for row in rows if row['File'] == 'zero.wav']
            for row in zero_rows:
                for key in ('MAPE_F0mean_pct', 'MAPE_F0std_pct', 'MAPE_F0num_pct', 'Composite_Score_pct'):
                    self.assertEqual(row[key], '')
                self.assertEqual(row['Abs_Err_F0mean'], '0.0')
            self.assertIn('N/A', run.stdout)
            self.assertNotIn('Độ chính xác tương đối', run.stdout)
            self.assertNotIn('TB 2 MAPE', run.stdout)

    def test_script04_handles_zero_reference_without_false_score(self):
        testing = script('04_run_testing')
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            audio_pair(folder)
            report = Path(folder)/'test.json'
            summary = Path(folder)/'test.csv'
            testing.run_testing(test_dir=folder, threshold_file=str(Path(folder)/'absent.json'),
                                output_json=str(report), output_csv=str(summary), output_dir_fig=str(Path(folder)/'figures'))
            data = json.loads(report.read_text(encoding='utf-8'))
            with summary.open(encoding='utf-8', newline='') as stream:
                row = next(csv.DictReader(stream))
        self.assertIsNone(data['average_metrics']['final_composite_score_pct'])
        self.assertIsNone(data['average_metrics']['mean_v_uv_accuracy'])
        self.assertEqual(row['Final_Score_Pct'], '')

    def test_script05_never_ranks_undefined_scores_as_champions(self):
        sweep = script('05_compare_plugins')
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            audio_pair(folder)
            report = Path(folder)/'sweep.json'
            sweep.run_evaluation(test_dir=folder, output_json=str(report),
                                 output_csv=str(Path(folder)/'sweep.csv'), output_ranking_fig=str(Path(folder)/'ranking.png'))
            data = json.loads(report.read_text(encoding='utf-8'))
            raw = json.loads(report.with_name('sweep_raw.json').read_text(encoding='utf-8'))
            self.assertEqual(len(raw['all_configurations']), 16)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['config_name'], 'Baseline')
        self.assertIsNone(data[0]['final_composite_score_pct'])

    def test_benchmark14_propagates_missing_components(self):
        benchmark = script('14_compare_acf_amdf_cepstrum')
        with tempfile.TemporaryDirectory() as folder:
            audio_pair(folder)
            result = benchmark.evaluate_suite({'ACF Baseline': lambda: build_baseline('acf')}, folder)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.plot_bt2_comparison([result], str(Path(folder)/'comparison.png'))
            self.assertTrue((Path(folder)/'comparison.png').exists())
        self.assertEqual(result['average_error_hz'], 0.)
        self.assertIsNone(result['mape_f0_pct'])
        self.assertIsNone(result['mape_std_pct'])
        self.assertIsNone(result['mape_num_pct'])
        self.assertIsNone(result['final_composite_score_pct'])


if __name__ == '__main__':
    unittest.main()
