"""Legacy rounding boundaries, complete populations and comparison metadata."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from src.metrics.aggregation import (
    FIELDS, aggregate_rows, table_metrics, FILE_MACRO, ROUNDED_MACRO, ADVANCED_MACRO, COMPONENT_MACRO,
)
from src.metrics.bundle import evaluation_context
from src.metrics.comparison import MetricSeries, compare_series
from src.metrics.contracts import EvaluationContext, metric_contract
from src.metrics.formatting import format_metric, csv_value
from src.metrics.legacy import compute_error_metrics
from src.metrics.numeric import mean_metric, round_metric, rank_metric
from src.pipeline.contracts import FeatureSpec
from src.experiments.protocol_specs import default_spec
from src.experiments.dependencies import source_scopes

ROOT = Path(__file__).resolve().parents[1]


def context():
    return {'schema_version': 1, 'dataset_identity': 'a'*64, 'files': ['one.wav', 'two.wav'], 'split': 'test',
        'metric_version': 'legacy-v1', 'aggregation': FILE_MACRO,
        'feature_contract': FeatureSpec('acf', 30., 10., 70., 400., .008, 'baseline').to_dict(),
        'post_processors': [], 'smoothing': {'enabled': True, 'size': 3},
        'fit': {'mode': 'frozen', 'population': 'train_active', 'training_identity': 'b'*64},
        'decision': {'strategy': 'otsu', 'model_sha256': 'c'*64, 'parameters': {'reference_threshold': .45}}}


def rows():
    return [{'file': name, **compute_error_metrics(110., 12., 100., 10., count, 40)}
            for name, count in [('one.wav', 48), ('two.wav', 44)]]


def series(document=None, values=None):
    return MetricSeries(EvaluationContext(document or context()), rows() if values is None else values)


class LegacyMetricTests(unittest.TestCase):

    def test_score_uses_raw_percentages_before_rounding_components(self):
        value = compute_error_metrics(100.0149, 100.0049, 100., 100., 100, 100)
        self.assertEqual([value[key] for key in ('mape_f0', 'mape_std', 'mape_num')], [.01, 0., 0.])
        self.assertEqual(value['final_score'], .01)
        self.assertEqual(aggregate_rows([value], FILE_MACRO)['final_score'], .01)
        self.assertEqual(round_metric(aggregate_rows([value], COMPONENT_MACRO)['final_score']), 0.)

    def test_count_error_keeps_legacy_double_rounding(self):
        value = compute_error_metrics(100., 10., 100., 10., 101.495, 100.)
        self.assertEqual(value['abs_error_num'], 2)
        self.assertEqual(value['pred_f0_num'], 101)
        self.assertEqual(value['mape_num'], 1.5)

    def test_zero_missing_and_nonfinite_components_never_become_two_component_scores(self):
        for missing in (None, 0., np.nan, np.inf):
            value = compute_error_metrics(100., 10., 100., missing, 20, 20)
            self.assertIsNone(value['mape_std'])
            self.assertIsNone(value['final_score'])
        self.assertIsNone(mean_metric([1., None, 3.]))
        self.assertIsNone(mean_metric([]))
        self.assertIsNone(round_metric(np.inf))
        self.assertEqual(rank_metric(None), float('inf'))


    def test_missing_file_propagates_null_to_the_full_population(self):
        valid = rows()[0]
        invalid = {'file': 'missing.wav', **compute_error_metrics(110., 12., None, 10., 48, 40)}
        for profile in (FILE_MACRO, ROUNDED_MACRO, ADVANCED_MACRO, COMPONENT_MACRO):
            value = aggregate_rows([valid, invalid], profile)
            self.assertIsNone(value['mape_f0'])
            self.assertIsNone(value['final_score'])
            self.assertIsNotNone(value['mape_num'])
        document = context(); document['files'] = ['one.wav', 'missing.wav']
        compared = compare_series(series(document, [valid, invalid]), series(document, [valid, invalid]))
        self.assertEqual(compared['aggregate']['final_score']['status'], 'missing_both')
        self.assertIsNone(compared['aggregate']['final_score']['delta'])

    def test_presentation_and_compatibility_exports_keep_values(self):
        from src.analysis.evaluation import compute_error_metrics as old_compute, format_metric as old_format
        from src.metrics.evaluation import evaluate_result
        self.assertIs(old_compute, compute_error_metrics)
        self.assertIs(old_format, format_metric)
        self.assertEqual(format_metric(None, '<8.2f'), 'N/A     ')
        self.assertEqual(csv_value(None), '')
        self.assertEqual(csv_value(1.23456789), '1.23456789')
        self.assertEqual(metric_contract()['metric_version'], 'legacy-v1')

    def test_numeric_imports_do_not_load_analysis_formatting_or_runners(self):
        code = ('import sys; from src.metrics.legacy import compute_error_metrics; '
                'from src.metrics.aggregation import aggregate_rows; '
                'assert not any(n == "src.analysis" or n.startswith("src.experiments") '
                'or n.startswith("matplotlib") or n == "src.metrics.formatting" for n in sys.modules)')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_formatting_is_outside_scientific_source_scope(self):
        scopes = source_scopes(ROOT)
        self.assertIn('src/metrics/legacy.py', scopes['train'])
        self.assertNotIn('src/metrics/formatting.py', scopes['train'])
        self.assertIn('src/metrics/formatting.py', scopes['report'])


class ComparisonTests(unittest.TestCase):
    def test_context_and_series_are_immutable_deep_copies(self):
        document, values = context(), rows()
        actual = MetricSeries(EvaluationContext(document), values)
        document['smoothing']['size'] = 9
        values[0]['final_score'] = 999
        saved = actual.to_dict()
        self.assertEqual(saved['context']['smoothing']['size'], 3)
        self.assertNotEqual(saved['rows'][0]['final_score'], 999)
        saved['rows'][0]['file'] = 'changed'
        self.assertEqual(actual.to_dict()['rows'][0]['file'], 'one.wav')

    def test_context_rejects_missing_metadata_unknown_versions_and_invalid_fit(self):
        for key in context():
            document = context()
            del document[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                EvaluationContext(document)
        for key, value in [('metric_version', 'raw-v2'), ('schema_version', True), ('aggregation', 'unknown')]:
            document = context(); document[key] = value
            with self.assertRaises(ValueError): EvaluationContext(document)
        for fit in ({'mode': 'frozen', 'population': 'input_active', 'training_identity': 'b'*64},
                    {'mode': 'input_adaptive', 'population': 'train_active', 'training_identity': None}):
            document = context(); document['fit'] = fit
            with self.assertRaises(ValueError): EvaluationContext(document)
        document = context(); document['decision']['model_sha256'] = None
        with self.assertRaises(ValueError): EvaluationContext(document)

    def test_series_rejects_duplicate_incomplete_and_invalid_rows(self):
        invalid = [[], [rows()[0], rows()[0]]]
        for field in FIELDS:
            values = rows(); del values[0][field]; invalid.append(values)
        for value in (np.nan, float('inf'), True, -1.):
            values = rows(); values[0]['mape_f0'] = value; invalid.append(values)
        for values in invalid:
            with self.assertRaises(ValueError): series(values=values)

    def test_comparison_rejects_frame_range_gate_and_zcr_mismatch(self):
        from dataclasses import replace
        base = FeatureSpec.from_dict(context()['feature_contract'])
        for change in ({'frame_duration_ms': 25.}, {'f0_min': 100.}, {'ste_silence_ratio': .01}, {'profile': 'centered_all'}):
            document = context(); document['feature_contract'] = replace(base, **change).to_dict()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'feature_contract'):
                compare_series(series(), series(document))

    def test_comparison_rejects_fit_population_mode_or_training_set_mismatch(self):
        for fit in ({'mode': 'frozen', 'population': 'train_all', 'training_identity': 'b'*64},
                    {'mode': 'frozen', 'population': 'train_active', 'training_identity': 'd'*64},
                    {'mode': 'input_adaptive', 'population': 'input_active', 'training_identity': None}):
            document = context(); document['fit'] = fit
            with self.assertRaisesRegex(ValueError, 'fit'):
                compare_series(series(), series(document))

    def test_study_differences_are_explicit_and_recorded(self):
        document = context()
        document['smoothing']['size'] = 5
        document['post_processors'] = [{'name': 'Example', 'parameters': {}}]
        with self.assertRaises(ValueError): compare_series(series(), series(document))
        result = compare_series(series(), series(document), allow_differences=['smoothing', 'post_processors'])
        self.assertEqual(set(result['declared_differences']), {'smoothing', 'post_processors'})
        self.assertEqual(result['declared_differences']['smoothing']['right']['size'], 5)
        with self.assertRaises(ValueError): compare_series(series(), series(), allow_differences=['anything'])

    def test_dataset_split_and_aggregation_can_never_be_waived(self):
        for key, value in [('dataset_identity', 'f'*64), ('split', 'train'), ('aggregation', COMPONENT_MACRO)]:
            document = context(); document[key] = value
            with self.assertRaisesRegex(ValueError, key): compare_series(series(), series(document))
            with self.assertRaises(ValueError): compare_series(series(), series(document), allow_differences=[key])

    def test_missing_files_are_rejected_instead_of_inner_joined(self):
        with self.assertRaisesRegex(ValueError, 'population'):
            compare_series(series(), series(values=rows()[:1]))
        swapped = list(reversed(rows()))
        value = compare_series(series(), series(values=swapped))
        self.assertEqual([row['file'] for row in value['files']], ['one.wav', 'two.wav'])
        self.assertTrue(all(metric['delta'] == 0 for row in value['files'] for metric in row['metrics'].values()))

    def test_delta_direction_and_missing_values_are_unambiguous(self):
        left, right = rows(), rows()
        right[0]['final_score'] -= 1.
        left[0]['mape_std'] = None
        right[1]['mape_num'] = None
        value = compare_series(series(values=left), series(values=right))
        self.assertEqual(value['files'][0]['metrics']['final_score']['delta'], -1.)
        self.assertEqual(value['files'][0]['metrics']['mape_std']['status'], 'missing_left')
        self.assertEqual(value['files'][1]['metrics']['mape_num']['status'], 'missing_right')
        self.assertIsNone(value['aggregate']['mape_num']['delta'])



if __name__ == '__main__':
    unittest.main()
