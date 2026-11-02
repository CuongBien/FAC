"""Capture a reviewed metric baseline, then verify without changing old runs."""
import argparse
import ast
from copy import deepcopy
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.evaluation import compute_error_metrics, mean_metric
from src.analysis.validation import aggregate
from src.experiments.serialization import json_bytes, json_digest
from src.experiments.manifest import sha256, source_manifest
from src.experiments.registry import RunRegistry
from src.experiments.runner import dataset_record, runtime_record
from src.experiments.protocol_specs import default_spec
from src.experiments.metric_artifacts import metric_artifact
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.metrics.aggregation import aggregate_rows, table_metrics, ROUNDED_MACRO, ADVANCED_MACRO
from src.metrics.comparison import MetricSeries, compare_series
from src.metrics.contracts import EvaluationContext
from src.metrics.evaluation import evaluate_result


def script(filename):
    spec = importlib.util.spec_from_file_location('metric_reference_'+filename[:2], ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture():
    benchmark14 = script('14_compare_acf_amdf_cepstrum.py')
    benchmark16 = script('16_evaluate_advanced_unsupervised.py')
    rng = np.random.default_rng(17)
    arguments = [[110., 12., 100., 10., 48, 40], [0., 0., 0., 0., 0, 0],
                 [100., 10., None, None, 20, 20], [110., 12., 100., 10., None, None],
                 [100.0149, 100.0049, 100., 100., 100, 100],
                 [100., 10., 100., 10., 101.495, 100.]]
    for index in range(3):
        values = [110., 12., 100., 10., 48, 40]
        values[(2, 3, 5)[index]] = 0.
        arguments.append(values)
    for _ in range(80):
        arguments.append([float(rng.uniform(0, 300)), float(rng.uniform(0, 100)),
                          float(rng.uniform(1, 300)), float(rng.uniform(1, 100)),
                          int(rng.integers(0, 500)), int(rng.integers(1, 500))])
    vectors = [{'arguments': args, 'expected': compute_error_metrics(*args)} for args in arguments]
    evaluation = json.loads((ROOT/'outputs/champion_unsupervised_2026-11-01/evaluation.json').read_bytes())
    groups = []
    for item in evaluation['pipelines']+evaluation['protocols']:
        rows = item['files']
        rounded = {key: mean_metric([row[key] for row in rows], 2) for key in aggregate(rows)}
        groups.append({'id': [item[key] for key in ('dataset', 'method', 'strategy')],
                       'rows': rows, 'file_macro': aggregate(rows), 'rounded_macro': rounded,
                       'advanced_table': benchmark16.aggregate_pipeline_metrics(rows)})
    factories, _ = benchmark14.build_bt2_detectors()
    controls = {folder: benchmark14.evaluate_suite(factories, folder)
                for folder in ('TinHieuHuanLuyen', 'TinHieuKiemThu')}
    return {'schema_version': 1, 'base_commit': 'c5324e2', 'vectors': vectors,
            'aggregation_groups': groups, 'baseline_champion_suite': controls}


def verify(reference_path, output=None):
    if output:
        if any((output/name).exists() for name in ('validation.json', 'source_snapshot.tar.gz', 'metrics.json')):
            raise FileExistsError('Proof exists; choose a fresh directory')
        if (output/'reference.json').exists() and sha256(output/'reference.json') != sha256(reference_path):
            raise ValueError('Output reference differs')
        output.mkdir(parents=True, exist_ok=True)
    provenance = source_manifest(ROOT)
    registry = RunRegistry(ROOT).verify()
    assert all(item['status'] == 'verified' for item in registry)
    reference = json.loads(reference_path.read_bytes())
    for vector in reference['vectors']:
        assert compute_error_metrics(*vector['arguments']) == vector['expected']
    for group in reference['aggregation_groups']:
        assert aggregate_rows(group['rows']) == group['file_macro']
        assert aggregate_rows(group['rows'], ROUNDED_MACRO) == group['rounded_macro']
        assert table_metrics(group['rows'], ADVANCED_MACRO) == group['advanced_table']
    benchmark14 = script('14_compare_acf_amdf_cepstrum.py')
    factories, _ = benchmark14.build_bt2_detectors()
    for folder, expected in reference['baseline_champion_suite'].items():
        assert benchmark14.evaluate_suite(factories, folder) == expected
    old = ROOT/'outputs/champion_unsupervised_2026-11-01'
    evaluation = json.loads((old/'evaluation.json').read_bytes())
    recomputed = deepcopy(evaluation)
    cases = frames = 0
    for item in recomputed['pipelines']+recomputed['protocols']:
        folder = 'TinHieuHuanLuyen' if item['dataset'] == 'train' else 'TinHieuKiemThu'
        for row in item['files']:
            with (old/row['contour_path']).open(newline='', encoding='utf-8') as stream:
                saved = list(csv.DictReader(stream))
            labels = np.array([record['final_label'] for record in saved], dtype=object)
            contour = np.array([float(record['f0_hz']) for record in saved])
            voiced = contour[labels == 'v']
            prediction = {'frame_times': np.array([float(record['time_sec']) for record in saved]),
                'labels': labels, 'f0_contour': contour, 'f0_mean': float(np.mean(voiced)) if len(voiced) else 0.,
                'f0_std': float(np.std(voiced)) if len(voiced) else 0., 'num_voiced': int(len(voiced))}
            actual = evaluate_result(prediction, ROOT/folder/row['file'])
            assert all(actual[key] == value for key, value in row.items() if key in actual), (item['strategy'], row['file'])
            row.update(actual)
            cases += 1
            frames += len(saved)
        assert aggregate_rows(item['files']) == item['metrics']
    datasets = {'train': dataset_record(ROOT/'TinHieuHuanLuyen'), 'test': dataset_record(ROOT/'TinHieuKiemThu')}
    # This is an explicit annotation of the fixed legacy benchmark, never an
    # in-place upgrade. Cross-check the physical settings against its freeze and
    # the pre-refactor 48-control reference before constructing contexts.
    spec = default_spec('champion').to_dict()
    frozen = json.loads((old/'frozen_config.json').read_bytes())['experiment']
    for method, pipeline in spec['pipelines'].items():
        assert pipeline['pre_processors'] == frozen['native_plugins'][method]['pre']
        assert pipeline['post_processors'] == frozen['native_plugins'][method]['post']
        for key in ('frame_duration_ms', 'hop_duration_ms'):
            assert pipeline[key] == frozen['champions'][method][key]
        assert pipeline['reference_threshold'] == frozen['champions'][method]['threshold']
        for key in ('f0_min', 'f0_max', 'ste_silence_ratio'):
            assert pipeline[key] == frozen[key]
    for records in reference['baseline_champion_suite'].values():
        for record in records:
            method = 'cepstrum' if 'Cepstrum' in record['name'] else ('amdf' if 'AMDF' in record['name'] else 'acf')
            pipeline = spec['controls']['baseline'][method] if 'Baseline' in record['name'] else spec['pipelines'][method]
            for key in ('frame_duration_ms', 'hop_duration_ms'):
                assert pipeline[key] == record['configuration'][key]
            assert pipeline['reference_threshold'] == record['configuration']['threshold']
    with tempfile.TemporaryDirectory(prefix='fac-metric-proof-') as directory:
        work = Path(directory)
        for name in ('frozen_config.json', 'frozen_models.json', 'evaluation.json'):
            shutil.copyfile(old/name, work/name)
        shutil.copytree(old/'input_models', work/'input_models')
        (work/'config.json').write_bytes(json_bytes(spec))
        before = metric_artifact(work, {'datasets': {'train': datasets['train']}}, {'test_dataset': datasets['test']})
        (work/'evaluation.json').write_bytes(json_bytes(recomputed))
        after = metric_artifact(work, {'datasets': {'train': datasets['train']}}, {'test_dataset': datasets['test']})
        comparisons = []
        for left, right in zip(before['series'], after['series']):
            result = compare_series(MetricSeries(EvaluationContext(left['context']), left['rows']),
                                    MetricSeries(EvaluationContext(right['context']), right['rows']))
            assert all(value['delta'] == 0 for row in result['files'] for value in row['metrics'].values())
            comparisons.append(result)
        # Bind the persisted annotation to the immutable original evaluation;
        # all recomputed metric rows were proved exactly equal above.
        if output:
            (output/'metrics.json').write_bytes(json_bytes(before))
    by_id = {(item['dataset'], item['method'], item['strategy']): item for item in before['series']}
    baseline, champion = by_id['test', 'acf', 'baseline'], by_id['test', 'acf', 'champion']
    refused = []
    for left, right in ((baseline, champion), (by_id['test', 'acf', 'gmm'], by_id['test', 'acf', 'hmm_input_active'])):
        try:
            compare_series(MetricSeries(EvaluationContext(left['context']), left['rows']),
                           MetricSeries(EvaluationContext(right['context']), right['rows']))
        except ValueError as exc:
            refused.append(str(exc))
    assert len(refused) == 2
    with tarfile.open(ROOT/'outputs/refactor_phase3/source_snapshot.tar.gz') as archive:
        previous = ast.parse(archive.extractfile('src/analysis/evaluation.py').read().decode('utf-8'))
    old_functions = {node.name: ast.dump(node) for node in previous.body if isinstance(node, ast.FunctionDef)}
    for filename, names in [('numeric.py', ('round_metric', 'mean_metric', 'rank_metric')),
                            ('formatting.py', ('format_metric', 'csv_value')),
                            ('legacy.py', ('compute_error_metrics',)), ('evaluation.py', ('evaluate_against_ground_truth',))]:
        functions = {node.name: ast.dump(node) for node in ast.parse((ROOT/'src/metrics'/filename).read_text(encoding='utf-8')).body
                     if isinstance(node, ast.FunctionDef)}
        assert all(functions[name] == old_functions[name] for name in names)
    assert source_manifest(ROOT)['source_sha256'] == provenance['source_sha256']
    assert RunRegistry(ROOT).verify() == registry
    files = [ROOT/'predict.py', *sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'scripts').glob('*.py')),
             *sorted((ROOT/'tests').glob('test_*.py'))]
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), feature_version=(3, 9))
    result = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime_record(ROOT),
        'metric_version': 'legacy-v1', 'golden_vectors_matched': len(reference['vectors']),
        'aggregation_groups_matched': len(reference['aggregation_groups']), 'baseline_champion_cases_matched': 48,
        'file_metric_cases_matched': cases, 'contour_rows_used': frames, 'typed_series_compared': len(comparisons),
        'mismatched_protocols_rejected': refused, 'moved_metric_functions_ast_unchanged': True,
        'scope': '48 control detector replays plus metrics from 288 frozen contours; no model fit or legacy artifact rewrite.',
        'reference_sha256': sha256(reference_path),
        'source_provenance': provenance, 'python_39_ast_files': len(files),
        'test_source_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((ROOT/'tests').glob('test_*.py'))},
        'legacy_runs_verified': registry, 'original_evaluation_sha256': sha256(old/'evaluation.json'),
        'annotation_spec_sha256': json_digest(spec)}
    if output:
        target = output/'reference.json'
        if target.exists():
            assert sha256(target) == sha256(reference_path), 'Output reference differs'
        else:
            target.write_bytes(reference_path.read_bytes())
        snapshot = output/'source_snapshot.tar.gz'
        write_source_snapshot(ROOT, snapshot, provenance)
        result.update({'metrics_sha256': sha256(output/'metrics.json'), 'source_snapshot_sha256': sha256(snapshot),
                       'source_snapshot_files': verify_source_snapshot(snapshot, provenance)})
        with (output/'validation.json').open('xb') as stream:
            stream.write(json_bytes(result))
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_provenance', 'test_source_sha256', 'legacy_runs_verified')}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture', 'verify'))
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/refactor_phase4/reference.json')
    parser.add_argument('--output', type=Path, help='Write a new proof bundle; omit for read-only verification')
    args = parser.parse_args()
    if args.phase == 'capture':
        args.output = args.output or ROOT/'outputs/refactor_phase4'
        args.output.mkdir(parents=True, exist_ok=True)
        result = capture()
        with (args.output/'reference.json').open('xb') as stream:
            stream.write(json_bytes(result))
        print('Captured', len(result['vectors']), 'vectors,', len(result['aggregation_groups']), 'groups, 48 controls')
    else:
        verify(args.reference, args.output)


if __name__ == '__main__':
    main()
