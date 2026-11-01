"""Verify the real native lifecycle and compare with immutable historical runs.

Refits train-only Champion decisions once; never searches presets or selects on
test. Stage 4 uses its recorded models without repeating the calibration sweep.
The generated native run is archived instead of duplicating hundreds of CSVs.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.protocols import predict_strategies, result_from_labels
from src.analysis.validation import evaluate_result
from src.experiments.legacy import read_frozen
from src.experiments.manifest import sha256, source_manifest
from src.experiments.protocol_specs import default_spec
from src.experiments.registry import RunRegistry
from src.experiments.runner import RunService, runtime_record
from src.experiments.serialization import json_bytes
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.experiments.store import verify_run
from src.pipeline.cache import PitchCache
from src.pipeline.features import extract_file_features
from src.pipelines import build_detector


def compare_moved_code():
    folder = ROOT/'outputs/refactor_phase2'
    proof = json.loads((folder/'validation.json').read_bytes())
    snapshot = folder/'source_snapshot.tar.gz'
    assert sha256(snapshot) == proof['source_snapshot_sha256']
    checks = [
        ('scripts/17_validate_and_calibrate.py', 'src/experiments/calibration.py',
         ('specification', 'trial', 'write_trial_csv', 'save_contour'), ('train', 'evaluate')),
        ('scripts/21_benchmark_champion_unsupervised.py', 'src/experiments/champion.py',
         ('save_frames',), ('train', 'evaluate')),
        ('scripts/21_benchmark_champion_unsupervised.py', 'src/experiments/champion_report.py', ('report',), ()),
    ]
    with tarfile.open(snapshot) as archive:
        for old_name, current_name, helpers, loops in checks:
            previous = ast.parse(archive.extractfile(old_name).read().decode('utf-8'))
            current = ast.parse((ROOT/current_name).read_text(encoding='utf-8'))
            left = {node.name: node for node in previous.body if isinstance(node, ast.FunctionDef)}
            right = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
            for name in helpers:
                assert ast.dump(left[name]) == ast.dump(right[name]), (current_name, name)
            for name in loops:
                before = [ast.dump(node) for node in left[name].body if isinstance(node, ast.For)]
                after = [ast.dump(node) for node in right[name].body if isinstance(node, ast.For)]
                assert before == after, (current_name, name, 'scientific loop changed')
    return {'scientific_loops_ast_unchanged': True, 'moved_helpers_and_renderer_ast_unchanged': True,
            'reference_phase2_snapshot_sha256': sha256(snapshot)}


def compare_champion(folder):
    original = ROOT/'outputs/champion_unsupervised_2026-11-01'
    before = read_frozen(original/'frozen_models.json')['models']
    after = read_frozen(folder/'frozen_models.json')['models']
    for method, model in before.items():
        for key, value in model.items():
            assert json_bytes(after[method][key]) == json_bytes(value), ('model drift', method, key)
    saved = json.loads((original/'evaluation.json').read_bytes())
    actual = json.loads((folder/'evaluation.json').read_bytes())
    for group in ('pipelines', 'protocols'):
        assert actual[group] == saved[group], group+' numerical results drifted'
    cases = frames = train_rows = 0
    for item in saved['pipelines']+saved['protocols']:
        for row in item['files']:
            relative = row['contour_path']
            # CSV parsing keeps this comparison independent of EOL checkout.
            with (original/relative).open(newline='', encoding='utf-8') as stream:
                expected_rows = list(csv.reader(stream))
            with (folder/relative).open(newline='', encoding='utf-8') as stream:
                assert list(csv.reader(stream)) == expected_rows, relative
            frames += len(expected_rows)-1
            cases += 1
    for path in (original/'train_features').rglob('*.csv'):
        relative = path.relative_to(original)
        with path.open(newline='', encoding='utf-8') as stream:
            expected_rows = list(csv.reader(stream))
        with (folder/relative).open(newline='', encoding='utf-8') as stream:
            assert list(csv.reader(stream)) == expected_rows, str(relative)
        train_rows += len(expected_rows)-1
    tables = lambda path: [line for line in path.read_text(encoding='utf-8').splitlines() if line.startswith('|')]
    assert tables(folder/'README.md') == tables(original/'README.md'), 'Report table drift'
    return {'champion_models_matched': len(before), 'champion_file_cases_matched': cases,
            'contour_rows_matched_exactly': frames, 'train_feature_rows_matched_exactly': train_rows,
            'report_tables_matched': True}


def compare_stage4():
    folder = ROOT/'outputs/stage4_validation'
    models = read_frozen(folder/'frozen_models.json')['models']
    evaluation = json.loads((folder/'evaluation.json').read_bytes())
    cases = pipelines = 0
    cache = PitchCache()
    for dataset, directory in [('train', 'TinHieuHuanLuyen'), ('test', 'TinHieuKiemThu')]:
        for method, model in models.items():
            expected = {item['strategy']: item for item in evaluation['protocols']
                        if item['dataset'] == dataset and item['method'] == method}
            for path in sorted((ROOT/directory).glob('*.wav')):
                fd = extract_file_features(path, method, cache=cache)
                for name, labels in predict_strategies(fd, model).items():
                    result = result_from_labels(fd, labels, model['median_size'])
                    row = next(row for row in expected[name]['files'] if row['file'] == path.name)
                    actual = evaluate_result(result, path)
                    assert all(actual[key] == value for key, value in row.items() if key in actual), (name, path.name)
                    cases += 1
            for item in evaluation['pipelines']:
                if item['dataset'] == dataset and item['method'] == method:
                    detector = build_detector(**item['spec'], cache=cache)
                    for row in item['files']:
                        path = ROOT/directory/row['file']
                        actual = evaluate_result(detector.process_file(str(path)), path)
                        assert all(actual[key] == value for key, value in row.items() if key in actual), (item['strategy'], path.name)
                        pipelines += 1
    return {'stage4_protocol_cases_matched': cases, 'stage4_pipeline_cases_matched': pipelines,
            'stage4_calibration_sweep_repeated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/refactor_phase3')
    args = parser.parse_args()
    output = args.output.resolve()
    names = ('validation.json', 'source_snapshot.tar.gz', 'champion_run.tar.gz')
    if any((output/name).exists() for name in names):
        raise FileExistsError('Proof exists; choose a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    provenance = source_manifest(ROOT)
    legacy = RunRegistry(ROOT).verify()
    assert all(entry['status'] == 'verified' for entry in legacy)
    with tempfile.TemporaryDirectory(prefix='fac-runner-proof-') as temporary:
        folder = Path(temporary)/'champion-proof'
        service = RunService(folder)
        train, test = ROOT/'TinHieuHuanLuyen', ROOT/'TinHieuKiemThu'
        service.prepare(default_spec('champion'), train)
        service.train(train)
        frozen_sha = sha256(folder/'frozen_models.json')
        service.evaluate(train, test)
        service.report(train, test)
        # Completed phases verify idempotently; no rerun/refit or report rewrite.
        manifest_sha = sha256(folder/'manifest.json')
        service.train(train, resume=True)
        service.evaluate(train, test, resume=True)
        service.report(train, test, resume=True)
        assert sha256(folder/'manifest.json') == manifest_sha
        assert sha256(folder/'frozen_models.json') == frozen_sha
        result = compare_champion(folder)
        manifest = verify_run(folder)
        with tarfile.open(output/'champion_run.tar.gz', 'w:gz') as archive:
            for name in ['manifest.json', *sorted(manifest['artifacts'])]:
                archive.add(folder/name, arcname=folder.name+'/'+name, recursive=False)
        result['native_run'] = {'run_id': manifest['run_id'], 'experiment_id': manifest['experiment_id'],
            'phase': manifest['phase'], 'artifact_count': len(manifest['artifacts']),
            'completed_stages': list(manifest['completed_stages']), 'resume_without_rewrite': True,
            'bundle_sha256': sha256(output/'champion_run.tar.gz')}
    result.update(compare_stage4())
    result.update(compare_moved_code())
    assert source_manifest(ROOT)['source_sha256'] == provenance['source_sha256'], 'Source changed during proof'
    assert RunRegistry(ROOT).verify() == legacy, 'Existing evidence changed during proof'
    files = [ROOT/'predict.py', *sorted((ROOT/'src').rglob('*.py')),
             *sorted((ROOT/'scripts').glob('*.py')), *sorted((ROOT/'tests').glob('test_*.py'))]
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), feature_version=(3, 9))
    snapshot = output/'source_snapshot.tar.gz'
    write_source_snapshot(ROOT, snapshot, provenance)
    result.update({'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime_record(ROOT),
        'scope': 'Native lifecycle refit on train only; no preset search or test-based selection; historical runs untouched.',
        'python_39_ast_files': len(files), 'source_provenance': provenance,
        'source_snapshot_files': verify_source_snapshot(snapshot, provenance),
        'source_snapshot_sha256': sha256(snapshot), 'legacy_runs_verified': legacy,
        'original_champion_evaluation_sha256': sha256(ROOT/'outputs/champion_unsupervised_2026-11-01/evaluation.json'),
        'original_stage4_evaluation_sha256': sha256(ROOT/'outputs/stage4_validation/evaluation.json'),
        'test_source_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((ROOT/'tests').glob('test_*.py'))}})
    with (output/'validation.json').open('xb') as stream:
        stream.write(json_bytes(result))
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_provenance', 'test_source_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
