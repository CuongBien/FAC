"""Read-only numerical replay, plus explicit capture of a pre-refactor reference.

Capture is for a reviewed baseline checkout only. Verify never fits frozen
models, updates expected values, or writes into the historical benchmark run.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.champion_unsupervised import predict_decisions
from src.pipeline.champion import extract_file_features
from src.analysis.protocols import threshold_labels
from src.analysis.validation import evaluate_result
from src.experiments.serialization import json_bytes
from src.experiments.manifest import sha256, dataset_manifest, source_manifest
from src.experiments.registry import RunRegistry
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.pipelines import build_baseline, build_champion
from src.pipeline.cache import PitchCache

METHODS = ('acf', 'amdf', 'cepstrum')
RUN = ROOT/'outputs/champion_unsupervised_2026-11-01'


def module(number, filename):
    spec = importlib.util.spec_from_file_location('reference_'+str(number), ROOT/'scripts'/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fingerprint(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Object arrays contain pointers; serialize labels by value instead.
            raw = json_bytes(value.tolist()) if value.dtype.hasobject else value.tobytes()
            result[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                           'sha256': hashlib.sha256(raw).hexdigest()}
        elif value is None or isinstance(value, (str, int, float, bool)):
            # Absolute path is intentionally excluded from the identity.
            if key != 'wav_path':
                result[key] = value
    return result


def capture():
    old15 = module(15, '15_evaluate_unsupervised_thresholds.py')
    old16 = module(16, '16_evaluate_advanced_unsupervised.py')
    records = {}
    cache = PitchCache()
    for dataset, folder in [('train', 'TinHieuHuanLuyen'), ('test', 'TinHieuKiemThu')]:
        paths = sorted((ROOT/folder).glob('*.wav'))
        for method in METHODS:
            records[f'{dataset}/{method}/histogram'] = fingerprint(
                old15.extract_all_and_active_metrics([str(p) for p in paths], method))
            for path in paths:
                prefix = f'{dataset}/{method}/{path.stem}'
                records[prefix+'/raw'] = fingerprint(old16.extract_features_for_file(str(path), method))
                for name, builder in [('baseline', build_baseline), ('champion', build_champion)]:
                    records[prefix+'/'+name] = fingerprint(builder(method, cache=cache).process_file(str(path)))
    return {'schema_version': 1, 'records': records,
            'datasets': {folder: dataset_manifest(ROOT/folder)
                         for folder in ('TinHieuHuanLuyen', 'TinHieuKiemThu')}}


def verify_reference(expected):
    actual = capture()
    assert actual['datasets'] == expected['datasets'], 'Dataset changed'
    assert set(actual['records']) == set(expected['records'])
    for name, record in expected['records'].items():
        for key, value in record.items():
            assert actual['records'][name].get(key) == value, f'Baseline drift: {name}/{key}'
    return len(expected['records'])


def verify_champion():
    frozen = json.loads((RUN/'frozen_config.json').read_text(encoding='utf-8'))
    models = json.loads((RUN/'frozen_models.json').read_text(encoding='utf-8'))['models']
    evaluation = json.loads((RUN/'evaluation.json').read_text(encoding='utf-8'))
    expected = {(item['dataset'], item['method'], item['strategy'], row['file']): row
                for item in evaluation['pipelines']+evaluation['protocols'] for row in item['files']}
    cases = frames = train_rows = 0
    cache = PitchCache()
    for dataset, folder in [('train', 'TinHieuHuanLuyen'), ('test', 'TinHieuKiemThu')]:
        for method in METHODS:
            for path in sorted((ROOT/folder).glob('*.wav')):
                fd = extract_file_features(path, method, frozen['experiment'], cache=cache)
                if dataset == 'train':
                    with (RUN/'train_features'/method/(path.stem+'.csv')).open(newline='', encoding='utf-8') as stream:
                        rows = list(csv.DictReader(stream))
                    assert len(rows) == fd['num_frames']
                    for i, row in enumerate(rows):
                        for column, key in [('time_sec', 'frame_times'), ('periodicity', 'peak_values'),
                                            ('zcr', 'zcr_vals'), ('candidate_f0', 'candidate_f0')]:
                            assert float(row[column]) == fd[key][i], (path.name, column, i)
                        assert (row['active'] == 'True') == fd['is_active'][i]
                    train_rows += len(rows)
                decisions, _, _ = predict_decisions(fd, models[method], frozen['experiment'], True)
                detector = build_champion(method, cache=cache)
                decisions = {'champion': threshold_labels(fd, method, detector.base_detector.threshold), **decisions}
                results = {name: detector.process_feature_labels(fd, labels) for name, labels in decisions.items()}
                results['baseline'] = build_baseline(method, cache=cache).process_file(str(path))
                for name, result in results.items():
                    row = expected[dataset, method, name, path.name]
                    metrics = evaluate_result(result, path)
                    assert all(metrics[key] == value for key, value in row.items() if key in metrics), (path.name, name)
                    with (RUN/row['contour_path']).open(newline='', encoding='utf-8') as stream:
                        rows = list(csv.DictReader(stream))
                    assert len(rows) == result['num_frames']
                    for i, saved in enumerate(rows):
                        assert float(saved['time_sec']) == result['frame_times'][i]
                        assert saved['final_label'] == result['labels'][i]
                        assert float(saved['f0_hz']) == result['f0_contour'][i]
                        if name != 'baseline':
                            assert saved['initial_label'] == decisions[name][i]
                            assert float(saved['candidate_f0']) == fd['candidate_f0'][i]
                    frames += len(rows)
                    cases += 1
    assert cases == len(expected)
    return {'file_cases_matched': cases, 'contour_rows_matched_exactly': frames,
            'train_feature_rows_matched_exactly': train_rows,
            'original_evaluation_sha256': sha256(RUN/'evaluation.json'),
            'original_models_sha256': sha256(RUN/'frozen_models.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture', 'verify'))
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/refactor_phase2/reference.json')
    parser.add_argument('--output', type=Path, help='Write a new proof bundle; never overwrite existing proof files')
    args = parser.parse_args()
    if args.phase == 'capture':
        value = capture()
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        with args.reference.open('xb') as stream:
            stream.write(json_bytes(value))
        print('Captured', len(value['records']), 'reference records')
    else:
        if args.output:
            for name in ('validation.json', 'source_snapshot.tar.gz'):
                if (args.output/name).exists():
                    raise FileExistsError('Proof exists; choose a fresh output directory')
        provenance = source_manifest(ROOT)
        registry = RunRegistry(ROOT)
        legacy = registry.verify()
        if any(entry['status'] != 'verified' for entry in legacy):
            raise ValueError('Historical evidence failed integrity verification')
        expected = json.loads(args.reference.read_text(encoding='utf-8'))
        records = verify_reference(expected)
        result = {'reference_records_matched': records, **verify_champion()}
        if source_manifest(ROOT)['source_sha256'] != provenance['source_sha256']:
            raise ValueError('Source changed during replay')
        if registry.verify() != legacy:
            raise ValueError('Legacy evidence changed during replay')
        if args.output:
            files = [ROOT/'predict.py', *sorted((ROOT/'src').rglob('*.py')),
                     *sorted((ROOT/'scripts').glob('*.py')), *sorted((ROOT/'tests').glob('test_*.py'))]
            for path in files:
                ast.parse(path.read_text(encoding='utf-8-sig'), feature_version=(3, 9))
            args.output.mkdir(parents=True, exist_ok=True)
            snapshot = args.output/'source_snapshot.tar.gz'
            write_source_snapshot(ROOT, snapshot, provenance)
            source_count = verify_source_snapshot(snapshot, provenance)
            proof = {**result, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Exact refactor replay; frozen train models unchanged; input-adaptive controls replay their declared fits.',
                'reference_sha256': sha256(args.reference), 'source_provenance': provenance,
                'source_snapshot_sha256': sha256(snapshot), 'source_snapshot_files': source_count,
                'python_39_ast_files': len(files), 'legacy_runs_verified': legacy,
                'runtime': {'python': sys.version.split()[0],
                            **{name: version(name) for name in ('numpy', 'scipy', 'matplotlib')}},
                'test_source_sha256': {path.relative_to(ROOT).as_posix(): sha256(path)
                                      for path in sorted((ROOT/'tests').glob('test_*.py'))}}
            with (args.output/'validation.json').open('xb') as stream:
                stream.write(json_bytes(proof))
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
