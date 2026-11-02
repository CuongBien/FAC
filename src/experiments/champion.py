"""Train/freeze, evaluate, and report four decisions on unmodified pitch frames.

No preset is selected here. Primary models fit active train frames from the
reference frame extraction. Two per-input adaptive controls are separate.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from src.analysis.champion_unsupervised import (
    PRIMARY, ADAPTIVE, experiment_configuration,
    fit_champion_models, predict_decisions,
)
from src.analysis.protocols import threshold_labels
from src.metrics.evaluation import evaluate_result
from src.metrics.aggregation import aggregate
from src.pipeline.cache import PitchCache
from src.pipeline.champion import extract_file_features
from src.experiments.manifest import source_manifest, dataset_manifest, verify_dataset, sha256
from src.experiments.serialization import write_json
from src.experiments.legacy import freeze_json, read_frozen
from src.experiments.snapshots import verify_source_snapshot
from src.pipelines import build_champion, build_baseline

METHODS = ('acf', 'amdf', 'cepstrum')


def save_frames(path, result, initial=None, decision_threshold=None, features=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = result if features is None else features
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_index', 'time_sec', 'active', 'periodicity', 'zcr', 'candidate_f0',
                         'decision_threshold', 'initial_label', 'final_label', 'f0_hz'])
        for i, (time, label, f0) in enumerate(zip(result['frame_times'], result['labels'], result['f0_contour'])):
            writer.writerow([i, float(time), bool(fd.get('is_active', result['labels'] != 'sil')[i]),
                             float(fd['peak_values'][i]),
                             float(fd['zcr_vals'][i]) if 'zcr_vals' in fd else '',
                             float(fd['candidate_f0'][i]) if 'candidate_f0' in fd else '',
                             float(decision_threshold[i]) if decision_threshold is not None else '',
                             str(initial[i]) if initial is not None else '', str(label), float(f0)])


def train(args, *, context):
    folder = Path(args.output_dir)
    manifest = dataset_manifest(args.train_dir)
    paths = sorted(Path(args.train_dir).glob('*.wav'))
    settings = experiment_configuration()
    provenance = context.provenance
    models = {}
    cache = PitchCache()
    for method in METHODS:
        features = []
        for path in paths:
            fd = extract_file_features(path, method, settings, cache=cache)
            features.append(fd)
            save_frames(folder/'train_features'/method/(path.stem+'.csv'), fd)
        models[method] = fit_champion_models(features, method, settings)
        print('TRAIN FIT', method, models[method]['num_train_active'], 'active frames', flush=True)
    # Check data/source did not change during fitting. No test directory is used.
    verify_dataset(args.train_dir, manifest)
    common = {'schema_version': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'train_manifest': manifest, 'provenance': provenance}
    freeze_json(folder/'frozen_models.json', {**common, 'models': models})
    freeze_json(folder/'frozen_config.json', {**common, 'experiment': settings,
        'test_policy': 'Train-only fitting; no test-based selection. Test has historical exposure.',
        'selection': 'No new Champion or strategy is selected; all predefined strategies are reported.'})
    print('FROZEN before test evaluation:', folder, flush=True)


def checked_freezes(folder, train_dir):
    frozen = read_frozen(folder/'frozen_config.json')
    models = read_frozen(folder/'frozen_models.json')
    if frozen['train_manifest'] != models['train_manifest'] or frozen['provenance'] != models['provenance']:
        raise ValueError('Frozen config/models come from different training runs')
    verify_dataset(train_dir, frozen['train_manifest'])
    verify_source_snapshot(folder/'training_source_snapshot.tar.gz', frozen['provenance'])
    runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
    if runtime['snapshot_sha256'] != sha256(folder/'training_source_snapshot.tar.gz'):
        raise ValueError('Runtime/source snapshot mismatch')
    if experiment_configuration() != frozen['experiment']:
        raise ValueError('Native pipeline configuration changed after freeze')
    if source_manifest(ROOT)['source_sha256'] != frozen['provenance']['source_sha256']:
        raise ValueError('Source changed after freeze; rerun from the recorded snapshot or use a new run')
    return frozen, models


def evaluate(args, *, context=None):
    folder = Path(args.output_dir)
    if (folder/'evaluation.json').exists():
        raise FileExistsError('Evaluation exists; do not overwrite a recorded run')
    if context:
        frozen = read_frozen(folder/'frozen_config.json')
        models = read_frozen(folder/'frozen_models.json')
    else:
        frozen, models = checked_freezes(folder, args.train_dir)
    settings = frozen['experiment']
    test_manifest = dataset_manifest(args.test_dir)
    pipelines, protocols = [], []
    comparison_count = 0
    cache = PitchCache()
    for dataset, data_dir in [('train', args.train_dir), ('test', args.test_dir)]:
        paths = sorted(Path(data_dir).glob('*.wav'))
        for method in METHODS:
            rows = {}
            for path in paths:
                fd = extract_file_features(path, method, settings, cache=cache)
                decisions, thresholds, adaptive_models = predict_decisions(
                    fd, models['models'][method], settings, include_input_adaptation=True)
                write_json(folder/'input_models'/dataset/method/(path.stem+'.json'), adaptive_models)
                control = threshold_labels(fd, method, settings['champions'][method]['threshold'])
                decisions = {'champion': control, **decisions}
                detector = build_champion(method, cache=cache)
                native = detector.process_file(str(path))
                for strategy, labels in decisions.items():
                    result = detector.process_feature_labels(fd, labels)
                    if strategy == 'champion':
                        for key in ('labels', 'frame_times', 'f0_contour'):
                            if not np.array_equal(result[key], native[key]):
                                raise AssertionError(f'Native Champion replay mismatch: {method}/{path.name}/{key}')
                        comparison_count += 1
                    row = evaluate_result(result, path)
                    row.update({'num_frames': result['num_frames'],
                                'initial_voiced': int(np.sum(labels == 'v')),
                                'energy_guard_threshold': detector.base_detector.threshold,
                                'post_added_voiced': int(np.sum((labels != 'v') & (result['labels'] == 'v'))),
                                'post_removed_voiced': int(np.sum((labels == 'v') & (result['labels'] != 'v')))})
                    contour = Path('contours')/dataset/method/strategy/(path.stem+'.csv')
                    save_frames(folder/contour, result, initial=labels,
                                decision_threshold=thresholds.get(strategy), features=fd)
                    row['contour_path'] = contour.as_posix()
                    rows.setdefault(strategy, []).append(row)
                baseline = build_baseline(method, cache=cache).process_file(str(path))
                row = evaluate_result(baseline, path)
                contour = Path('contours')/dataset/method/'baseline'/(path.stem+'.csv')
                save_frames(folder/contour, baseline)
                row.update({'num_frames': baseline['num_frames'], 'contour_path': contour.as_posix()})
                rows.setdefault('baseline', []).append(row)
            for strategy, records in rows.items():
                group = 'control' if strategy in ('baseline', 'champion') else (
                    'input_adaptive' if strategy in ADAPTIVE else 'train_fitted')
                item = {'dataset': dataset, 'method': method, 'strategy': strategy,
                        'group': group, 'metrics': aggregate(records), 'files': records}
                (pipelines if group == 'control' else protocols).append(item)
            print('EVALUATED', dataset, method, len(rows), 'strategies', flush=True)
    cache_record = dict(cache)
    verify_dataset(args.train_dir, frozen['train_manifest'])
    verify_dataset(args.test_dir, test_manifest)
    if context is None and source_manifest(ROOT)['source_sha256'] != frozen['provenance']['source_sha256']:
        raise ValueError('Source changed during evaluation; do not publish mixed-source results')
    result = {'config_sha256': sha256(folder/'frozen_config.json'),
              'models_sha256': sha256(folder/'frozen_models.json'),
              'provenance': context.provenance if context else source_manifest(ROOT), 'test_manifest': test_manifest,
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
              'pipelines': pipelines, 'protocols': protocols,
              'native_champion_replays_matched': comparison_count, 'cache': cache_record}
    write_json(folder/'evaluation.json', result)
    fields = ['dataset', 'method', 'strategy', 'group', 'file', 'pred_f0_mean', 'ref_f0_mean',
              'abs_error_mean', 'mape_f0', 'pred_f0_std', 'ref_f0_std', 'abs_error_std', 'mape_std',
              'pred_f0_num', 'ref_f0_num', 'abs_error_num', 'mape_num', 'final_score',
              'num_frames', 'initial_voiced', 'post_added_voiced', 'post_removed_voiced',
              'energy_guard_threshold', 'contour_path']
    with (folder/'per_file.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in pipelines+protocols:
            for row in item['files']:
                writer.writerow({key: (row.get(key) if key in row else item.get(key)) for key in fields})
    print('EVALUATION COMPLETE; no models or presets were changed', flush=True)
