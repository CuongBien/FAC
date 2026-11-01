"""Stage 4: train-only calibration, checksum freeze, then held-out evaluation.

Run train first, then evaluate using the resulting immutable artifacts.
Test data already has historical analyses in this project; this is a fixed
train-selection protocol, not a claim of a pristine unseen test set.
"""
import csv
from datetime import datetime, timezone
import itertools
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]

from src import config
from src.pipelines import build_detector
from src.pipeline.cache import PitchCache
from src.pipeline.features import extract_file_features
from src.analysis.validation import (
    aggregate, evaluate_result, select_trial, selection_key,
)
from src.experiments.manifest import source_manifest, dataset_manifest, verify_dataset, sha256
from src.experiments.serialization import write_json
from src.experiments.legacy import freeze_json, read_frozen
from src.experiments.snapshots import verify_source_snapshot
from src.analysis.protocols import fit_models, predict_strategies, result_from_labels


from .protocol_specs import METHODS, HISTORICAL_CHAMPIONS, THRESHOLD_GRIDS, FRAME_GRID


def specification(method, plugins=(), threshold=None, frame=25.):
    return {'method': method, 'plugins': list(plugins),
            'threshold': config.threshold_for_plugins(method, plugins) if threshold is None else float(threshold),
            'frame_duration_ms': float(frame), 'hop_duration_ms': 10., 'clipping_ratio': .40}


def trial(spec, paths, tag, *, cache=None):
    detector = build_detector(**spec, cache=cache)
    rows = [evaluate_result(detector.process_file(str(path)), path) for path in paths]
    return {'id': tag, 'spec': spec, 'metrics': aggregate(rows), 'files': rows}


def write_trial_csv(path, trials):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['trial', 'method', 'plugins', 'threshold', 'frame_ms', 'hop_ms', 'file',
              'pred_f0_mean', 'ref_f0_mean', 'mape_f0', 'pred_f0_std', 'ref_f0_std', 'mape_std',
              'pred_f0_num', 'ref_f0_num', 'mape_num', 'final_score']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in trials:
            spec = item['spec']
            for row in item['files']:
                writer.writerow({'trial': item['id'], 'method': spec['method'], 'plugins': '+'.join(spec['plugins']),
                                 'threshold': spec['threshold'], 'frame_ms': spec['frame_duration_ms'],
                                 'hop_ms': spec['hop_duration_ms'], **{key: row.get(key) for key in fields[6:]}})


def train(args, *, context):
    output = Path(args.output_dir)
    paths = sorted(Path(args.train_dir).glob('*.wav'))
    manifest = dataset_manifest(args.train_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = context.provenance
    started = time.perf_counter()
    all_trials, winners, sweep = [], {}, []
    combos = [combo for size in range(5) for combo in itertools.combinations(('BP', 'Clip', 'Energy', 'Viterbi'), size)]
    cache_stats = PitchCache()
    for method in METHODS:
        historical = []
        for index, combo in enumerate(combos):
            item = trial(specification(method, combo), paths, f'{method}_sweep_{index:02d}', cache=cache_stats)
            historical.append(item)
            print(item['id'], item['metrics']['final_score'], flush=True)
        sweep.extend(historical)
        all_trials.extend(historical)
        # Keep the two best combinations at historical settings plus the
        # historical Champion, before any test feature is read.
        shortlisted = [item['spec']['plugins'] for item in sorted(historical, key=selection_key)[:2]]
        if HISTORICAL_CHAMPIONS[method] not in shortlisted:
            shortlisted.append(HISTORICAL_CHAMPIONS[method])
        candidates = list(historical)
        index = 0
        seen = {(tuple(item['spec']['plugins']), item['spec']['threshold'], 25.) for item in historical}
        for combo in shortlisted:
            thresholds = sorted(set(THRESHOLD_GRIDS[method] + [config.threshold_for_plugins(method, combo)]))
            for frame, threshold in itertools.product(FRAME_GRID, thresholds):
                key = (tuple(combo), threshold, frame)
                if key in seen:
                    continue
                seen.add(key)
                item = trial(specification(method, combo, threshold, frame), paths, f'{method}_tune_{index:03d}', cache=cache_stats)
                candidates.append(item)
                all_trials.append(item)
                index += 1
            print(method, '+'.join(combo) or 'Baseline', 'frame/threshold grid complete', flush=True)
        winner = select_trial(candidates)
        winners[method] = winner
        write_json(output/'train'/f'{method}_trials.json', candidates)
        print('FROZEN WINNER', method, winner['spec'], winner['metrics']['final_score'], flush=True)
    elapsed = time.perf_counter()-started
    cache_record = dict(cache_stats)
    write_json(output/'train/plugin_sweep.json', sweep)
    write_trial_csv(output/'train/all_trials_per_file.csv', all_trials)
    models = {}
    for method in METHODS:
        train_features = [extract_file_features(str(path), method) for path in paths]
        models[method] = fit_models(train_features, method)
    common = {'schema_version': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'train_dir': 'train', 'train_manifest': manifest,
              'provenance': provenance,
              'test_policy': 'no test data consumed by train or selection; historical test analyses already exist'}
    freeze_json(output/'frozen_models.json', {**common, 'models': models})
    freeze_json(output/'frozen_config.json', {
        **common, 'selection': 'mean of file-level train Final Scores; ties: worst file, plugin count, distance from 25ms, plugin names, threshold',
        'search_space': {'phase1': 'all 16 combinations at historical threshold and 25/10ms',
                         'phase2': 'top 2 phase1 combinations plus historical Champion',
                         'threshold_grids': THRESHOLD_GRIDS, 'frame_grid_ms': FRAME_GRID,
                         'hop_ms': 10, 'clipping_ratio': .40},
        'selected': {method: item['spec'] for method, item in winners.items()},
        'train_metrics': {method: item['metrics'] for method, item in winners.items()},
        'selected_trial_ids': {method: item['id'] for method, item in winners.items()},
        'num_trials': len(all_trials), 'num_pipeline_file_pairs': len(all_trials)*len(paths),
        'cache': cache_record, 'calibration_seconds': elapsed,
    })
    print('TRAIN FINISHED; checksummed models/config written BEFORE test evaluation', flush=True)


def save_contour(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_sec', 'label', 'f0_hz'])
        writer.writerows(zip(result['frame_times'], result['labels'], result['f0_contour']))


def evaluate(args, *, context=None):
    output = Path(args.output_dir)
    frozen = read_frozen(output/'frozen_config.json')
    models = read_frozen(output/'frozen_models.json')
    verify_source_snapshot(output/'training_source_snapshot.tar.gz', frozen['provenance'])
    verify_dataset(args.train_dir, frozen['train_manifest'])
    if frozen['train_manifest'] != models['train_manifest']:
        raise ValueError('Frozen models/config use different training data')
    if (output/'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists; retain it and choose a new run directory')
    pipeline_rows, protocol_rows = [], []
    for dataset, folder in [('train', args.train_dir), ('test', args.test_dir)]:
        for method in METHODS:
            paths = sorted(Path(folder).glob('*.wav'))
            pipeline_configs = {
                'baseline': specification(method),
                'historical_champion': specification(method, HISTORICAL_CHAMPIONS[method]),
                'selected_champion': frozen['selected'][method],
            }
            for name, spec in pipeline_configs.items():
                detector = build_detector(**spec)
                records = []
                for path in paths:
                    result = detector.process_file(str(path))
                    row = evaluate_result(result, path)
                    records.append(row)
                    if name == 'selected_champion':
                        save_contour(output/'contours'/dataset/method/(path.stem+'.csv'), result)
                pipeline_rows.append({'dataset': dataset, 'method': method, 'strategy': name,
                                      'spec': spec, 'metrics': aggregate(records), 'files': records})
            protocol_records = {}
            for path in paths:
                fd = extract_file_features(str(path), method)
                for strategy, labels in predict_strategies(fd, models['models'][method]).items():
                    result = result_from_labels(fd, labels)
                    row = evaluate_result(result, path)
                    protocol_records.setdefault(strategy, []).append(row)
            for strategy, rows in protocol_records.items():
                protocol_rows.append({'dataset': dataset, 'method': method, 'strategy': strategy,
                                      'metrics': aggregate(rows), 'files': rows})
            print('EVALUATED', dataset, method, '3 pipelines +', len(protocol_records), 'protocols', flush=True)
    write_json(output/'evaluation.json', {
        'config_sha256': (output/'frozen_config.sha256').read_text().strip(),
        'models_sha256': (output/'frozen_models.sha256').read_text().strip(),
        'provenance': context.provenance if context else source_manifest(ROOT), 'test_manifest': dataset_manifest(args.test_dir),
        'pipelines': pipeline_rows, 'protocols': protocol_rows,
    })
    write_trial_csv(output/'evaluation_pipelines_per_file.csv', [
        {'id': row['dataset']+'_'+row['strategy'], **row} for row in pipeline_rows])
    with (output/'evaluation_protocols_per_file.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'method', 'strategy', 'file', 'pred_mean', 'pred_std', 'pred_num', 'mape_mean', 'mape_std', 'mape_num', 'final_score'])
        for item in protocol_rows:
            for row in item['files']:
                writer.writerow([item['dataset'], item['method'], item['strategy'], row['file']] +
                                [row.get(k) for k in ('pred_f0_mean', 'pred_f0_std', 'pred_f0_num', 'mape_f0', 'mape_std', 'mape_num', 'final_score')])
    print('EVALUATION COMPLETE; frozen choices were not changed', flush=True)
