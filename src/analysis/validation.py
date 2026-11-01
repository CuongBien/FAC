"""Evaluation invariants, train-only selection and legacy experiment imports.

New experiment infrastructure lives in src.experiments. Re-exports preserve the
public imports used by older scripts and callers during the staged refactor.
"""
from pathlib import Path
import numpy as np

from src.pipeline.cache import pitch_cache
from src.analysis.evaluation import mean_metric, evaluate_against_ground_truth
from src.experiments.serialization import json_default, write_json
from src.experiments.manifest import sha256, source_manifest, dataset_manifest, verify_dataset
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.experiments.legacy import freeze_json, read_frozen, read_evaluation_bundle


def aggregate(rows):
    keys = ('abs_error_mean', 'mape_f0', 'abs_error_std', 'mape_std', 'abs_error_num', 'mape_num', 'final_score')
    return {key: mean_metric((row.get(key) for row in rows), digits=None) for key in keys}


def evaluate_result(result, wav_path):
    f0 = np.asarray(result['f0_contour'])
    labels = np.asarray(result['labels'])
    voiced = labels == 'v'
    if not (len(f0) == len(labels) == len(result['frame_times'])):
        raise AssertionError('Frame alignment mismatch')
    if not np.all(np.isfinite(f0)) or np.any(f0[~voiced] != 0):
        raise AssertionError('Non-finite F0 or pitch outside voiced labels')
    if np.any((f0[voiced] < 70) | (f0[voiced] > 400)) or int(voiced.sum()) != result['num_voiced']:
        raise AssertionError('Pitch range/count mismatch')
    return {'file': Path(wav_path).name, **evaluate_against_ground_truth(result, str(Path(wav_path).with_suffix('.lab')))}


def selection_key(trial):
    score = trial['metrics']['final_score']
    if score is None or not np.isfinite(score):
        return (float('inf'),)
    spec = trial['spec']
    worst = max(row['final_score'] for row in trial['files'])
    return (score, worst, len(spec['plugins']), abs(spec['frame_duration_ms']-25),
            tuple(spec['plugins']), spec['threshold'])


def select_trial(trials):
    valid = [trial for trial in trials if trial['metrics']['final_score'] is not None
             and np.isfinite(trial['metrics']['final_score'])]
    if not valid:
        raise ValueError('No complete train scores are available for selection')
    return min(valid, key=selection_key)
