"""Evaluation invariants, train-only selection and legacy experiment imports.

New experiment infrastructure lives in src.experiments. Re-exports preserve the
public imports used by older scripts and callers during the staged refactor.
"""
import numpy as np

from src.pipeline.cache import pitch_cache
from src.metrics.aggregation import aggregate
from src.metrics.evaluation import evaluate_result
from src.experiments.serialization import json_default, write_json
from src.experiments.manifest import sha256, source_manifest, dataset_manifest, verify_dataset
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.experiments.legacy import freeze_json, read_frozen, read_evaluation_bundle


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
