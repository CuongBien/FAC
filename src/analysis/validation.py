"""Evaluation invariants, train-only selection and legacy experiment imports.

New experiment infrastructure lives in src.experiments. Re-exports preserve the
public imports used by older scripts and callers during the staged refactor.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np

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


@contextmanager
def pitch_cache():
    """Cache deterministic frame calculations only, with exact input/config keys.

    Detector decisions, medians, extension and Viterbi DP are still run normally.
    The patch is confined to the benchmark context and is restored on exit.
    """
    import src.core.pitch_detector as core
    import src.plugins.plugin_detector as plugin
    from src.plugins.post_processing.viterbi_tracking import ViterbiTrackingPlugin
    cache = {}
    stats = {'hits': 0, 'misses': 0}

    def wrapper(original, method=False):
        def call(*args, **kwargs):
            owner, frame = (args[0], args[1]) if method else (None, args[0])
            rest = args[2:] if method else args[1:]
            values = np.ascontiguousarray(frame)
            owner_key = (owner.f0_min, owner.f0_max, owner.max_candidates) if method else ()
            key = (original, values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest(),
                   rest, tuple(sorted(kwargs.items())), owner_key)
            if key in cache:
                stats['hits'] += 1
            else:
                stats['misses'] += 1
                cache[key] = original(*args, **kwargs)
            return deepcopy(cache[key])
        return call

    with ExitStack() as stack:
        for name in ('find_f0_acf', 'find_f0_amdf', 'find_f0_cepstrum'):
            wrapped = wrapper(getattr(core, name))
            stack.enter_context(patch.object(core, name, wrapped))
            stack.enter_context(patch.object(plugin, name, wrapped))
        for name in ('_extract_acf_candidates', '_extract_amdf_candidates', '_extract_cepstrum_candidates'):
            stack.enter_context(patch.object(ViterbiTrackingPlugin, name, wrapper(getattr(ViterbiTrackingPlugin, name), True)))
        yield stats
