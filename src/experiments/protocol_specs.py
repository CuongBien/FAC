"""Executable v1 protocols: explicit fixed settings, never ignored overrides.

This refactor executes the two existing scientific protocols. Supporting a new
search/fit configuration requires an explicit protocol revision, not silent use
of defaults when JSON and implementation disagree.
"""
from copy import deepcopy

from src import config
from src.analysis.champion_unsupervised import PRIMARY, ADAPTIVE, experiment_configuration
from src.pipeline.contracts import plugin_configuration
from src.pipelines import build_champion, build_baseline, build_detector
from .spec import ExperimentSpec
from .serialization import json_bytes

METHODS = ('acf', 'amdf', 'cepstrum')
HISTORICAL_CHAMPIONS = {'acf': ['BP', 'Energy', 'Viterbi'], 'amdf': ['Energy', 'Viterbi'],
                        'cepstrum': ['BP', 'Clip', 'Energy', 'Viterbi']}
THRESHOLD_GRIDS = {'acf': [.25, .30, .35, .40, .45, .50, .55, .60, .65],
                  'amdf': [.20, .25, .30, .35, .40, .45, .50, .55, .60],
                  'cepstrum': [.02, .04, .06, .08, .10, .12, .14, .16, .18]}
FRAME_GRID = (20., 25., 30.)


def fit_parameters():
    return {'histogram': {'n_bins': 100, 'valley_smooth_sigma': 2.},
            'kmeans': {'clusters': 2, 'max_iter': 100, 'tol': 1e-5},
            'gmm': {'prior': .5, 'max_iter': 150, 'tol': 1e-5},
            'kmeans2d': {'max_iter': 100, 'tol': 1e-5},
            'hmm': {'prior': .5, 'max_iter': 40, 'tol': 1e-4,
                    'A': [[.85, .15], [.15, .85]], 'learn_transitions': False},
            'dynamic': {'window_frames': 50, 'alpha': .30,
                        'context': 'within active segments; STE normalized within each segment; offline window'},
            'energy_guard': 'fixed_champion_threshold_for_every_strategy',
            'zcr_source': 'DC-centered pitch_frames after bandpass/center clipping'}


def pipeline_document(detector):
    base = getattr(detector, 'base_detector', detector)
    return {**{key: getattr(base, key) for key in (
        'frame_duration_ms', 'hop_duration_ms', 'f0_min', 'f0_max', 'ste_silence_ratio',
        'use_median_filter', 'median_size')}, 'reference_threshold': base.threshold,
        'pre_processors': plugin_configuration(getattr(detector, 'pre_processors', [])),
        'post_processors': plugin_configuration(getattr(detector, 'post_processors', []))}


def default_document(protocol):
    if protocol not in ('champion', 'stage4'):
        raise ValueError('Unknown experiment protocol')
    document = {'schema_version': 1, 'protocol': protocol, 'name': protocol+'-v1',
                'metric_version': 'legacy-v1',
                'controls': {'baseline': {method: pipeline_document(build_baseline(method)) for method in METHODS}},
                'pipelines': {method: pipeline_document(build_champion(method) if protocol == 'champion'
                                                       else build_baseline(method)) for method in METHODS},
                'fit': {'mode': 'frozen', 'population': 'train_active', 'strategies': list(PRIMARY),
                        'input_adaptive_controls': list(ADAPTIVE), 'parameters': fit_parameters()}}
    if protocol == 'stage4':
        document['controls']['historical_champion'] = {
            method: pipeline_document(build_detector(method, HISTORICAL_CHAMPIONS[method])) for method in METHODS}
        document['fit'].update(population='train_all', populations=['train_all', 'train_active'],
            strategies=['historical_static']+[
                f'{name}_train_{population}' for population in ('all', 'active')
                for name in ('median', 'otsu', 'valley', 'kmeans', 'gmm', 'dynamic')
            ]+['kmeans2d_train_active', 'hmm_train_active'])
        document['fit']['parameters']['dynamic']['context'] = 'whole input timeline including inactive frames; offline window'
        document['fit']['parameters']['zcr_source'] = 'DC-centered raw frames; all frames measured'
        document['fit']['parameters']['energy_guard'] = 'not applied to raw protocols'
        document['calibration'] = {'historical_champions': deepcopy(HISTORICAL_CHAMPIONS),
            'threshold_grids': deepcopy(THRESHOLD_GRIDS), 'frame_grid_ms': list(FRAME_GRID),
            'hop_ms': 10., 'clipping_ratio': .4, 'phase1_plugins': ['BP', 'Clip', 'Energy', 'Viterbi'],
            'shortlist': 'top two phase1 combinations plus historical Champion',
            'selection': 'train file macro Final Score; ties: worst file, plugin count, distance from 25ms, plugin names, threshold',
            'historical_thresholds': {method: {'baseline': config.baseline_threshold(method),
                'with_clip': config.threshold_for_plugins(method, ['Clip']),
                'without_clip': config.threshold_for_plugins(method, ['BP'])} for method in METHODS},
            'native_champions': experiment_configuration()}
    return document


def validate_protocol(spec):
    document = spec.to_dict()
    protocol = document.get('protocol')
    expected = default_document(protocol)
    # Human labels are metadata; all settings, control lists and search rules
    # are validated, including extra keys, rather than silently ignored.
    for value in (document, expected):
        value.pop('name', None)
        value.pop('description', None)
    if json_bytes(document) != json_bytes(expected):
        raise ValueError('Executable protocol v1 configuration changed; unsupported overrides require a new protocol revision')
    return protocol


def default_spec(protocol):
    return ExperimentSpec(default_document(protocol))
