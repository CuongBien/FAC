"""Explicit executable configuration for the selected unsupervised experiment."""
from src.analysis.champion_unsupervised import PRIMARY, ADAPTIVE
from src.pipelines import build_champion, build_baseline
from src.pipeline.contracts import plugin_configuration
from .spec import ExperimentSpec
from .serialization import json_bytes
METHODS = ('acf', 'amdf', 'cepstrum')

def fit_parameters():
    return {'histogram': {'n_bins': 100},
            'kmeans': {'clusters': 2, 'max_iter': 100, 'tol': 1e-5},
            'kmeans2d': {'max_iter': 100, 'tol': 1e-5},
            'dynamic': {'base': 'otsu', 'window_frames': 50, 'alpha': .30,
                        'context': 'within active segments; offline centered window'}}

def pipeline_document(detector):
    base = getattr(detector, 'base_detector', detector)
    return {**{key: getattr(base, key) for key in (
        'frame_duration_ms', 'hop_duration_ms', 'f0_min', 'f0_max', 'ste_silence_ratio',
        'use_median_filter', 'median_size')}, 'reference_threshold': base.threshold,
        'pre_processors': plugin_configuration(getattr(detector, 'pre_processors', [])),
        'post_processors': plugin_configuration(getattr(detector, 'post_processors', []))}

def default_document(protocol):
    if protocol != 'champion':
        raise ValueError('Unknown experiment protocol')
    return {'schema_version': 1, 'protocol': protocol, 'name': 'selected-unsupervised-v2',
            'metric_version': 'legacy-v1',
            'controls': {'baseline': {m: pipeline_document(build_baseline(m)) for m in METHODS}},
            'pipelines': {m: pipeline_document(build_champion(m)) for m in METHODS},
            'fit': {'mode': 'frozen', 'population': 'train_active', 'strategies': list(PRIMARY),
                    'input_adaptive_controls': list(ADAPTIVE), 'parameters': fit_parameters()}}

def validate_protocol(spec):
    document = spec.to_dict()
    expected = default_document(document.get('protocol'))
    for value in (document, expected):
        value.pop('name', None)
        value.pop('description', None)
    if json_bytes(document) != json_bytes(expected):
        raise ValueError('Unsupported executable protocol settings')
    return 'champion'

def default_spec(protocol):
    return ExperimentSpec(default_document(protocol))
