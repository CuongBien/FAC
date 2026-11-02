"""Pure adapters from the selected run schema to comparison contexts."""
from copy import deepcopy
import json
from src.pipeline.contracts import FeatureSpec
from .aggregation import FILE_MACRO
from .contracts import EvaluationContext

ADAPTIVE = ('kmeans2d_input_active',)


def evaluation_context(spec, frozen, item, dataset_identity, train_identity, model_sha256,
                       *, files, input_model_sha256=None):
    method, strategy = item['method'], item['strategy']
    if spec['protocol'] != 'champion':
        raise ValueError('Unknown run protocol')
    allowed = {'baseline', 'champion'} | set(spec['fit']['strategies']) | set(spec['fit']['input_adaptive_controls'])
    if strategy not in allowed:
        raise ValueError('Unknown strategy in the declared evaluation protocol')
    pipeline = deepcopy(spec['controls']['baseline'][method] if strategy == 'baseline' else spec['pipelines'][method])
    control = strategy in ('baseline', 'champion')
    population = 'train_active'
    active_fit = strategy in ADAPTIVE
    mode = 'control' if control else ('input_adaptive' if active_fit else 'frozen')
    parameters = {'reference_threshold': pipeline['reference_threshold']}
    if active_fit:
        parameters['input_models_recorded'] = input_model_sha256 is not None
    if strategy.startswith('dynamic'):
        parameters['dynamic'] = deepcopy(spec['fit']['parameters']['dynamic'])
    feature = FeatureSpec(method, pipeline['frame_duration_ms'], pipeline['hop_duration_ms'],
                          pipeline['f0_min'], pipeline['f0_max'], pipeline['ste_silence_ratio'],
                          'baseline' if strategy == 'baseline' else 'plugins',
                          json.dumps(pipeline['pre_processors']))
    return EvaluationContext({'schema_version': 1, 'dataset_identity': dataset_identity, 'files': list(files), 'split': item['dataset'],
        'metric_version': spec['metric_version'], 'aggregation': FILE_MACRO,
        'feature_contract': feature.to_dict(), 'post_processors': pipeline['post_processors'],
        'smoothing': {'enabled': pipeline['use_median_filter'], 'size': pipeline['median_size']},
        'fit': {'mode': mode, 'population': None if control else ('input_active' if active_fit else population),
                'training_identity': train_identity if mode == 'frozen' else None},
        'decision': {'strategy': strategy, 'parameters': parameters,
                     'model_sha256': None if control else (input_model_sha256 if active_fit else model_sha256)}})
