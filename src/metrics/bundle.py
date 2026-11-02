"""Pure adapters from the two fixed run schemas to comparison contexts."""
from copy import deepcopy
import json
from src.pipeline.contracts import FeatureSpec
from .aggregation import FILE_MACRO
from .contracts import EvaluationContext

PLUGIN_NAMES = {'BP': 'BandpassPreFilter', 'Clip': 'CenterClipping',
                'Energy': 'EnergyEdgeExtension', 'Viterbi': 'ViterbiTracking'}
ADAPTIVE = ('kmeans2d_input_active', 'hmm_input_active')


def _selected_pipeline(spec, selected, method):
    pipeline = deepcopy(spec['pipelines'][method])
    catalog = {}
    sources = [spec['pipelines'], *spec['controls'].values()]
    for source in sources:
        for settings in source.values():
            for stage in ('pre_processors', 'post_processors'):
                for plugin in settings[stage]:
                    if plugin['name'] in catalog and catalog[plugin['name']] != plugin:
                        raise ValueError('Ambiguous v1 plugin parameters')
                    catalog[plugin['name']] = plugin
    pipeline.update({key: selected[key] for key in ('frame_duration_ms', 'hop_duration_ms')})
    pipeline['reference_threshold'] = selected['threshold']
    pipeline['pre_processors'], pipeline['post_processors'] = [], []
    for name in selected['plugins']:
        plugin = deepcopy(catalog[PLUGIN_NAMES[name]])
        if name == 'Clip' and selected['clipping_ratio'] != plugin['parameters']['clipping_ratio']:
            raise ValueError('Selected clipping differs from the fixed v1 catalog')
        pipeline['pre_processors' if name in ('BP', 'Clip') else 'post_processors'].append(plugin)
    return pipeline


def evaluation_context(spec, frozen, item, dataset_identity, train_identity, model_sha256,
                       *, files, input_model_sha256=None):
    method, strategy = item['method'], item['strategy']
    controls = ('baseline', 'champion') if spec['protocol'] == 'champion' else (
        'baseline', 'historical_champion', 'selected_champion', 'historical_static')
    allowed = set(controls) | set(spec['fit']['strategies']) | set(spec['fit']['input_adaptive_controls'])
    if strategy not in allowed:
        raise ValueError('Unknown strategy in the declared evaluation protocol')
    if spec['protocol'] == 'champion':
        pipeline = deepcopy(spec['controls']['baseline'][method] if strategy == 'baseline' else spec['pipelines'][method])
        control = strategy in ('baseline', 'champion')
        profile = 'baseline' if strategy == 'baseline' else 'plugins'
        population = 'train_active'
    elif spec['protocol'] == 'stage4':
        control = strategy in ('baseline', 'historical_champion', 'selected_champion', 'historical_static')
        if strategy == 'historical_champion':
            pipeline = deepcopy(spec['controls']['historical_champion'][method])
        elif strategy == 'selected_champion':
            pipeline = _selected_pipeline(spec, frozen['selected'][method], method)
        else:
            pipeline = deepcopy(spec['pipelines'][method])
        if strategy in ('baseline', 'historical_champion', 'selected_champion'):
            profile = 'plugins' if pipeline['pre_processors'] or pipeline['post_processors'] else 'baseline'
        else:
            profile = 'centered_all'
            pipeline['pre_processors'], pipeline['post_processors'] = [], []
        population = 'train_all' if strategy.endswith('_train_all') else 'train_active'
    else:
        raise ValueError('Unknown run protocol')
    active_fit = strategy in ADAPTIVE
    mode = 'control' if control else ('input_adaptive' if active_fit else 'frozen')
    parameters = {'reference_threshold': pipeline['reference_threshold']}
    if active_fit:
        parameters['input_models_recorded'] = input_model_sha256 is not None
    if strategy.startswith('dynamic'):
        parameters['dynamic'] = deepcopy(spec['fit']['parameters']['dynamic'])
    feature = FeatureSpec(method, pipeline['frame_duration_ms'], pipeline['hop_duration_ms'],
                          pipeline['f0_min'], pipeline['f0_max'], pipeline['ste_silence_ratio'], profile,
                          # FrameFeatures stores native pre-processing by actual parameter names.
                          json.dumps(pipeline['pre_processors']))
    return EvaluationContext({'schema_version': 1, 'dataset_identity': dataset_identity, 'files': list(files), 'split': item['dataset'],
        'metric_version': spec['metric_version'], 'aggregation': FILE_MACRO,
        'feature_contract': feature.to_dict(), 'post_processors': pipeline['post_processors'],
        'smoothing': {'enabled': pipeline['use_median_filter'], 'size': pipeline['median_size']},
        'fit': {'mode': mode, 'population': None if control else ('input_active' if active_fit else population),
                'training_identity': train_identity if mode == 'frozen' else None},
        'decision': {'strategy': strategy, 'parameters': parameters,
                     'model_sha256': None if control else (input_model_sha256 if active_fit else model_sha256)}})
