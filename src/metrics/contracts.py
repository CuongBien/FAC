"""Immutable JSON metadata: physical evaluation space and fitting population."""
from dataclasses import InitVar, dataclass, field
import json
import math
import re

from src.pipeline.contracts import FeatureSpec
from .aggregation import PROFILES

VERSION = 'legacy-v1'


def metric_contract():
    return {'metric_version': VERSION, 'units': 'percent_except_absolute_hz_and_frame_count',
        'field_units': {'abs_error_mean': 'Hz', 'abs_error_std': 'Hz', 'abs_error_num': 'frames',
                        'mape_f0': 'percent', 'mape_std': 'percent', 'mape_num': 'percent', 'final_score': 'percent'},
        'file_digits': 2, 'final_score': 'mean_of_three_unrounded_MAPEs_then_round_2',
        'count_error': 'absolute_round_2_then_round_to_integer',
        'zero_reference': 'absolute_error_defined_MAPE_null',
        'missing_reference': 'component_null_Final_Score_null',
        'missing_file': 'propagate_null_never_drop',
        'classification_without_timed_labels': 'null', 'aggregation_profiles': list(PROFILES)}


def canonical_json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest_or_none(value):
    return value is None or isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


@dataclass(frozen=True)
class EvaluationContext:
    document: InitVar[dict]
    _canonical: str = field(init=False, repr=False)

    def __post_init__(self, document):
        value = json.loads(canonical_json(document))
        keys = {'schema_version', 'dataset_identity', 'files', 'split', 'metric_version', 'aggregation',
                'feature_contract', 'post_processors', 'smoothing', 'fit', 'decision'}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError('Evaluation context requires complete explicit metadata')
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('Unsupported evaluation context schema')
        if value['metric_version'] != VERSION or value['aggregation'] not in PROFILES:
            raise ValueError('Unsupported metric version/aggregation')
        if not value['dataset_identity'] or not digest_or_none(value['dataset_identity']):
            raise ValueError('Explicit dataset identity required')
        if not isinstance(value['split'], str) or not value['split'].strip():
            raise ValueError('Explicit dataset split required')
        files = value['files']
        if (not isinstance(files, list) or not files or any(not isinstance(name, str) or not name.strip() for name in files)
                or len(set(files)) != len(files)):
            raise ValueError('Explicit complete file population required')
        FeatureSpec.from_dict(value['feature_contract'])
        smoothing = value['smoothing']
        if (not isinstance(smoothing, dict) or set(smoothing) != {'enabled', 'size'}
                or type(smoothing['enabled']) is not bool or type(smoothing['size']) is not int or smoothing['size'] < 1):
            raise ValueError('Explicit smoothing settings required')
        plugins = value['post_processors']
        if not isinstance(plugins, list):
            raise ValueError('Explicit post-processing configuration required')
        for plugin in plugins:
            if (not isinstance(plugin, dict) or set(plugin) != {'name', 'parameters'}
                    or not isinstance(plugin['name'], str) or not plugin['name'].strip()
                    or not isinstance(plugin['parameters'], dict)):
                raise ValueError('Invalid post-processing configuration')
        fit, decision = value['fit'], value['decision']
        if (not isinstance(fit, dict) or set(fit) != {'mode', 'population', 'training_identity'}
                or not digest_or_none(fit['training_identity'])):
            raise ValueError('Explicit fit metadata required')
        mode, population = fit['mode'], fit['population']
        if (mode == 'control' and (population is not None or fit['training_identity'] is not None)
                or mode == 'frozen' and (population not in ('train_all', 'train_active') or fit['training_identity'] is None)
                or mode == 'input_adaptive' and (population != 'input_active' or fit['training_identity'] is not None)
                or mode not in ('control', 'frozen', 'input_adaptive')):
            raise ValueError('Fit population does not match its mode')
        if (not isinstance(decision, dict) or set(decision) != {'strategy', 'model_sha256', 'parameters'}
                or not isinstance(decision['strategy'], str) or not decision['strategy'].strip()
                or not digest_or_none(decision['model_sha256'])
                or (mode == 'frozen' and decision['model_sha256'] is None)
                or (mode == 'control' and decision['model_sha256'] is not None)):
            raise ValueError('Explicit strategy/model identity required')
        parameters = decision['parameters']
        threshold = parameters.get('reference_threshold') if isinstance(parameters, dict) else None
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError('Explicit reference threshold required')
        object.__setattr__(self, '_canonical', canonical_json(value))

    def to_dict(self):
        return json.loads(self._canonical)
