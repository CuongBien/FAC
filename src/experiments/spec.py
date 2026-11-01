"""Immutable JSON-backed experiment contracts, independent of detector defaults."""
from dataclasses import InitVar, dataclass, field
import json
import math
from .serialization import json_bytes, json_digest


@dataclass(frozen=True)
class ExperimentSpec:
    document: InitVar[dict]
    _canonical: bytes = field(init=False, repr=False)

    def __post_init__(self, document):
        value = json.loads(json_bytes(document))
        if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
                or value['schema_version'] != 1 or not isinstance(value.get('name'), str) or not value['name'].strip()):
            raise ValueError('Experiment spec requires schema_version=1 and a non-empty name')
        pipelines = value.get('pipelines')
        if not isinstance(pipelines, dict) or not pipelines:
            raise ValueError('Experiment spec requires explicit pipelines')
        for method, pipeline in pipelines.items():
            if not method or not isinstance(pipeline, dict):
                raise ValueError('Invalid pipeline')
            for key in ('frame_duration_ms', 'hop_duration_ms', 'f0_min', 'f0_max', 'ste_silence_ratio', 'reference_threshold'):
                number = pipeline.get(key)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                    raise ValueError(f'Pipeline requires a finite {key}')
            if (pipeline['frame_duration_ms'] <= 0 or pipeline['hop_duration_ms'] <= 0
                    or not 0 < pipeline['f0_min'] < pipeline['f0_max'] or pipeline['ste_silence_ratio'] < 0):
                raise ValueError('Invalid frame, pitch range or silence ratio')
            if (type(pipeline.get('use_median_filter')) is not bool
                    or type(pipeline.get('median_size')) is not int or pipeline['median_size'] < 1):
                raise ValueError('Median settings must be explicit')
            for stage in ('pre_processors', 'post_processors'):
                if not isinstance(pipeline.get(stage), list):
                    raise ValueError(f'Pipeline requires explicit {stage}')
                for plugin in pipeline[stage]:
                    if (not isinstance(plugin, dict) or not isinstance(plugin.get('name'), str)
                            or not plugin['name'].strip() or not isinstance(plugin.get('parameters'), dict)):
                        raise ValueError('Plugins require a name and explicit parameters')
        fit = value.get('fit', {})
        if not isinstance(fit, dict) or fit.get('mode') not in ('frozen', 'input_adaptive'):
            raise ValueError('Explicit frozen/input_adaptive fit mode required')
        allowed = ('train_all', 'train_active') if fit['mode'] == 'frozen' else ('input_active',)
        if fit.get('population') not in allowed:
            raise ValueError('Fit population does not match fit mode')
        strategies = fit.get('strategies')
        if (not isinstance(strategies, list) or not strategies
                or any(not isinstance(name, str) or not name for name in strategies)
                or len(set(strategies)) != len(strategies)):
            raise ValueError('Declare unique strategy IDs')
        if value.get('metric_version') != 'legacy-v1':
            raise ValueError('Only the existing legacy-v1 metric contract is supported')
        object.__setattr__(self, '_canonical', json_bytes(value))

    def to_dict(self):
        return json.loads(self._canonical)

    @property
    def sha256(self):
        return json_digest(self.to_dict())
