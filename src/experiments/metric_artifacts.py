"""I/O boundary for versioned metrics; never computes pitch or refits a model."""
import json
from pathlib import Path

from src.metrics.aggregation import FIELDS
from src.metrics.bundle import evaluation_context, ADAPTIVE
from src.metrics.comparison import MetricSeries
from src.metrics.contracts import metric_contract
from .manifest import sha256
from .paths import resolve_member
from .serialization import json_digest, json_bytes

FILENAME = 'metrics.json'


def metric_artifact(folder, manifest, stage_inputs):
    folder = Path(folder)
    spec = json.loads((folder/'config.json').read_bytes())
    frozen = json.loads((folder/'frozen_config.json').read_bytes())
    evaluation = json.loads((folder/'evaluation.json').read_bytes())
    model_sha = sha256(folder/'frozen_models.json')
    train_identity = manifest['datasets']['train']['identity_sha256']
    datasets = {'train': manifest['datasets']['train'], 'test': stage_inputs['test_dataset']}
    records = []
    for item in evaluation['pipelines']+evaluation['protocols']:
        models = {}
        if spec['protocol'] == 'champion' and item['strategy'] in ADAPTIVE:
            for row in item['files']:
                relative = f"input_models/{item['dataset']}/{item['method']}/{Path(row['file']).stem}.json"
                models[row['file']] = {'path': relative, 'sha256': sha256(resolve_member(folder, relative))}
        input_sha = json_digest({name: value['sha256'] for name, value in models.items()}) if models else None
        dataset = datasets[item['dataset']]
        context = evaluation_context(spec, frozen, item, dataset['identity_sha256'], train_identity,
                                     model_sha, files=sorted(name for name in dataset['files'] if name.lower().endswith('.wav')),
                                     input_model_sha256=input_sha)
        if any(not set(FIELDS) <= row.keys() for row in item['files']):
            raise ValueError('Incomplete metric row in evaluation')
        rows = [{'file': row['file'], **{key: row[key] for key in FIELDS}} for row in item['files']]
        series = MetricSeries(context, rows).to_dict()
        if series['aggregate'] != item['metrics']:
            raise ValueError('Evaluation aggregate differs from its declared metric contract')
        records.append({**{key: item[key] for key in ('dataset', 'method', 'strategy')},
                        **series, 'input_model_artifacts': models})
    return {'schema_version': 1, 'contract': metric_contract(),
            'spec_sha256': json_digest(spec),
            'config_sha256': sha256(folder/'frozen_config.json'), 'models_sha256': model_sha,
            'evaluation_sha256': sha256(folder/'evaluation.json'), 'series': records}


def write_metric_artifact(folder, manifest, stage_inputs):
    path = Path(folder)/FILENAME
    with path.open('xb') as stream:
        stream.write(json_bytes(metric_artifact(folder, manifest, stage_inputs)))


def verify_metric_artifact(folder, manifest):
    value = json.loads((Path(folder)/FILENAME).read_bytes())
    expected = metric_artifact(folder, manifest, manifest['completed_stages']['evaluate']['inputs'])
    if json_bytes(value) != json_bytes(expected):
        raise ValueError('Metric context/values do not match the pinned evaluation')
