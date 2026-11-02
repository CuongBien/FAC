"""Read verified run artifacts without consulting WAVs, LABs or current models."""
import hashlib
import json
from pathlib import Path

from src.experiments.legacy import inspect_legacy_run, pinned_bytes
from src.experiments.manifest import sha256
from src.experiments.paths import resolve_member
from src.experiments.serialization import json_digest
from src.experiments.store import verify_run
from .view import ReportView


def load_source(folder):
    folder = Path(folder).resolve()
    native = (folder/'manifest.json').is_file()
    manifest = verify_run(folder) if native else None
    if native and 'evaluate' not in manifest.get('completed_stages', {}):
        raise ValueError('Reports require a completed evaluation')
    info = manifest if native else inspect_legacy_run(folder)
    blobs = {}
    names = ['frozen_config.json', 'frozen_models.json', 'evaluation.json',
             'frozen_config.sha256', 'frozen_models.sha256',
             'training_source_snapshot.tar.gz', 'runtime.json']
    for optional in ('metrics.json', 'per_file.csv', 'evaluation_pipelines_per_file.csv',
                     'evaluation_protocols_per_file.csv', 'config.json'):
        if (folder/optional).is_file() and (not native or optional in manifest['artifacts']):
            names.append(optional)
    for name in names:
        if native:
            if name not in manifest['artifacts']:
                raise ValueError('Report input is not pinned: '+name)
            expected = manifest['artifacts'][name]['sha256']
        elif name.startswith('frozen_') and name.endswith('.json'):
            expected = (folder/Path(name).with_suffix('.sha256')).read_text(encoding='ascii').strip()
        else:
            expected = sha256(folder/name)
        blobs[name] = pinned_bytes(folder/name, expected, text=not name.endswith('.gz'))[0]
    frozen, models, evaluation = [json.loads(blobs[name]) for name in
                                ('frozen_config.json', 'frozen_models.json', 'evaluation.json')]
    protocol = 'champion'
    extras = {}
    source = {'run_id': info['run_id'], 'format': info['format'], 'metric_version': 'legacy-v1',
              'config_sha256': evaluation['config_sha256'], 'models_sha256': evaluation['models_sha256'],
              'evaluation_sha256': sha256(folder/'evaluation.json'),
              'metrics_sha256': sha256(folder/'metrics.json') if 'metrics.json' in blobs else None,
              'context_kind': 'verified-sidecar' if 'metrics.json' in blobs else 'recorded-legacy',
              'inputs_sha256': json_digest({name: hashlib.sha256(data).hexdigest()
                                           for name, data in blobs.items()})}
    view = ReportView({'schema_version': 1, 'protocol': protocol, 'source': source,
                       'frozen': frozen, 'models': models, 'evaluation': evaluation, 'extras': extras})
    return view, blobs
