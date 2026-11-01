"""Atomic run metadata and append-only artifact storage for new experiments."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from .manifest import sha256, validate_source, validate_dataset
from .paths import resolve_member
from .serialization import json_bytes, json_digest
from .spec import ExperimentSpec

FORMAT = 'fac.run.v1'
STAGE_FILES = {'prepare': {'runtime.json', 'training_source_snapshot.tar.gz'},
               'train': {'frozen_config.json', 'frozen_config.sha256', 'frozen_models.json', 'frozen_models.sha256'},
               'evaluate': {'evaluation.json'}, 'report': {'report_manifest.json', 'README.md'}}


def atomic_write_json(path, value):
    path = Path(path)
    data = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(folder):
    lock = Path(folder)/'.writer.lock'
    with lock.open('x', encoding='ascii') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def experiment_identity(spec, datasets, source):
    validate_source(source)
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError('At least one explicit dataset is required')
    for record in datasets.values():
        validate_dataset(record)
    if not {'train', 'predict'} <= source['stages'].keys():
        raise ValueError('Experiment identity requires train and predict source scopes')
    return json_digest({'spec_sha256': spec.sha256,
                        'datasets': {name: record['identity_sha256'] for name, record in sorted(datasets.items())},
                        'source': {stage: source['stages'][stage]['canonical_sha256'] for stage in ('train', 'predict')}})


def _legacy_dataset(record):
    files = record['files']
    return {name: {'wav_sha256': value['sha256'],
                   'lab_sha256': files[str(Path(name).with_suffix('.lab'))]['sha256']}
            for name, value in files.items() if name.lower().endswith('.wav')}


def verify_run(folder, manifest=None):
    folder = Path(folder)
    if manifest is None:
        manifest = json.loads((folder/'manifest.json').read_bytes())
    if (not isinstance(manifest, dict) or manifest.get('format') != FORMAT
            or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1):
        raise ValueError('Unsupported run manifest')
    if manifest.get('phase') not in ('prepared', 'trained', 'evaluated', 'reported'):
        raise ValueError('Unsupported run phase')
    if not isinstance(manifest.get('artifacts'), dict) or 'config.json' not in manifest['artifacts']:
        raise ValueError('Run manifest is missing config.json')
    for name, record in manifest['artifacts'].items():
        if any(part in ('manifest.json', '.writer.lock', '.run.lock', '.transaction.json', '.attempts')
               for part in Path(name).parts):
            raise ValueError('Reserved run artifact')
        if not isinstance(record, dict) or not isinstance(record.get('sha256'), str):
            raise ValueError(f'Invalid artifact metadata: {name}')
        if sha256(resolve_member(folder, name)) != record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    spec = ExperimentSpec(json.loads((folder/'config.json').read_bytes()))
    if manifest['spec_sha256'] != spec.sha256:
        raise ValueError('Spec identity mismatch')
    if experiment_identity(spec, manifest['datasets'], manifest['source']) != manifest['experiment_id']:
        raise ValueError('Experiment identity mismatch')
    stages = manifest.get('completed_stages', {})
    required = {'prepared': [], 'trained': ['prepare', 'train'],
                'evaluated': ['prepare', 'train', 'evaluate'],
                'reported': ['prepare', 'train', 'evaluate', 'report']}[manifest['phase']]
    if not isinstance(stages, dict) or set(stages)-{'prepare', 'train', 'evaluate', 'report'}:
        raise ValueError('Invalid completed stages')
    if not set(required) <= stages.keys():
        raise ValueError('Run phase is missing completed stages')
    allowed = set(required) | ({'prepare'} if manifest['phase'] == 'prepared' else set())
    if set(stages)-allowed:
        raise ValueError('Run phase/stage mismatch')
    for stage, record in stages.items():
        if not isinstance(record, dict) or not isinstance(record.get('inputs'), dict):
            raise ValueError('Invalid stage metadata')
        names = record.get('artifact_names')
        if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or not STAGE_FILES[stage] <= set(names)
                or not set(names) <= manifest['artifacts'].keys()):
            raise ValueError('Incomplete stage artifacts: '+stage)
        artifacts = {name: manifest['artifacts'][name] for name in names}
        if record.get('result_sha256') != json_digest({'inputs': record['inputs'], 'artifacts': artifacts}):
            raise ValueError('Stage checksum mismatch: '+stage)
        if (record['inputs'].get('spec_sha256') != manifest['spec_sha256']
                or record['inputs'].get('train_identity') != manifest['datasets']['train']['identity_sha256']):
            raise ValueError('Stage inputs do not match the prepared run')
    if 'prepare' in stages:
        runtime = json.loads((folder/'runtime.json').read_bytes())
        if runtime['snapshot_sha256'] != manifest['artifacts']['training_source_snapshot.tar.gz']['sha256']:
            raise ValueError('Runtime/source snapshot mismatch')
    if 'train' in stages:
        from .legacy import read_frozen
        frozen = read_frozen(folder/'frozen_config.json')
        models = read_frozen(folder/'frozen_models.json')
        if frozen['train_manifest'] != models['train_manifest'] or frozen['provenance'] != models['provenance']:
            raise ValueError('Frozen config/models come from different training runs')
        if frozen['train_manifest'] != _legacy_dataset(manifest['datasets']['train']):
            raise ValueError('Frozen training data differs from the prepared run')
        expected_sources = {name: record['raw_sha256'] for stage in manifest['source']['stages'].values()
                            for name, record in stage['files'].items()}
        if frozen['provenance']['source_sha256'] != expected_sources:
            raise ValueError('Frozen source differs from preparation')
    if 'evaluate' in stages:
        from .legacy import read_evaluation_bundle
        _, _, evaluation = read_evaluation_bundle(folder)
        expected = {name: manifest['artifacts'][name]['sha256']
                    for name in ('frozen_config.json', 'frozen_models.json')}
        if stages['evaluate']['inputs'].get('freeze') != expected:
            raise ValueError('Evaluation does not use the pinned freeze')
        if evaluation.get('test_manifest') != _legacy_dataset(stages['evaluate']['inputs']['test_dataset']):
            raise ValueError('Evaluation test data does not match its stage inputs')
    if 'report' in stages:
        report = json.loads((folder/'report_manifest.json').read_bytes())
        if report['evaluation_sha256'] != manifest['artifacts']['evaluation.json']['sha256']:
            raise ValueError('Report does not use the pinned evaluation')
        links = {'config_sha256': 'frozen_config.json', 'models_sha256': 'frozen_models.json',
                 'per_file_sha256': 'per_file.csv', 'report_sha256': 'README.md',
                 'figure_sha256': 'figures/score_matrix.png'}
        if any(report.get(key) != manifest['artifacts'][name]['sha256'] for key, name in links.items()):
            raise ValueError('Report artifact links do not match the run')
    return manifest


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def create(self, spec, datasets, source, run_id=None):
        identity = experiment_identity(spec, datasets, source)
        run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run_id):
            raise ValueError('Invalid run ID')
        folder = resolve_member(self.root, run_id, 'run')
        folder.mkdir(parents=True, exist_ok=False)
        atomic_write_json(folder/'config.json', spec.to_dict())
        atomic_write_json(folder/'manifest.json', {
            'schema_version': 1, 'format': FORMAT, 'run_id': run_id, 'experiment_id': identity,
            'phase': 'prepared', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'spec_sha256': spec.sha256, 'datasets': datasets, 'source': source,
            'artifacts': {'config.json': {'sha256': sha256(folder/'config.json'), 'role': 'config'}},
        })
        return folder

    def add_json(self, run_id, relative_path, value, role):
        relative_path = str(relative_path)
        if not isinstance(role, str) or not role:
            raise ValueError('Artifact role is required')
        folder = resolve_member(self.root, run_id, 'run')
        path = resolve_member(folder, relative_path)
        if any(part in ('manifest.json', '.writer.lock', '.run.lock', '.transaction.json', '.attempts')
               for part in Path(relative_path).parts):
            raise ValueError('Reserved artifact name')
        from .transactions import run_lock
        with run_lock(folder), writer_lock(folder):
            manifest = verify_run(folder)
            if path.exists() or relative_path in manifest['artifacts']:
                raise FileExistsError(f'Artifact already exists: {relative_path}')
            atomic_write_json(path, value)
            manifest['artifacts'][relative_path] = {'sha256': sha256(path), 'role': role}
            atomic_write_json(folder/'manifest.json', manifest)
        return path

    def verify(self, run_id):
        return verify_run(resolve_member(self.root, run_id, 'run'))
