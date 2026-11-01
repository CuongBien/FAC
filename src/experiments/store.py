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


def verify_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_bytes())
    if (not isinstance(manifest, dict) or manifest.get('format') != FORMAT
            or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1):
        raise ValueError('Unsupported run manifest')
    if manifest.get('phase') != 'prepared':
        raise ValueError('Unsupported run phase; this store currently prepares runs only')
    if not isinstance(manifest.get('artifacts'), dict) or 'config.json' not in manifest['artifacts']:
        raise ValueError('Run manifest is missing config.json')
    for name, record in manifest['artifacts'].items():
        if not isinstance(record, dict) or not isinstance(record.get('sha256'), str):
            raise ValueError(f'Invalid artifact metadata: {name}')
        if sha256(resolve_member(folder, name)) != record['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    spec = ExperimentSpec(json.loads((folder/'config.json').read_bytes()))
    if manifest['spec_sha256'] != spec.sha256:
        raise ValueError('Spec identity mismatch')
    if experiment_identity(spec, manifest['datasets'], manifest['source']) != manifest['experiment_id']:
        raise ValueError('Experiment identity mismatch')
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
        if path.name in ('manifest.json', '.writer.lock'):
            raise ValueError('Reserved artifact name')
        with writer_lock(folder):
            manifest = verify_run(folder)
            if path.exists() or relative_path in manifest['artifacts']:
                raise FileExistsError(f'Artifact already exists: {relative_path}')
            atomic_write_json(path, value)
            manifest['artifacts'][relative_path] = {'sha256': sha256(path), 'role': role}
            atomic_write_json(folder/'manifest.json', manifest)
        return path

    def verify(self, run_id):
        return verify_run(resolve_member(self.root, run_id, 'run'))
