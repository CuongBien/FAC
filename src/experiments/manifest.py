"""Legacy byte manifests and versioned portable experiment fingerprints."""
import hashlib
from pathlib import Path
import subprocess
import re
from .paths import resolve_member
from .serialization import json_digest

SOURCE_POLICY = 'utf8-lf-v1'
DATA_POLICY = 'raw-sha256-v1'


def canonical_source_bytes(data):
    """Normalize only Unicode line endings; preserve BOM, spaces and all code."""
    return data.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n').encode('utf-8')


def source_fingerprint(root, files):
    records = {}
    for name in sorted(set(files)):
        data = resolve_member(root, name, 'source').read_bytes()
        records[name] = {'raw_sha256': hashlib.sha256(data).hexdigest(),
                         'canonical_sha256': hashlib.sha256(canonical_source_bytes(data)).hexdigest()}
    if not records:
        raise ValueError('A source scope must contain at least one explicit file')
    return {'files': records,
            'raw_sha256': json_digest({name: value['raw_sha256'] for name, value in records.items()}),
            'canonical_sha256': json_digest({name: value['canonical_sha256'] for name, value in records.items()})}


def capture_source(root, scopes):
    """Capture explicitly declared stage dependencies; Git is optional metadata."""
    if not scopes or set(scopes)-{'train', 'predict', 'evaluate', 'report'}:
        raise ValueError('Declare train/predict/evaluate/report source scopes explicitly')
    stages = {stage: source_fingerprint(root, files) for stage, files in sorted(scopes.items())}
    git = None
    try:
        def read(*arguments):
            return subprocess.check_output(['git', *arguments], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        git = {'commit': read('rev-parse', 'HEAD'), 'branch': read('branch', '--show-current'),
               'dirty': bool(read('status', '--porcelain'))}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {'schema_version': 1, 'hash_policy': SOURCE_POLICY, 'stages': stages, 'git': git}


def verify_source_scope(expected, actual, stage):
    validate_source(expected)
    validate_source(actual)
    if stage not in expected['stages'] or stage not in actual['stages']:
        raise ValueError(f'Undeclared source scope: {stage}')
    if expected['stages'][stage]['canonical_sha256'] != actual['stages'][stage]['canonical_sha256']:
        raise ValueError(f'Source changed for {stage}')


def capture_dataset(root, files):
    """Dataset identity uses relative names and raw bytes, never machine paths."""
    records = {}
    for name in sorted(set(files)):
        path = resolve_member(root, name, 'dataset')
        records[name] = {'sha256': sha256(path), 'size': path.stat().st_size}
    if not records:
        raise ValueError('A dataset must contain at least one file')
    return {'schema_version': 1, 'hash_policy': DATA_POLICY, 'files': records,
            'identity_sha256': json_digest(records)}


def validate_source(value):
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('hash_policy') != SOURCE_POLICY
            or not isinstance(value.get('stages'), dict) or not value['stages']):
        raise ValueError('Unsupported source manifest')
    for stage, record in value['stages'].items():
        if (stage not in ('train', 'predict', 'evaluate', 'report') or not isinstance(record, dict)
                or not isinstance(record.get('files'), dict) or not record['files']):
            raise ValueError('Invalid source scope')
        for policy in ('raw', 'canonical'):
            files = {name: item.get(policy+'_sha256') if isinstance(item, dict) else None
                     for name, item in record['files'].items()}
            if any(not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest) for digest in files.values()):
                raise ValueError('Invalid source digest')
            if record[policy+'_sha256'] != json_digest(files):
                raise ValueError(f'Source manifest checksum mismatch: {stage}')


def validate_dataset(value):
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('hash_policy') != DATA_POLICY
            or not isinstance(value.get('files'), dict) or not value['files']):
        raise ValueError('Unsupported dataset manifest')
    for record in value['files'].values():
        if (not isinstance(record, dict) or not isinstance(record.get('sha256'), str)
                or not re.fullmatch(r'[a-f0-9]{64}', record['sha256'])
                or type(record.get('size')) is not int or record['size'] < 0):
            raise ValueError('Invalid dataset file metadata')
    if value['identity_sha256'] != json_digest(value['files']):
        raise ValueError('Dataset manifest checksum mismatch')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def source_manifest(root):
    root = Path(root)
    files = sorted(list((root/'src').rglob('*.py')) + list((root/'scripts').glob('*.py')) + [root/'predict.py'])
    return {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip(),
            'source_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in files}}



def dataset_manifest(folder):
    folder = Path(folder)
    files = sorted(folder.glob('*.wav'))
    if not files:
        raise FileNotFoundError(f"No WAV files: {folder}")
    return {p.name: {'wav_sha256': sha256(p), 'lab_sha256': sha256(p.with_suffix('.lab'))} for p in files}



def verify_dataset(folder, manifest):
    if dataset_manifest(folder) != manifest:
        raise ValueError(f"Dataset has changed since the training freeze: {folder}")
