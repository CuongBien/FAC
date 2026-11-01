"""Compatibility readers for immutable historical experiment artifacts."""
import json
from pathlib import Path
from .serialization import write_json
from .manifest import sha256
from .serialization import json_digest
from .snapshots import verify_source_snapshot


def freeze_json(path, data):
    path = Path(path)
    if path.exists() or path.with_suffix('.sha256').exists():
        raise FileExistsError(f"Frozen artifact already exists; use a new output directory: {path}")
    write_json(path, data)
    path.with_suffix('.sha256').write_text(sha256(path)+'\n', encoding='ascii')



def read_frozen(path):
    path = Path(path)
    expected = path.with_suffix('.sha256').read_text(encoding='ascii').strip()
    if sha256(path) != expected:
        raise ValueError(f"Frozen artifact checksum mismatch: {path}")
    return json.loads(path.read_text(encoding='utf-8'))



def read_evaluation_bundle(folder):
    """Reject mixing evaluations with independently valid but different freezes."""
    folder = Path(folder)
    frozen = read_frozen(folder/'frozen_config.json')
    models = read_frozen(folder/'frozen_models.json')
    result = json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    if frozen['train_manifest'] != models['train_manifest']:
        raise ValueError('Frozen models/config use different training data')
    for name in ('config', 'models'):
        if result.get(name+'_sha256') != sha256(folder/f'frozen_{name}.json'):
            raise ValueError(f'Evaluation uses a different frozen {name}')
    return frozen, models, result


def pinned_bytes(path, expected, text=False):
    """Recover original text EOL bytes only if they match the recorded raw SHA.

    This read-only adapter never changes a legacy file or redefines its hash.
    """
    import hashlib
    raw = Path(path).read_bytes()
    candidates = [raw]
    if text:
        lf = raw.replace(b'\r\n', b'\n')
        candidates.extend((lf, lf.replace(b'\n', b'\r\n')))
    for value in candidates:
        if hashlib.sha256(value).hexdigest() == expected:
            return value, value != raw
    raise ValueError(f'Legacy artifact checksum mismatch: {path}')


def inspect_legacy_run(folder):
    """Verify archived Stage 4/Champion evidence without importing algorithms."""
    folder = Path(folder)
    reconstructed = []
    documents, digests = {}, {}
    for key in ('config', 'models'):
        name = f'frozen_{key}.json'
        expected = (folder/f'frozen_{key}.sha256').read_text(encoding='ascii').strip()
        data, restored = pinned_bytes(folder/name, expected, text=True)
        documents[key], digests[key] = json.loads(data), expected
        if restored:
            reconstructed.append(name)
    frozen, models = documents['config'], documents['models']
    if frozen['train_manifest'] != models['train_manifest']:
        raise ValueError('Frozen models/config use different training data')
    if 'selected' in frozen:
        kind = 'legacy.stage4.v1'
    elif 'experiment' in frozen:
        kind = 'legacy.champion-unsupervised.v1'
    else:
        raise ValueError('Unsupported legacy run format')
    evaluation = json.loads((folder/'evaluation.json').read_bytes())
    if any(evaluation.get(key+'_sha256') != digest for key, digest in digests.items()):
        raise ValueError('Evaluation does not match frozen config/models')
    count = verify_source_snapshot(folder/'training_source_snapshot.tar.gz', frozen['provenance'])
    runtime = json.loads((folder/'runtime.json').read_bytes())
    if runtime['snapshot_sha256'] != sha256(folder/'training_source_snapshot.tar.gz'):
        raise ValueError('Runtime/source snapshot mismatch')
    # A report may pin evaluation/exports, but its editable publication targets
    # are outside the run and are deliberately not treated as immutable inputs.
    for name in ('report_manifest.json', 'report_generation.json'):
        path = folder/name
        if not path.exists():
            continue
        report = json.loads(path.read_bytes())
        mapping = {'evaluation_sha256': 'evaluation.json', 'per_file_sha256': 'per_file.csv',
                   'report_sha256': 'README.md', 'figure_sha256': 'figures/score_matrix.png'}
        for key, target in mapping.items():
            if key in report:
                _, restored = pinned_bytes(folder/target, report[key], text=not target.endswith('.png'))
                if restored and target not in reconstructed:
                    reconstructed.append(target)
    pipeline_cases = sum(len(row['files']) for row in evaluation['pipelines'])
    protocol_cases = sum(len(row['files']) for row in evaluation['protocols'])
    identity = json_digest({'format': kind, **digests})
    return {'run_id': kind.split('.')[1]+'-'+identity[:16], 'format': kind,
            'phase': 'evaluated', 'created_at_utc': frozen.get('frozen_at_utc'),
            'config_sha256': digests['config'], 'models_sha256': digests['models'],
            'metric_version': 'legacy-v1', 'pipeline_cases': pipeline_cases,
            'protocol_cases': protocol_cases, 'source_files': count,
            'eol_reconstructed_in_memory': sorted(reconstructed)}
