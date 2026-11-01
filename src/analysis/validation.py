"""Reproducible experiment utilities; no test-dependent selection rules."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from unittest.mock import patch

import numpy as np

from src.analysis.evaluation import mean_metric, evaluate_against_ground_truth


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _source_path(root, name):
    relative = PurePosixPath(name)
    root = Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Invalid source snapshot path: {name}')
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents:
        raise ValueError(f'Source escapes repository: {name}')
    return path


def write_source_snapshot(root, path, provenance, overwrite=False):
    """Save exact recorded bytes, using the recorded Git revision if necessary.

    Reconstruct LF/CRLF only when the resulting bytes match the recorded SHA.
    Never substitute today's modified source for a historical training source.
    """
    root, path = Path(root), Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Source snapshot already exists: {path}')
    sources = {}
    for name, expected in provenance['source_sha256'].items():
        source = _source_path(root, name)
        candidates = [source.read_bytes()] if source.is_file() else []
        if not any(hashlib.sha256(value).hexdigest() == expected for value in candidates):
            blob = subprocess.check_output(
                ['git', 'show', f"{provenance['base_commit']}:{name}"], cwd=root)
            lf = blob.replace(b'\r\n', b'\n')
            candidates.extend((blob, lf, lf.replace(b'\n', b'\r\n')))
        match = next((value for value in candidates if hashlib.sha256(value).hexdigest() == expected), None)
        if match is None:
            raise ValueError(f'Cannot reconstruct recorded source bytes: {name}')
        sources[name] = match
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as archive:
        for name, value in sorted(sources.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            archive.addfile(entry, io.BytesIO(value))
    return verify_source_snapshot(path, provenance)


def verify_source_snapshot(path, provenance):
    """Check archive bytes without extracting any member onto the filesystem."""
    expected = provenance['source_sha256']
    with tarfile.open(path, 'r:gz') as archive:
        members = [entry for entry in archive.getmembers() if entry.isfile()]
        if len(members) != len(expected) or {entry.name for entry in members} != set(expected):
            raise ValueError('Source snapshot member list differs from training provenance')
        for entry in members:
            if hashlib.sha256(archive.extractfile(entry).read()).hexdigest() != expected[entry.name]:
                raise ValueError(f'Source snapshot checksum mismatch: {entry.name}')
    return len(expected)


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


def aggregate(rows):
    keys = ('abs_error_mean', 'mape_f0', 'abs_error_std', 'mape_std', 'abs_error_num', 'mape_num', 'final_score')
    return {key: mean_metric((row.get(key) for row in rows), digits=None) for key in keys}


def evaluate_result(result, wav_path):
    f0 = np.asarray(result['f0_contour'])
    labels = np.asarray(result['labels'])
    voiced = labels == 'v'
    if not (len(f0) == len(labels) == len(result['frame_times'])):
        raise AssertionError('Frame alignment mismatch')
    if not np.all(np.isfinite(f0)) or np.any(f0[~voiced] != 0):
        raise AssertionError('Non-finite F0 or pitch outside voiced labels')
    if np.any((f0[voiced] < 70) | (f0[voiced] > 400)) or int(voiced.sum()) != result['num_voiced']:
        raise AssertionError('Pitch range/count mismatch')
    return {'file': Path(wav_path).name, **evaluate_against_ground_truth(result, str(Path(wav_path).with_suffix('.lab')))}


def selection_key(trial):
    score = trial['metrics']['final_score']
    if score is None or not np.isfinite(score):
        return (float('inf'),)
    spec = trial['spec']
    worst = max(row['final_score'] for row in trial['files'])
    return (score, worst, len(spec['plugins']), abs(spec['frame_duration_ms']-25),
            tuple(spec['plugins']), spec['threshold'])


def select_trial(trials):
    valid = [trial for trial in trials if trial['metrics']['final_score'] is not None
             and np.isfinite(trial['metrics']['final_score'])]
    if not valid:
        raise ValueError('No complete train scores are available for selection')
    return min(valid, key=selection_key)


@contextmanager
def pitch_cache():
    """Cache deterministic frame calculations only, with exact input/config keys.

    Detector decisions, medians, extension and Viterbi DP are still run normally.
    The patch is confined to the benchmark context and is restored on exit.
    """
    import src.core.pitch_detector as core
    import src.plugins.plugin_detector as plugin
    from src.plugins.post_processing.viterbi_tracking import ViterbiTrackingPlugin
    cache = {}
    stats = {'hits': 0, 'misses': 0}

    def wrapper(original, method=False):
        def call(*args, **kwargs):
            owner, frame = (args[0], args[1]) if method else (None, args[0])
            rest = args[2:] if method else args[1:]
            values = np.ascontiguousarray(frame)
            owner_key = (owner.f0_min, owner.f0_max, owner.max_candidates) if method else ()
            key = (original, values.shape, values.dtype.str, hashlib.sha256(values.tobytes()).digest(),
                   rest, tuple(sorted(kwargs.items())), owner_key)
            if key in cache:
                stats['hits'] += 1
            else:
                stats['misses'] += 1
                cache[key] = original(*args, **kwargs)
            return deepcopy(cache[key])
        return call

    with ExitStack() as stack:
        for name in ('find_f0_acf', 'find_f0_amdf', 'find_f0_cepstrum'):
            wrapped = wrapper(getattr(core, name))
            stack.enter_context(patch.object(core, name, wrapped))
            stack.enter_context(patch.object(plugin, name, wrapped))
        for name in ('_extract_acf_candidates', '_extract_amdf_candidates', '_extract_cepstrum_candidates'):
            stack.enter_context(patch.object(ViterbiTrackingPlugin, name, wrapper(getattr(ViterbiTrackingPlugin, name), True)))
        yield stats
