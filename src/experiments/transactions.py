"""Phase publication with an OS writer lock and a recoverable commit journal."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .manifest import sha256
from .paths import resolve_member
from .serialization import json_digest
from .store import atomic_write_json, verify_run, STAGE_FILES

PHASES = {'prepare': ('prepared', 'prepared'), 'train': ('prepared', 'trained'),
          'evaluate': ('trained', 'evaluated'), 'report': ('evaluated', 'reported')}
JOURNAL = '.transaction.json'
RESERVED = {'manifest.json', JOURNAL, '.writer.lock', '.run.lock', '.attempts'}


def _read_journal(path):
    value = json.loads(Path(path).read_bytes())
    if (not isinstance(value, dict) or value.get('stage') not in PHASES
            or not isinstance(value.get('inputs'), dict) or not isinstance(value.get('work'), str)
            or not isinstance(value.get('artifacts'), dict)):
        raise ValueError('Invalid phase journal')
    if not STAGE_FILES[value['stage']] <= value['artifacts'].keys():
        raise ValueError('Incomplete phase journal')
    for name, record in value['artifacts'].items():
        if (not isinstance(name, str) or not isinstance(record, dict)
                or not isinstance(record.get('sha256'), str)
                or not re.fullmatch('[a-f0-9]{64}', record['sha256']) or record.get('role') != value['stage']):
            raise ValueError('Invalid journal artifact')
    return value


@contextmanager
def run_lock(folder):
    """The OS releases ownership on process death; never infer stale PIDs."""
    path = Path(folder)/'.run.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise FileExistsError('Another writer owns this run') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _remove_attempt(folder, work):
    folder, work = Path(folder).resolve(), Path(work).resolve()
    attempts = folder/'.attempts'
    if work.parent != attempts or not work.is_dir():
        raise ValueError('Invalid phase work directory')
    # Only an explicitly checked, generated attempt under this run is removed.
    shutil.rmtree(work)


def _publish(folder, manifest, journal):
    stage = journal['stage']
    if (stage not in PHASES or not isinstance(journal['artifacts'], dict)
            or not STAGE_FILES[stage] <= journal['artifacts'].keys()
            or not isinstance(journal['inputs'], dict)):
        raise ValueError('Invalid phase journal')
    work = resolve_member(folder, journal['work'], 'phase work')
    if work.parent != Path(folder).resolve()/'.attempts':
        raise ValueError('Invalid phase work directory')
    for name, record in journal['artifacts'].items():
        if any(part in RESERVED for part in Path(name).parts):
            raise ValueError('Reserved phase artifact')
        target = resolve_member(folder, name)
        if name in manifest['artifacts']:
            raise ValueError('Phase journal attempts to replace a committed artifact')
        if target.exists():
            if sha256(target) != record['sha256']:
                raise ValueError('Conflicting interrupted artifact: '+name)
        else:
            source = resolve_member(work, name)
            if sha256(source) != record['sha256']:
                raise ValueError('Interrupted artifact checksum mismatch: '+name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    manifest['artifacts'].update(journal['artifacts'])
    manifest.setdefault('completed_stages', {})[stage] = {
        'inputs': journal['inputs'], 'artifact_names': sorted(journal['artifacts']),
        'result_sha256': json_digest({'inputs': journal['inputs'], 'artifacts': journal['artifacts']}),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest['phase'] = PHASES[stage][1]
    # Files may already be in place after a crash, but completion is published
    # only after validating the entire proposed manifest and its freeze links.
    verify_run(folder, manifest=manifest)
    atomic_write_json(Path(folder)/'manifest.json', manifest)
    (Path(folder)/JOURNAL).unlink()
    _remove_attempt(folder, work)
    return verify_run(folder)


def execute_stage(folder, stage, inputs, callback, *, resume=False):
    """Resume publication, or restart an uncommitted computation in a new attempt.

    Completed phases are immutable. A failed computation is retained for audit;
    its files are never assumed to be checkpoints or promoted without validation.
    """
    folder = Path(folder).resolve()
    if stage not in PHASES:
        raise ValueError('Unknown run stage')
    with run_lock(folder):
        manifest = verify_run(folder)
        completed = manifest.get('completed_stages', {}).get(stage)
        pending = folder/JOURNAL
        if completed:
            if not resume:
                raise FileExistsError('Stage already completed; use resume to verify it')
            if completed['inputs'] != inputs:
                raise ValueError('Completed stage inputs changed')
            if pending.exists():
                journal = _read_journal(pending)
                if journal['stage'] != stage or journal['inputs'] != inputs:
                    raise ValueError('Different stage publication is pending')
                if any(manifest['artifacts'].get(name) != record for name, record in journal['artifacts'].items()):
                    raise ValueError('Committed journal mismatch')
                pending.unlink()
                work = resolve_member(folder, journal['work'])
                if work.exists():
                    _remove_attempt(folder, work)
            return manifest
        if manifest['phase'] != PHASES[stage][0]:
            raise ValueError('Stage requires phase '+PHASES[stage][0])
        if stage == 'train' and 'prepare' not in manifest.get('completed_stages', {}):
            raise ValueError('Run preparation is incomplete')
        if pending.exists():
            if not resume:
                raise FileExistsError('Interrupted publication; use resume')
            journal = _read_journal(pending)
            if journal['stage'] != stage or journal['inputs'] != inputs:
                raise ValueError('Interrupted stage inputs changed')
            return _publish(folder, manifest, journal)
        work = folder/'.attempts'/uuid4().hex
        work.mkdir(parents=True)
        for name in manifest['artifacts']:
            destination = resolve_member(work, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolve_member(folder, name), destination)
        callback(work)
        # Recheck pinned inputs after the callback before publishing any output.
        if verify_run(folder) != manifest:
            raise ValueError('Committed run changed during phase execution')
        for name, record in manifest['artifacts'].items():
            if sha256(resolve_member(work, name)) != record['sha256']:
                raise ValueError('Phase modified a committed input: '+name)
        artifacts = {}
        for path in sorted(work.rglob('*')):
            if not path.is_file():
                continue
            name = path.relative_to(work).as_posix()
            if name not in manifest['artifacts']:
                if any(part in RESERVED for part in Path(name).parts):
                    raise ValueError('Reserved phase artifact')
                if resolve_member(folder, name).exists():
                    raise FileExistsError('Unregistered output already exists: '+name)
                artifacts[name] = {'sha256': sha256(path), 'role': stage}
        if not artifacts:
            raise ValueError('A phase must produce artifacts')
        if not STAGE_FILES[stage] <= artifacts.keys():
            raise ValueError('Phase did not produce all required artifacts: '+stage)
        journal = {'stage': stage, 'inputs': inputs, 'work': work.relative_to(folder).as_posix(),
                   'artifacts': artifacts}
        atomic_write_json(pending, journal)
        return _publish(folder, manifest, journal)
