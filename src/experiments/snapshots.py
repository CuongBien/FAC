"""Exact recorded source snapshots, independent of scientific evaluation."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
from .paths import resolve_member


def _source_path(root, name):
    return resolve_member(root, name, label='source snapshot')



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
