"""Logical source names shared by checkouts, extracted snapshots and wheels."""
import hashlib
import json
from pathlib import Path

from .paths import resolve_member


def source_path(root, name, *, label='source'):
    """Wheel support files retain the same logical IDs as a source checkout.

    Never fall back to another checkout or silently use a machine-wide file.
    Installed resource blobs are selected and verified by the wheel's index.
    """
    root = Path(root)
    path = resolve_member(root, name, label)
    resources = root/'src/_runtime'
    index = resources/'manifest.json'
    if not index.is_file():
        return path
    manifest = json.loads(index.read_bytes())
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unsupported packaged resource index')
    if name not in manifest['files']:
        return path
    record = manifest['files'][name]
    blob = resolve_member(resources, record['path'], 'packaged source')
    if hashlib.sha256(blob.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Packaged source checksum mismatch: '+name)
    return blob
