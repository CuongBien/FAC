"""Rebuildable run catalogue with read-only adapters for historical layouts."""
from datetime import datetime, timezone
from pathlib import Path
import tarfile

from .legacy import inspect_legacy_run
from .paths import resolve_member
from .store import atomic_write_json, verify_run, writer_lock


class RunRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def scan(self):
        outputs = self.root/'outputs'
        folders = ({p.parent for p in outputs.rglob('frozen_config.json')}
                   | {p.parent for p in outputs.rglob('manifest.json')
                      if 'runs' in p.relative_to(outputs).parts or (p.parent/'config.json').is_file()})
        folders = {folder for folder in folders if '.attempts' not in folder.relative_to(outputs).parts}
        entries = []
        for folder in sorted(folders):
            relative = folder.relative_to(self.root).as_posix()
            try:
                resolve_member(self.root, relative, 'run')
                if (folder/'manifest.json').exists():
                    manifest = verify_run(folder)
                    info = {key: manifest[key] for key in ('run_id', 'format', 'phase', 'experiment_id', 'created_at_utc')}
                    info.update({'metric_version': 'legacy-v1', 'artifact_count': len(manifest['artifacts'])})
                else:
                    info = inspect_legacy_run(folder)
                entries.append({**info, 'path': relative, 'status': 'verified'})
            except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as exc:
                entries.append({'run_id': None, 'path': relative, 'status': 'invalid', 'error': str(exc)})
        return entries

    def verify(self, selector=None):
        entries = self.scan()
        if selector is not None:
            entries = [entry for entry in entries if selector in (entry['run_id'], entry['path'])]
            if len(entries) != 1:
                raise ValueError('Unknown or ambiguous run; use an exact registry path')
        if not entries:
            raise ValueError('No runs found')
        return entries

    def write_index(self, relative_path='outputs/registry.json'):
        entries = self.scan()
        path = resolve_member(self.root, relative_path, 'registry')
        for entry in entries:
            run = resolve_member(self.root, entry['path'], 'run')
            if run in path.parents or path == run:
                raise ValueError('Registry output cannot modify a managed run')
        path.parent.mkdir(parents=True, exist_ok=True)
        with writer_lock(path.parent):
            atomic_write_json(path, {'schema_version': 1,
                'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': entries})
        return path
