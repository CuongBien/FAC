"""Explicit marker publication with byte-preserving previews and preflight."""
from dataclasses import dataclass
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, quote
from uuid import uuid4

from src.experiments.paths import resolve_member
from src.experiments.manifest import sha256
from .bundle import MANIFEST, verify_report

CHAMPION_MARKERS = ('<!-- CHAMPION_REPORT_START -->', '<!-- CHAMPION_REPORT_END -->')


def replace_generated_block(source, content, start_marker, end_marker):
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError('Expected exactly one generated block: '+start_marker+' / '+end_marker)
    begin, end = source.index(start_marker), source.index(end_marker)
    if end < begin:
        raise ValueError('Generated block markers are reversed: '+start_marker)
    return source[:begin]+start_marker+'\n'+content.rstrip()+'\n'+end_marker+source[end+len(end_marker):]


def _rebase(content, bundle, parent):
    def replace(match):
        link = match.group(2)
        if '://' in link or link.startswith('#'):
            return match.group(0)
        name, separator, anchor = link.partition('#')
        path = resolve_member(bundle, unquote(name.strip('<>')), 'report link')
        relative = Path(os.path.relpath(path, parent)).as_posix()
        return match.group(1)+quote(relative, safe='/._-')+(separator+anchor if separator else '')+')'
    return re.sub(r'(!?\[[^\]]*\]\()([^)]+)\)', replace, content)


@dataclass(frozen=True)
class DocumentChange:
    name: str
    before: bytes
    after: bytes


@dataclass(frozen=True)
class PublishPlan:
    root: Path
    bundle: Path
    manifest_sha256: str
    changes: tuple

    def diff(self):
        return ''.join(''.join(difflib.unified_diff(change.before.decode('utf-8').replace('\r\n', '\n').splitlines(True),
                                                   change.after.decode('utf-8').replace('\r\n', '\n').splitlines(True),
                                                   fromfile=change.name, tofile=change.name)) for change in self.changes)


def preview(bundle, root):
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    manifest = verify_report(bundle)
    sections = json.loads((bundle/'sections.json').read_bytes())
    owned = {'docs/CHAMPION_UNSUPERVISED_RESULTS.md': CHAMPION_MARKERS}
    if set(sections) != set(owned):
        raise ValueError('Publication targets differ from the renderer contract')
    changes = []
    for name, markers in owned.items():
        path = resolve_member(root, name, 'publication target')
        if path == bundle or bundle in path.parents:
            raise ValueError('Cannot publish into a report bundle')
        before = path.read_bytes()
        text = before.decode('utf-8')  # Preserve BOM and every byte outside the owned block.
        content = _rebase(sections[name], bundle, path.parent)
        # Replace just the byte interval, avoiding newline translation of manual text.
        replace_generated_block(text, '', *markers)
        start, end = markers
        begin, finish = before.index(start.encode()), before.index(end.encode())
        newline = b'\r\n' if b'\r\n' in before[begin:finish] else b'\n'
        replacement = start.encode()+newline+content.rstrip().replace('\r\n', '\n').encode('utf-8').replace(b'\n', newline)+newline+end.encode()
        after = before[:begin]+replacement+before[finish+len(end.encode()):]
        changes.append(DocumentChange(name, before, after))
    return PublishPlan(root, bundle, sha256(bundle/MANIFEST), tuple(changes))


def apply(plan):
    """Reject stale previews, validate every marker before writing, rollback I/O errors.

    Each document replacement is atomic. Process death across several documents
    can leave a partial publication; regenerate the preview and reapply to finish.
    The immutable report bundle remains the authoritative published source.
    """
    from src.experiments.transactions import run_lock
    with run_lock(plan.root):
        return _apply(plan)


def _apply(plan):
    verify_report(plan.bundle)
    if sha256(plan.bundle/MANIFEST) != plan.manifest_sha256:
        raise ValueError('Report changed after publication preview')
    # Rebuild the entire preview to prevent stale inputs or forged plan bytes.
    if preview(plan.bundle, plan.root) != plan:
        raise ValueError('Documents changed after publication preview')
    temporary, completed = {}, []
    try:
        for change in plan.changes:
            if change.before == change.after:
                continue
            path = resolve_member(plan.root, change.name)
            staged = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
            staged.write_bytes(change.after)
            temporary[change.name] = staged
        # Check all originals once more after staging; no document has changed yet.
        if any(resolve_member(plan.root, item.name).read_bytes() != item.before for item in plan.changes):
            raise ValueError('Documents changed while preparing publication')
        for change in plan.changes:
            if change.name in temporary:
                os.replace(temporary[change.name], resolve_member(plan.root, change.name))
                completed.append(change)
    except Exception:
        for change in reversed(completed):
            path = resolve_member(plan.root, change.name)
            rollback = path.with_name('.'+path.name+'.'+uuid4().hex+'.rollback')
            rollback.write_bytes(change.before)
            os.replace(rollback, path)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return {change.name: hashlib.sha256(change.after).hexdigest() for change in plan.changes}
