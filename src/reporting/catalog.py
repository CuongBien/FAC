"""Derived Markdown catalogue from verified report manifests, no latest-run alias."""
from pathlib import Path
from urllib.parse import quote
import os

from src.experiments.registry import RunRegistry
from src.experiments.paths import resolve_member
from .bundle import MANIFEST, verify_report


def catalogue(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    # The run registry is still authoritative for scientific results.
    runs = RunRegistry(root).scan()
    bundles = []
    for path in sorted((root/'outputs').rglob(MANIFEST)):
        if any(part.startswith('.') for part in path.relative_to(root).parts):
            continue
        import json
        if json.loads(path.read_bytes()).get('format') != 'fac.report.v1':
            continue
        manifest = verify_report(path.parent)
        bundles.append((path.parent, manifest))
    if root not in output.parents:
        raise ValueError('Catalogue output must stay inside the selected project')
    for run in runs:
        folder = resolve_member(root, run['path'])
        if output == folder or folder in output.parents:
            raise ValueError('Catalogue cannot modify a managed run')
    if any(output == folder or folder in output.parents for folder, _ in bundles):
        raise ValueError('Catalogue cannot modify a report bundle')
    lines = ['# Danh mục báo cáo', '', '| Run | Trạng thái | Nguồn |', '|---|---|---|']
    for run in runs:
        relative = Path(os.path.relpath(root/run['path'], output.parent)).as_posix()
        lines.append('| '+str(run['run_id'] or '-')+' | '+run['status']+' | [run]('+quote(relative, safe='/._-')+') |')
    lines += ['', '| Report | Run nguồn | Giao thức | Evaluation SHA |', '|---|---|---|---|']
    for folder, manifest in bundles:
        relative = Path(os.path.relpath(folder/'README.md', output.parent)).as_posix()
        source = manifest['source']
        lines.append('| ['+manifest['report_id']+']('+quote(relative, safe='/._-')+') | '+source['run_id']+
                     ' | '+manifest['protocol']+' | `'+source['evaluation_sha256']+'` |')
    return '\n'.join(lines)+'\n'
