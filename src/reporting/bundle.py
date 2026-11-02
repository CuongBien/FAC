"""Atomic, self-contained report revisions, independent of experiment lifecycle."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from src.experiments.manifest import canonical_source_bytes, sha256
from src.experiments.paths import resolve_member
from src.experiments.serialization import json_bytes, json_digest
from src.experiments.snapshots import write_source_snapshot, verify_source_snapshot
from src.experiments.transactions import run_lock
from .source import load_source
from .view import ReportView

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'fac.report.v1'
MANIFEST = 'report_manifest.json'


def renderer_record():
    # Include the reader/verification helpers executed by this revision, not
    # just the templates. No current scientific source is substituted for the
    # independently pinned training snapshot.
    support = ('src/__init__.py', 'src/experiments/__init__.py', 'src/experiments/legacy.py',
               'src/experiments/manifest.py', 'src/experiments/paths.py', 'src/experiments/serialization.py',
               'src/experiments/registry.py',
               'src/experiments/snapshots.py', 'src/experiments/spec.py', 'src/experiments/store.py',
               'src/experiments/transactions.py', 'src/experiments/metric_artifacts.py',
               'src/pipeline/__init__.py', 'src/pipeline/contracts.py')
    files = sorted([*Path(__file__).parent.glob('*.py'), *(ROOT/name for name in support),
                    *(ROOT/'src/metrics').glob('*.py')])
    raw = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
    import hashlib
    canonical = {path.relative_to(ROOT).as_posix(): hashlib.sha256(canonical_source_bytes(path.read_bytes())).hexdigest()
                 for path in files}
    return {'version': 'reporting-v1', 'source_sha256': raw, 'canonical_sha256': json_digest(canonical),
            'runtime': {'python': sys.version.split()[0], **{name: version(name) for name in ('numpy', 'matplotlib')}}}


def _identity(view, renderer):
    return {'view_sha256': json_digest(view.to_dict()),
            'renderer_sha256': renderer['canonical_sha256'], 'runtime': renderer['runtime']}


def _files(folder):
    return {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def create_report(source, output=None):
    """Identical inputs reuse a verified revision; changed renderer creates a new ID."""
    source = Path(source).resolve()
    view, blobs = load_source(source)
    renderer = renderer_record()
    identity = _identity(view, renderer)
    report_id = 'report-'+json_digest(identity)[:24]
    output = Path(output).resolve() if output else source/'reports'/report_id
    # Never publish a report into the scientific artifact namespace.
    if output == source or (source in output.parents and output.parent != source/'reports'):
        raise ValueError('Reports inside a run must use a direct reports/<revision> directory')
    def check():
        after_view, after_blobs = load_source(source)
        if after_view != view or after_blobs != blobs:
            raise ValueError('Report inputs changed during rendering')
    return _create(view, blobs, renderer, output, check)


def rebuild_report(bundle, output):
    """Render entirely from the portable input closure, with no original run."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError('Use a separate revision when rebuilding a report')
    manifest = verify_report(bundle)
    view = ReportView(json.loads((bundle/'view.json').read_bytes()))
    blobs = {name[len('inputs/'):]: resolve_member(bundle, name).read_bytes()
             for name in manifest['artifacts'] if name.startswith('inputs/')}
    def check():
        if verify_report(bundle) != manifest:
            raise ValueError('Report inputs changed during rendering')
    return _create(view, blobs, renderer_record(), output, check)


def _create(view, blobs, renderer, output, check):
    identity = _identity(view, renderer)
    report_id = 'report-'+json_digest(identity)[:24]
    output.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(output.parent):
        if output.exists():
            previous = verify_report(output)
            if previous['identity'] != identity:
                raise FileExistsError('Report inputs changed; use a new report revision directory')
            return output
        work = output.with_name('.'+output.name+'.'+uuid4().hex+'.tmp')
        work.mkdir(exist_ok=False)
        for name, content in blobs.items():
            destination = resolve_member(work, 'inputs/'+name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        (work/'view.json').write_bytes(json_bytes(view.to_dict()))
        write_source_snapshot(ROOT, work/'renderer_source.tar.gz',
                              {'base_commit': 'unavailable', 'source_sha256': renderer['source_sha256']})
        if view.to_dict()['protocol'] == 'champion':
            from .champion import render
        else:
            from .stage4 import render
        render(view, work)
        _export_csv(view, work)
        check()
        if renderer_record() != renderer:
            raise ValueError('Renderer changed during rendering')
        manifest = {'schema_version': 1, 'format': FORMAT, 'report_id': report_id,
                    'protocol': view.to_dict()['protocol'], 'identity': identity, 'renderer': renderer,
                    'source': view.to_dict()['source'], 'artifacts': _files(work)}
        (work/MANIFEST).write_bytes(json_bytes(manifest))
        verify_report(work)
        os.rename(work, output)  # Publish only the complete, verified directory.
    return output


def _export_csv(view, folder):
    import csv
    from src.metrics.formatting import csv_value
    data = view.to_dict()
    fields = ('dataset', 'method', 'strategy', 'final_score', 'run_id', 'evaluation_sha256', 'metric_version')
    with (folder/'tables.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in data['evaluation']['pipelines']+data['evaluation']['protocols']:
            writer.writerow({**{key: row[key] for key in fields[:3]},
                             'final_score': csv_value(row['metrics']['final_score']),
                             **{key: data['source'][key] for key in fields[4:]}})


def verify_report(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/MANIFEST).read_bytes())
    if manifest.get('format') != FORMAT or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported report bundle')
    artifacts = manifest['artifacts']
    if not {'view.json', 'README.md', 'tables.csv', 'renderer_source.tar.gz',
            'inputs/evaluation.json', 'inputs/frozen_config.json', 'inputs/frozen_models.json'} <= artifacts.keys():
        raise ValueError('Incomplete report bundle')
    actual_names = set(_files(folder))-{MANIFEST}
    if actual_names != set(artifacts):
        raise ValueError('Report artifact population differs from its manifest')
    for name, digest in artifacts.items():
        if sha256(resolve_member(folder, name)) != digest:
            raise ValueError('Report artifact checksum mismatch: '+name)
    view = ReportView(json.loads((folder/'view.json').read_bytes()))
    data = view.to_dict()
    if (_identity(view, manifest['renderer']) != manifest['identity']
            or manifest['report_id'] != 'report-'+json_digest(manifest['identity'])[:24]
            or data['source'] != manifest['source'] or data['protocol'] != manifest['protocol']):
        raise ValueError('Report identity mismatch')
    for key, name in (('frozen', 'frozen_config.json'), ('models', 'frozen_models.json'), ('evaluation', 'evaluation.json')):
        if data[key] != json.loads((folder/'inputs'/name).read_bytes()):
            raise ValueError('Report view differs from its input: '+name)
    for name, value in data['extras'].items():
        if value != json.loads(resolve_member(folder/'inputs', name).read_bytes()):
            raise ValueError('Report supplemental input mismatch')
    for key, name in (('config_sha256', 'frozen_config.json'), ('models_sha256', 'frozen_models.json'),
                      ('evaluation_sha256', 'evaluation.json'), ('metrics_sha256', 'metrics.json')):
        if data['source'][key] is not None and sha256(folder/'inputs'/name) != data['source'][key]:
            raise ValueError('Report source link mismatch: '+name)
    inputs = {name[len('inputs/'):]: digest for name, digest in artifacts.items() if name.startswith('inputs/')}
    if json_digest(inputs) != data['source']['inputs_sha256']:
        raise ValueError('Report input closure mismatch')
    verify_source_snapshot(folder/'renderer_source.tar.gz', manifest['renderer'])
    verify_source_snapshot(folder/'inputs/training_source_snapshot.tar.gz', data['frozen']['provenance'])
    validate_links(folder)
    return manifest


def validate_links(folder):
    """Every local Markdown link in the portable report resolves inside the bundle."""
    import re
    from urllib.parse import unquote
    folder = Path(folder).resolve()
    documents = [(folder/'README.md').read_text(encoding='utf-8')]
    if (folder/'sections.json').exists():
        documents += list(json.loads((folder/'sections.json').read_bytes()).values())
    for text in documents:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'):
                continue
            name = unquote(link.split('#', 1)[0].strip('<>'))
            if not resolve_member(folder, name, 'report link').is_file():
                raise ValueError('Broken report link: '+name)
