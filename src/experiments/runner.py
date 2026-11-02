"""Shared prepare/freeze/evaluate lifecycle for the two fixed v1 protocols."""
import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from .dependencies import capture_runtime_source
from .manifest import capture_dataset, dataset_manifest, sha256, verify_source_scope, canonical_source_bytes
from .protocol_specs import default_spec, validate_protocol
from .serialization import write_json
from .snapshots import write_source_snapshot, verify_source_snapshot
from .spec import ExperimentSpec
from .store import RunStore, verify_run
from .transactions import execute_stage
from .resources import source_path

ROOT = Path(__file__).resolve().parents[2]


def dataset_record(folder):
    names = dataset_manifest(folder)
    files = sorted([name for wav in names for name in (wav, str(Path(wav).with_suffix('.lab')))])
    return capture_dataset(folder, files)


def runtime_record(root):
    return {'python': sys.version.split()[0],
            **{name: version(name) for name in ('numpy', 'scipy', 'matplotlib')},
            'lock_sha256': sha256(source_path(root, 'uv.lock')),
            'lock_canonical_sha256': hashlib.sha256(canonical_source_bytes(source_path(root, 'uv.lock').read_bytes())).hexdigest()}


def runtime_identity(record):
    # Preserve exact lock bytes as provenance without rejecting LF/CRLF checkouts.
    return {key: value for key, value in record.items() if key not in ('lock_sha256', 'snapshot_sha256')}


@dataclass
class RunContext:
    root: Path
    manifest: dict

    @property
    def provenance(self):
        source = self.manifest['source']
        git = source.get('git') or {}
        return {'base_commit': git.get('commit', 'unavailable'), 'branch': git.get('branch', 'snapshot'),
                'source_sha256': {name: record['raw_sha256'] for stage in source['stages'].values()
                                  for name, record in stage['files'].items()}}

    def check_source(self, stages):
        actual = capture_runtime_source(self.root, self.manifest['source'])
        for stage in stages:
            verify_source_scope(self.manifest['source'], actual, stage)
        if self.root != ROOT:
            # A relocated source root may differ in EOL, but must describe the
            # code this interpreter actually executes, not an arbitrary copy.
            loaded = capture_runtime_source(ROOT, self.manifest['source'])
            for stage in stages:
                verify_source_scope(actual, loaded, stage)
        return actual


class RunService:
    def __init__(self, folder, *, source_root=ROOT):
        self.folder = Path(folder).resolve()
        self.root = Path(source_root).resolve()

    def prepare(self, spec, train_dir, *, resume=False):
        protocol = validate_protocol(spec)
        dataset = dataset_record(train_dir)  # Never inspects a test directory.
        if not (self.folder/'manifest.json').exists():
            if self.folder.exists():
                if any(self.folder.iterdir()):
                    raise FileExistsError('Use a new run directory')
                self.folder.rmdir()  # Caller-selected empty output only.
            source = capture_runtime_source(self.root)
            if self.root != ROOT:
                RunContext(self.root, {'source': source}).check_source(('train', 'predict', 'evaluate'))
            RunStore(self.folder.parent).create(spec, {'train': dataset}, source, self.folder.name)
        manifest = verify_run(self.folder)
        if manifest['spec_sha256'] != spec.sha256 or manifest['datasets'] != {'train': dataset}:
            raise ValueError('Prepared spec or training data changed')
        context = RunContext(self.root, manifest)
        context.check_source(('train', 'predict', 'evaluate'))
        runtime = runtime_record(self.root)
        inputs = {'spec_sha256': spec.sha256, 'train_identity': dataset['identity_sha256'],
                  'runtime': runtime_identity(runtime), 'protocol': protocol}
        def create(work):
            snapshot = work/'training_source_snapshot.tar.gz'
            write_source_snapshot(self.root, snapshot, context.provenance)
            write_json(work/'runtime.json', {**runtime, 'snapshot_sha256': sha256(snapshot)})
            context.check_source(('train', 'predict', 'evaluate'))
            if dataset_record(train_dir) != dataset:
                raise ValueError('Training data changed during preparation')
        return execute_stage(self.folder, 'prepare', inputs, create, resume=resume)

    def _inputs(self, train_dir, stage, test_dir=None):
        manifest = verify_run(self.folder)
        prerequisite = {'train': 'prepare', 'evaluate': 'train'}[stage]
        if prerequisite not in manifest.get('completed_stages', {}):
            raise ValueError('Stage requires completed '+prerequisite)
        spec = ExperimentSpec(json.loads((self.folder/'config.json').read_bytes()))
        protocol = validate_protocol(spec)
        context = RunContext(self.root, manifest)
        stages = ('train', 'predict', 'evaluate') if stage != 'train' else ('train', 'predict')
        current_source = context.check_source(stages)
        train = dataset_record(train_dir)
        if train != manifest['datasets']['train']:
            raise ValueError('Training data changed since preparation')
        recorded = json.loads((self.folder/'runtime.json').read_bytes())
        current = runtime_record(self.root)
        if runtime_identity(recorded) != runtime_identity(current):
            raise ValueError('Runtime or dependency lock changed')
        verify_source_snapshot(self.folder/'training_source_snapshot.tar.gz', context.provenance)
        inputs = {'spec_sha256': spec.sha256, 'train_identity': train['identity_sha256'],
                  'source': {name: current_source['stages'][name]['canonical_sha256'] for name in stages},
                  'runtime': runtime_identity(current), 'protocol': protocol}
        if stage != 'train':
            inputs['freeze'] = {name: manifest['artifacts'][name]['sha256']
                                for name in ('frozen_config.json', 'frozen_models.json')}
        if test_dir is not None:
            inputs['test_dataset'] = dataset_record(test_dir)
        return inputs, context

    def train(self, train_dir, *, resume=False):
        inputs, context = self._inputs(train_dir, 'train')
        backend = _backend(inputs['protocol'])
        def compute(work):
            backend.train(SimpleNamespace(train_dir=str(train_dir), output_dir=str(work)), context=context)
            if self._inputs(train_dir, 'train')[0] != inputs:
                raise ValueError('Training inputs changed during execution')
        return execute_stage(self.folder, 'train', inputs, compute, resume=resume)

    def evaluate(self, train_dir, test_dir, *, resume=False):
        inputs, context = self._inputs(train_dir, 'evaluate', test_dir)
        backend = _backend(inputs['protocol'])
        def compute(work):
            backend.evaluate(SimpleNamespace(train_dir=str(train_dir), test_dir=str(test_dir),
                                              output_dir=str(work)), context=context)
            from .metric_artifacts import write_metric_artifact
            write_metric_artifact(work, context.manifest, inputs)
            if self._inputs(train_dir, 'evaluate', test_dir)[0] != inputs:
                raise ValueError('Evaluation inputs changed during execution')
        return execute_stage(self.folder, 'evaluate', inputs, compute, resume=resume)

    def report(self, train_dir=None, test_dir=None, *, resume=False, output=None):
        # Reports consume the verified evaluation, not the original data or the
        # current scientific runtime. Revisions never change the run manifest.
        from src.reporting.bundle import create_report
        return create_report(self.folder, output)


def _backend(protocol):
    if protocol == 'champion':
        from . import champion
        return champion
    if protocol == 'stage4':
        from . import calibration
        return calibration
    raise ValueError('Unknown experiment protocol')


def run_phase(protocol, phase, args):
    """Compatibility service for script 17/21 argument objects."""
    service = RunService(args.output_dir, source_root=getattr(args, 'root', ROOT))
    resume = getattr(args, 'resume', False)
    if protocol not in ('champion', 'stage4') or phase not in ('prepare', 'train', 'evaluate', 'report'):
        raise ValueError('Unknown run protocol/phase')
    if (service.folder/'manifest.json').exists():
        stored_spec = ExperimentSpec(json.loads((service.folder/'config.json').read_bytes()))
        if stored_spec.to_dict().get('protocol') != protocol:
            raise ValueError('Stored run uses a different protocol')
    if phase in ('prepare', 'train'):
        config_path = getattr(args, 'spec', None)
        if config_path:
            spec = ExperimentSpec(json.loads(Path(config_path).read_bytes()))
        elif (service.folder/'config.json').exists():
            spec = ExperimentSpec(json.loads((service.folder/'config.json').read_bytes()))
        else:
            spec = default_spec(protocol)
        if spec.to_dict().get('protocol') != protocol:
            raise ValueError('Spec does not match the requested runner')
        prepared = (service.folder/'manifest.json').exists()
        incomplete = prepared and 'prepare' not in verify_run(service.folder).get('completed_stages', {})
        if phase == 'prepare' or not prepared or incomplete:
            service.prepare(spec, args.train_dir, resume=resume)
        else:
            manifest = verify_run(service.folder)
            if spec.sha256 != manifest['spec_sha256']:
                raise ValueError('Configuration changed after preparation')
        return service.train(args.train_dir, resume=resume) if phase == 'train' else verify_run(service.folder)
    if not (service.folder/'manifest.json').exists():
        # Existing historical runs keep their original readers/guards. New runs
        # always use the manifest lifecycle; never upgrade old evidence in place.
        if phase == 'report':
            from src.reporting.bundle import create_report
            return create_report(service.folder, getattr(args, 'report_dir', None))
        return _backend(protocol).evaluate(args)
    if phase == 'evaluate':
        return service.evaluate(args.train_dir, args.test_dir, resume=resume)
    if phase == 'report':
        return service.report(output=getattr(args, 'report_dir', None), resume=resume)
    raise ValueError('Unknown run phase')


def legacy_cli(protocol, argv=None):
    parser = argparse.ArgumentParser(description='Versioned '+protocol+' experiment runner')
    phases = ['prepare', 'train', 'evaluate', 'report']
    parser.add_argument('phase', choices=phases)
    parser.add_argument('--train-dir', default=str(ROOT/'TinHieuHuanLuyen'))
    parser.add_argument('--test-dir', default=str(ROOT/'TinHieuKiemThu'))
    parser.add_argument('--output-dir', default=str(ROOT/'outputs'/('stage4_validation' if protocol == 'stage4'
                                                                else 'champion_unsupervised_2026-11-01')))
    parser.add_argument('--spec', type=Path, help='Explicit fixed v1 protocol JSON; unsupported overrides are rejected')
    parser.add_argument('--report-dir', type=Path, help='Separate immutable report revision destination')
    parser.add_argument('--resume', action='store_true', help='Verify completed stages or recover an interrupted phase')
    args = parser.parse_args(argv)
    try:
        result = run_phase(protocol, args.phase, args)
        if args.phase == 'report':
            print('REPORT:', result/'README.md')
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')
