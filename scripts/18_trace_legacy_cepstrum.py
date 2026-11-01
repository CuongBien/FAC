"""Replay the artifact's Git revision in an isolated temporary Python package."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.serialization import write_json
from src.experiments.manifest import sha256


DRIVER = '''import importlib.util, json, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('legacy14', 'scripts/14_compare_acf_amdf_cepstrum.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
factories, _ = m.build_bt2_detectors()
name = next(name for name in factories if 'Cepstrum Champion' in name)
factory = factories[name]
import src.plugins.plugin_detector as plugin_module
calls = {'acf': 0, 'cepstrum': 0}
def spy(original, method):
    def call(*args, **kwargs):
        calls[method] += 1
        return original(*args, **kwargs)
    return call
for method in ['acf', 'cepstrum']:
    function = 'find_f0_' + method
    if hasattr(plugin_module, function):
        setattr(plugin_module, function, spy(getattr(plugin_module, function), method))
out = {'train_clipping_hypotheses': {}, 'detector_function_calls': calls}
for ratio in [.2, .3, .4, .5]:
    def configured(ratio=ratio):
        detector = factory()
        for plugin in detector.plugins:
            if hasattr(plugin, 'clipping_ratio'): plugin.clipping_ratio = ratio
        return detector
    row = m.evaluate_suite({name: configured}, str(root/'TinHieuHuanLuyen'))[0]
    out['train_clipping_hypotheses'][str(ratio)] = row
out['test_recorded_configuration'] = m.evaluate_suite({name: factory}, str(root/'TinHieuKiemThu'))[0]
output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='caaf751')
    parser.add_argument('--output-dir', default=str(ROOT/'outputs/stage4_validation'))
    parser.add_argument('--patch-detector-from', help='Override only plugin_detector.py from this Git revision')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', args.revision], cwd=ROOT, text=True).strip()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(['git', 'archive', revision, 'src', 'scripts/14_compare_acf_amdf_cepstrum.py'], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='legacy-replay-', dir=output) as directory:
        checkout = Path(directory).resolve()
        # Verify the exact cleanup target before TemporaryDirectory removes it.
        if os.path.commonpath([str(checkout), str(output)]) != str(output):
            raise ValueError('Temporary checkout escaped its output root')
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for entry in tar.getmembers():
                if not entry.isfile() or not entry.name.endswith('.py'):
                    continue
                target = (checkout/entry.name).resolve()
                if os.path.commonpath([str(checkout), str(target)]) != str(checkout):
                    raise ValueError('Invalid archive path')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(entry).read())
        driver = checkout/'replay.py'
        if args.patch_detector_from:
            detector_source = subprocess.check_output(
                ['git', 'show', args.patch_detector_from+':src/plugins/plugin_detector.py'], cwd=ROOT)
            (checkout/'src/plugins/plugin_detector.py').write_bytes(detector_source)
        driver.write_text(DRIVER, encoding='utf-8')
        subprocess.run([sys.executable, '-B', str(driver), str(ROOT), str(output/'legacy_replay.json')],
                       cwd=checkout, check=True)
    replay = json.loads((output/'legacy_replay.json').read_text(encoding='utf-8'))
    historical = {}
    for split, filename in [('train', 'huanluyen_compare_acf_amdf_cepstrum.json'),
                             ('test', 'compare_acf_amdf_cepstrum.json')]:
        path = ROOT/'outputs/reports'/filename
        historical[split] = {'artifact_sha256': sha256(path),
                             'record': next(row for row in json.loads(path.read_text(encoding='utf-8'))
                                            if 'Cepstrum Champion' in row['name'])}
    result = {'source_revision': revision, 'plugin_detector_override': args.patch_detector_from,
              'historical_artifacts': historical, 'replay': replay,
              'interpretation': 'Compare exact source replay and bounded clipping hypotheses; no test-dependent calibration.'}
    write_json(output/'cepstrum_trace.json', result)
    print('Legacy replay complete:', output/'cepstrum_trace.json', flush=True)


if __name__ == '__main__':
    main()
