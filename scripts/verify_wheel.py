"""Exercise an extracted wheel in an isolated interpreter outside the checkout."""
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root/'dist').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('Build exactly one wheel under dist first')
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        package = work/'package'
        with ZipFile(wheels[0]) as archive:
            archive.extractall(package)
        code = r'''
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import numpy as np
from scipy.io import wavfile
from src.experiments.protocol_specs import default_spec
from src.experiments.runner import RunService
from src.reporting.bundle import verify_report
from src.analysis.ssl_pitch_shift import analyze_signal_ssl_pitch_shift
root = Path(sys.argv[2])
for split in ('train', 'test'):
    folder = root/split
    folder.mkdir()
    t = np.arange(4800)/16000
    signal = .3*np.sin(2*np.pi*180*t)
    wavfile.write(folder/'tone.wav', 16000, (signal*32767).astype(np.int16))
    (folder/'tone.lab').write_text('F0mean 180\nF0std 10\nF0num 20\n')
service = RunService(root/'run')
service.prepare(default_spec('champion'), root/'train')
service.train(root/'train')
service.evaluate(root/'train', root/'test')
verify_report(service.report())
result = analyze_signal_ssl_pitch_shift(signal, 16000)
assert np.any(result['labels'] == 'v')
print('Wheel smoke test passed: train, evaluate, report and SSL outside Git.')
'''
        subprocess.run([sys.executable, '-I', '-B', '-c', code, str(package), str(work)],
                       cwd=work, check=True)


if __name__ == '__main__':
    main()
