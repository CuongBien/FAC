"""Wheel-only support resources, generated from actual build inputs, not copies in Git."""
import hashlib
import json
from pathlib import Path
from setuptools.command.build_py import build_py


RESOURCE_FILES = ('README.md', 'pyproject.toml', 'uv.lock',
                  'scripts/21_benchmark_champion_unsupervised.py',
                  'configs/experiments/champion_unsupervised.json')


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        destination = Path(self.build_lib)/'src/_runtime'
        destination.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in RESOURCE_FILES:
            raw = (root/name).read_bytes().decode('utf-8').replace('\r\n', '\n').replace('\r', '\n').encode('utf-8')
            digest = hashlib.sha256(raw).hexdigest()
            relative = 'files/'+digest+'.txt'
            blob = destination/relative
            blob.parent.mkdir(exist_ok=True)
            blob.write_bytes(raw)
            files[name] = {'path': relative, 'sha256': digest}
        (destination/'manifest.json').write_bytes((json.dumps({'schema_version': 1, 'files': files},
                                                            sort_keys=True, separators=(',', ':'))+'\n').encode('utf-8'))
