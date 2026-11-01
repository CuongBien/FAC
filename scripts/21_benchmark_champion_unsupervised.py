"""Compatibility entrypoint; implementation lives in src.experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.experiments.runner import run_phase, legacy_cli

def train(args):
    return run_phase('champion', 'train', args)


def evaluate(args):
    return run_phase('champion', 'evaluate', args)


def report(args):
    return run_phase('champion', 'report', args)


def main():
    legacy_cli('champion')


if __name__ == '__main__':
    main()
