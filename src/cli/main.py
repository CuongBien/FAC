"""One entrypoint for prediction, experiments, evidence and report revisions."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if sys.platform == 'win32':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(prog='fac', description=__doc__)
    try:
        release = version('fac-speech-f0')
    except PackageNotFoundError:
        release = '0.1.0 (source checkout)'
    parser.add_argument('--version', action='version', version=release)
    parser.add_argument('command', nargs='?', choices=('predict', 'experiment', 'runs', 'report', 'compare', 'config'))
    if not argv or argv[0].startswith('-'):
        parser.parse_args(argv)
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    if args.command == 'predict':
        from .predict import main as command
    elif args.command == 'experiment':
        from src.experiments.__main__ import main as command
        return command(['run', *argv[1:]])
    elif args.command == 'runs':
        from src.experiments.__main__ import main as command
    elif args.command == 'report':
        from src.reporting.__main__ import main as command
    elif args.command == 'compare':
        from src.metrics.__main__ import main as command
    else:
        from .config import main as command
    return command(argv[1:])
