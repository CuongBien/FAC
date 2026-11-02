"""Inspect experiments: python -m src.experiments [--root ROOT] list|verify|index."""
import argparse
import json
from pathlib import Path

from .registry import RunRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list').add_argument('--json', action='store_true')
    verify = commands.add_parser('verify')
    verify.add_argument('run', nargs='?', help='Exact run ID or path from the registry; omit for all')
    index = commands.add_parser('index')
    index.add_argument('--output', default='outputs/registry.json', help='Root-relative derived index path')
    run = commands.add_parser('run', help='Execute a fixed versioned protocol')
    run.add_argument('protocol', choices=('champion', 'stage4'))
    run.add_argument('phase', choices=('prepare', 'train', 'evaluate', 'report'))
    run.add_argument('--train-dir')
    run.add_argument('--test-dir')
    run.add_argument('--output-dir', required=True)
    run.add_argument('--spec', type=Path)
    run.add_argument('--report-dir', type=Path)
    run.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    registry = RunRegistry(args.root)
    try:
        if args.command == 'run':
            if args.phase != 'report' and not args.train_dir:
                raise ValueError('This phase requires --train-dir')
            if args.phase == 'evaluate' and not args.test_dir:
                raise ValueError('Evaluation requires --test-dir')
            from .runner import run_phase
            result = run_phase(args.protocol, args.phase, args)
            if args.phase == 'report':
                print('REPORT:', result/'README.md')
            return 0
        if args.command == 'index':
            print(registry.write_index(args.output))
            return 0
        entries = registry.verify(args.run) if args.command == 'verify' else registry.scan()
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    if args.command == 'list' and not args.json:
        for entry in entries:
            print(f"{entry['status']:8} {entry['run_id'] or '-':42} {entry['path']}")
    else:
        print(json.dumps({'schema_version': 1, 'runs': entries}, ensure_ascii=True, indent=2))
    return int(args.command == 'verify' and any(entry['status'] != 'verified' for entry in entries))


if __name__ == '__main__':
    raise SystemExit(main())
