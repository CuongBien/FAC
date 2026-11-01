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
    args = parser.parse_args(argv)
    registry = RunRegistry(args.root)
    try:
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
