"""Render, verify, preview publication, or derive a report catalogue."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    render = commands.add_parser('render')
    render.add_argument('run', type=Path)
    render.add_argument('--output', type=Path)
    rebuild = commands.add_parser('rebuild', help='Render from a portable report bundle; original run is unnecessary')
    rebuild.add_argument('bundle', type=Path)
    rebuild.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('bundle', type=Path)
    publish = commands.add_parser('publish')
    publish.add_argument('bundle', type=Path)
    publish.add_argument('--root', type=Path, required=True)
    publish.add_argument('--apply', action='store_true', help='Apply the validated marker-only update; default is preview')
    index = commands.add_parser('index')
    index.add_argument('--root', type=Path, required=True)
    index.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'render':
            from .bundle import create_report
            print(create_report(args.run, args.output))
        elif args.command == 'rebuild':
            from .bundle import rebuild_report
            print(rebuild_report(args.bundle, args.output))
        elif args.command == 'verify':
            from .bundle import verify_report
            print(json.dumps(verify_report(args.bundle), ensure_ascii=True, indent=2))
        elif args.command == 'publish':
            from .publish import preview, apply
            plan = preview(args.bundle, args.root)
            if args.apply:
                print(json.dumps(apply(plan), ensure_ascii=True, indent=2))
            else:
                print(plan.diff())
        else:
            from .catalog import catalogue
            # Derive and validate first; only this explicitly named file is written.
            text = catalogue(args.root, args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding='utf-8')
            print(args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, str(exc)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
