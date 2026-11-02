"""Compare two verified series: python -m src.metrics RUN_A KEY_A RUN_B KEY_B."""
import argparse
import json
from pathlib import Path
from .comparison import ALLOWED_DIFFERENCES, read_series, compare_series


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left_run', type=Path)
    parser.add_argument('left_series', help='Exact split/method/strategy')
    parser.add_argument('right_run', type=Path)
    parser.add_argument('right_series', help='Exact split/method/strategy')
    parser.add_argument('--allow', action='append', choices=sorted(ALLOWED_DIFFERENCES), default=[])
    args = parser.parse_args(argv)
    try:
        value = compare_series(read_series(args.left_run, args.left_series),
                               read_series(args.right_run, args.right_series), allow_differences=args.allow)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
