"""Export packaged, fixed v1 experiment specifications without overriding files."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', choices=('champion', 'stage4'))
    parser.add_argument('--output', type=Path, help='Create this new JSON file; default is stdout')
    args = parser.parse_args(argv)
    from src.experiments.resources import source_path
    root = Path(__file__).resolve().parents[2]
    name = 'configs/experiments/'+('champion_unsupervised' if args.protocol == 'champion' else 'stage4_calibration')+'.json'
    try:
        data = source_path(root, name).read_bytes()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as stream:
                stream.write(data)
            print(args.output)
        else:
            sys.stdout.write(data.decode('utf-8'))
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')
    return 0
