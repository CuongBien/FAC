"""Render a Stage 4 report revision; document publication is an explicit step."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.bundle import create_report
from src.reporting.publish import (CHAPTERS, README_MARKERS, REPORT_MARKERS, CHAPTER_MARKERS,
                                   replace_generated_block)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/stage4_validation', help='Recorded run to read')
    parser.add_argument('--report-dir', type=Path, help='New immutable report destination; defaults to RUN/reports/<revision>')
    args = parser.parse_args(argv)
    folder = create_report(args.output_dir, args.report_dir)
    print('REPORT:', folder/'README.md')
    return folder


if __name__ == '__main__':
    main()
