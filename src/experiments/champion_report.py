"""Compatibility entrypoint for immutable report revisions."""
from src.reporting.bundle import create_report


def report(args):
    folder = create_report(args.output_dir, getattr(args, 'report_dir', None))
    print('REPORT:', folder/'README.md')
    return folder
