#!/usr/bin/env python3
"""Compatibility entrypoint; implementation is installed as fac predict."""
from src.cli.predict import (get_champion_detectors, get_baseline_detectors, process_single_file, main)

if __name__ == '__main__':
    raise SystemExit(main())
