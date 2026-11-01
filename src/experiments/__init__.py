"""Experiment specifications, evidence storage and run discovery.

Scientific algorithms remain in core/analysis; this package does not fit or
execute a detector when inspecting an experiment.
"""
from .spec import ExperimentSpec
from .store import RunStore
from .registry import RunRegistry

__all__ = ['ExperimentSpec', 'RunStore', 'RunRegistry']
