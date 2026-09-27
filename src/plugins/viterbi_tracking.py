"""Backward-compatibility module: Re-export ViterbiTrackingPlugin from post_processing package."""
from .post_processing.viterbi_tracking import ViterbiTrackingPlugin

__all__ = ["ViterbiTrackingPlugin"]
