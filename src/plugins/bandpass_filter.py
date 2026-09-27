"""Backward-compatibility module: Re-export BandpassFilterPlugin from pre_processing package."""
from .pre_processing.bandpass_filter import BandpassFilterPlugin

__all__ = ["BandpassFilterPlugin"]
