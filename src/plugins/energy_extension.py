"""Backward-compatibility module: Re-export EnergyExtensionPlugin from post_processing package."""
from .post_processing.energy_extension import EnergyExtensionPlugin

__all__ = ["EnergyExtensionPlugin"]
