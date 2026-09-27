"""Plugins package for optional speech processing enhancements."""
from .base import BasePlugin
from .plugin_detector import PluginPitchDetector
from .hysteresis import HysteresisPlugin
from .energy_extension import EnergyExtensionPlugin
from .bandpass_filter import BandpassFilterPlugin
from .center_clipping import CenterClippingPlugin

__all__ = [
    "BasePlugin",
    "PluginPitchDetector",
    "HysteresisPlugin",
    "EnergyExtensionPlugin",
    "BandpassFilterPlugin",
    "CenterClippingPlugin",
]
