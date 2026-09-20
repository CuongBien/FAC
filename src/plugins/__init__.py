"""Plugins package for optional speech processing enhancements."""
from .base import BasePlugin
from .plugin_detector import PluginPitchDetector
from .hysteresis import HysteresisPlugin
from .energy_extension import EnergyExtensionPlugin

__all__ = [
    "BasePlugin",
    "PluginPitchDetector",
    "HysteresisPlugin",
    "EnergyExtensionPlugin",
]
