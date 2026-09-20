"""Plugins package for optional speech processing enhancements."""
from .base import BasePlugin
from .plugin_detector import PluginPitchDetector
from .hysteresis import HysteresisPlugin

__all__ = [
    "BasePlugin",
    "PluginPitchDetector",
    "HysteresisPlugin",
]
