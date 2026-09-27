"""Plugins package for speech processing enhancements, organized by pipeline stages:
- pre_processing: BandpassFilterPlugin, CenterClippingPlugin
- decision: HysteresisPlugin
- post_processing: EnergyExtensionPlugin, ViterbiTrackingPlugin
"""
from .base import (
    BasePlugin,
    PluginStage,
    PreProcessingPlugin,
    DecisionPlugin,
    PostProcessingPlugin,
)
from .plugin_detector import PluginPitchDetector
from .pre_processing import BandpassFilterPlugin, CenterClippingPlugin
from .decision import HysteresisPlugin
from .post_processing import EnergyExtensionPlugin, ViterbiTrackingPlugin

__all__ = [
    # Base abstractions & stages
    "BasePlugin",
    "PluginStage",
    "PreProcessingPlugin",
    "DecisionPlugin",
    "PostProcessingPlugin",
    # Coordinator
    "PluginPitchDetector",
    # Stage 1: Pre-processing
    "BandpassFilterPlugin",
    "CenterClippingPlugin",
    # Stage 2: Decision
    "HysteresisPlugin",
    # Stage 3: Post-processing
    "EnergyExtensionPlugin",
    "ViterbiTrackingPlugin",
]
