"""Processing plugins used by the frozen Champion configurations."""
from .base import BasePlugin, PluginStage, PreProcessingPlugin, DecisionPlugin, PostProcessingPlugin
from .plugin_detector import PluginPitchDetector
from .pre_processing import BandpassFilterPlugin, CenterClippingPlugin
from .post_processing import EnergyExtensionPlugin, ViterbiTrackingPlugin
