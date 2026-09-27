"""Stage 3: Post-processing enhancement plugins."""
from .energy_extension import EnergyExtensionPlugin
from .viterbi_tracking import ViterbiTrackingPlugin

__all__ = ["EnergyExtensionPlugin", "ViterbiTrackingPlugin"]
