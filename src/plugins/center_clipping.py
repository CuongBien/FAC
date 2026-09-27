"""Backward-compatibility module: Re-export CenterClippingPlugin from pre_processing package."""
from .pre_processing.center_clipping import CenterClippingPlugin

__all__ = ["CenterClippingPlugin"]
