"""Backward-compatibility module: Re-export HysteresisPlugin from decision package."""
from .decision.hysteresis import HysteresisPlugin

__all__ = ["HysteresisPlugin"]
