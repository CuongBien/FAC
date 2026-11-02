"""Presentation adapters; no numerical metric calculation lives here."""
import re
import numpy as np


def format_metric(value, spec=".2f"):
    """Format numbers normally and preserve the width of N/A table entries."""
    if value is not None and np.isfinite(value):
        return format(value, spec)
    width = re.match(r"([<>^]?)(\d+)", spec)
    return format("N/A", (width[1] or ">") + width[2]) if width else "N/A"


def csv_value(value):
    return "" if value is None else str(value)
