"""JSON serialization shared by experiment artifacts."""
import json
import hashlib
from pathlib import Path
import numpy as np


def json_bytes(value):
    """Canonical JSON v1: sorted keys, finite values, UTF-8 and a final LF."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       default=json_default, allow_nan=False)+'\n').encode('utf-8')


def json_digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)



def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)+'\n', encoding='utf-8')
