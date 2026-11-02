"""Per-run cache of pure calculations; never patches module/class functions."""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib

import numpy as np

_legacy_cache = ContextVar('fac_pitch_cache', default=None)


class PitchCache(Mapping):
    def __init__(self):
        self._values = {}
        self._stats = {'hits': 0, 'misses': 0}

    def __getitem__(self, key):
        return self._stats[key]

    def __iter__(self):
        return iter(self._stats)

    def __len__(self):
        return len(self._stats)

    def call(self, function, frame, *args, namespace=(), **kwargs):
        values = np.ascontiguousarray(frame)
        # Bound methods compare by owner identity. Include numerical owner state
        # explicitly, and use the underlying function to share pure calculations.
        identity = getattr(function, '__func__', function)
        key = (identity, namespace, values.shape, values.dtype.str,
               hashlib.sha256(values.tobytes()).digest(), args, tuple(sorted(kwargs.items())))
        if key in self._values:
            self._stats['hits'] += 1
        else:
            self._stats['misses'] += 1
            self._values[key] = deepcopy(function(frame, *args, **kwargs))
        return deepcopy(self._values[key])


def cached_call(cache, function, frame, *args, namespace=(), **kwargs):
    selected = cache if cache is not None else _legacy_cache.get()
    if selected is None:
        return function(frame, *args, **kwargs)
    return selected.call(function, frame, *args, namespace=namespace, **kwargs)


@contextmanager
def pitch_cache():
    """Compatibility context for old callers; new APIs accept cache explicitly.

    ContextVar keeps nested and concurrent contexts separate. Decisions,
    smoothing and frame decisions are never cached.
    """
    cache = PitchCache()
    token = _legacy_cache.set(cache)
    try:
        yield cache
    finally:
        _legacy_cache.reset(token)
