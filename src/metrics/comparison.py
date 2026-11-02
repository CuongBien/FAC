"""Paired file comparison with mandatory context and explicit study differences."""
from dataclasses import InitVar, dataclass, field
import json
import math

from .aggregation import FIELDS, aggregate_rows
from .contracts import EvaluationContext, canonical_json

DIFFERENCE_FIELDS = ('feature_contract', 'post_processors', 'smoothing', 'fit')
ALLOWED_DIFFERENCES = frozenset(DIFFERENCE_FIELDS)


@dataclass(frozen=True)
class MetricSeries:
    context: EvaluationContext
    rows: InitVar[list]
    _canonical_rows: str = field(init=False, repr=False)

    def __post_init__(self, rows):
        if not isinstance(self.context, EvaluationContext):
            raise ValueError('Metric series requires an explicit evaluation context')
        values = json.loads(canonical_json(list(rows)))
        if not values:
            raise ValueError('Metric series requires its full non-empty file population')
        names = []
        for row in values:
            if not isinstance(row, dict) or not isinstance(row.get('file'), str) or not row['file'].strip():
                raise ValueError('Each metric row requires a file ID')
            names.append(row['file'])
            for key in FIELDS:
                if key not in row:
                    raise ValueError('Missing metric field: '+key)
                value = row[key]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value < 0):
                    raise ValueError('Metric values must be finite non-negative numbers or null: '+key)
        if len(set(names)) != len(names):
            raise ValueError('Duplicate file IDs in metric population')
        if set(names) != set(self.context.to_dict()['files']):
            raise ValueError('Metric file population does not match the declared full dataset')
        object.__setattr__(self, '_canonical_rows', canonical_json(values))

    def to_dict(self):
        rows = json.loads(self._canonical_rows)
        return {'context': self.context.to_dict(), 'rows': rows,
                'aggregate': aggregate_rows(rows, self.context.to_dict()['aggregation'])}


def _difference(left, right):
    if left is None or right is None:
        status = 'missing_both' if left is None and right is None else ('missing_left' if left is None else 'missing_right')
        return {'left': left, 'right': right, 'delta': None, 'status': status}
    return {'left': left, 'right': right, 'delta': right-left, 'status': 'available'}


def compare_series(left, right, *, allow_differences=()):
    """Delta is right minus left; negative error deltas are improvements.

    Scientific ablations can explicitly allow feature/post/fit differences;
    dataset, split, metric version, aggregation and file population never mix.
    Allowances are returned with both values, never treated as equivalent.
    """
    allowed = set(allow_differences)
    if not allowed <= ALLOWED_DIFFERENCES:
        raise ValueError('Unsupported comparison allowance')
    if not isinstance(left, MetricSeries) or not isinstance(right, MetricSeries):
        raise ValueError('Comparison requires two typed metric series')
    a, b = left.to_dict(), right.to_dict()
    ac, bc = a['context'], b['context']
    for name in ('dataset_identity', 'split', 'metric_version', 'aggregation'):
        if ac[name] != bc[name]:
            raise ValueError('Incompatible comparison '+name)
    differences = {}
    for name in DIFFERENCE_FIELDS:
        if ac[name] != bc[name]:
            if name not in allowed:
                raise ValueError('Incompatible comparison '+name+'; declare the study difference explicitly')
            differences[name] = {'left': ac[name], 'right': bc[name]}
    a_rows, b_rows = {row['file']: row for row in a['rows']}, {row['file']: row for row in b['rows']}
    if a_rows.keys() != b_rows.keys():
        raise ValueError('Comparison file populations differ; no inner join or file dropping is allowed')
    return {'schema_version': 1, 'metric_version': ac['metric_version'], 'aggregation': ac['aggregation'],
            'dataset_identity': ac['dataset_identity'], 'split': ac['split'],
            'left_context': ac, 'right_context': bc, 'declared_differences': differences,
            'files': [{'file': name, 'metrics': {key: _difference(a_rows[name][key], b_rows[name][key]) for key in FIELDS}}
                      for name in a_rows],
            'aggregate': {key: _difference(a['aggregate'][key], b['aggregate'][key]) for key in FIELDS}}


def read_series(folder, selector):
    """Read one exact split/method/strategy from a verified native run.

    Historical results without explicit contexts are deliberately not guessed.
    A reviewed adapter can construct MetricSeries with declared metadata.
    """
    from pathlib import Path
    from src.experiments.store import verify_run
    folder = Path(folder)
    manifest = verify_run(folder)
    if 'metrics.json' not in manifest['artifacts']:
        raise ValueError('This run has no versioned metric context; supply an explicit reviewed adapter')
    key = selector.split('/')
    if len(key) != 3 or not all(key):
        raise ValueError('Use an exact split/method/strategy selector')
    value = json.loads((folder/'metrics.json').read_bytes())
    matches = [item for item in value['series'] if [item[name] for name in ('dataset', 'method', 'strategy')] == key]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous metric series')
    item = matches[0]
    return MetricSeries(EvaluationContext(item['context']), item['rows'])
