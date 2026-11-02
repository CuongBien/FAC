"""Immutable, JSON-only presentation input shared by tables and figures."""
from dataclasses import InitVar, dataclass, field
import json

from src.metrics.contracts import canonical_json


@dataclass(frozen=True)
class ReportView:
    document: InitVar[dict]
    _json: str = field(init=False, repr=False)

    def __post_init__(self, document):
        value = json.loads(canonical_json(document))
        if value.get('schema_version') != 1 or value.get('protocol') != 'champion':
            raise ValueError('Unsupported report view')
        if not {'source', 'frozen', 'models', 'evaluation', 'extras'} <= value.keys():
            raise ValueError('Incomplete report view')
        identities = set()
        for row in value['evaluation']['pipelines']+value['evaluation']['protocols']:
            identity = tuple(row[key] for key in ('dataset', 'method', 'strategy'))
            if identity in identities:
                raise ValueError('Duplicate report series')
            identities.add(identity)
        object.__setattr__(self, '_json', canonical_json(value))

    def to_dict(self):
        return json.loads(self._json)
