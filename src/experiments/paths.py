"""Resolve portable artifact names without permitting repository escapes."""
from pathlib import Path, PurePosixPath


def resolve_member(root, name, label='artifact'):
    name = str(name)
    relative = PurePosixPath(name)
    root = Path(root).resolve()
    if not name or relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Invalid {label} path: {name}')
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents:
        raise ValueError(f'{label.capitalize()} escapes root: {name}')
    return path
