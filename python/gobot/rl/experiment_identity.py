"""Immutable experiment contracts and explicit checkpoint transfer modes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping


def canonical_digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return 'sha256:' + digest.hexdigest()


def freeze_protocol(path, protocol):
    """A protocol version cannot be rewritten after observing its evaluation."""
    path = Path(path)
    digest = canonical_digest(protocol)
    if path.exists():
        if canonical_digest(json.loads(path.read_text())) != digest:
            raise ValueError('protocol is already frozen; choose a new explicit version')
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects competing preparation commands.
    with path.open('x') as stream:
        stream.write(json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return digest


def make_run_identity(*, policy_contract, physics, data, algorithm, software, seed, parent=None):
    value = {'schema_version': 1, 'policy_contract': policy_contract, 'physics': physics,
             'data': data, 'algorithm': algorithm, 'software': software, 'seed': int(seed),
             'parent': parent}
    return {**value, 'digest': canonical_digest(value)}


def validate_transfer(saved: Mapping, expected: Mapping, mode: str):
    if mode not in ('resume', 'eval-import', 'fork'):
        raise ValueError('checkpoint transfer mode must be resume, eval-import or fork')
    for identity in (saved, expected):
        if identity.get('schema_version') != 1:
            raise ValueError('unknown experiment identity schema')
        payload = {key: value for key, value in identity.items() if key != 'digest'}
        if canonical_digest(payload) != identity.get('digest'):
            raise ValueError('experiment identity digest mismatch')
    if saved['policy_contract'] != expected['policy_contract']:
        raise ValueError('policy protocol changed; an explicit observation/action converter or distillation is required')
    if mode == 'resume' and saved['digest'] != expected['digest']:
        raise ValueError('resume requires the same experiment identity; use an explicit fork')
    return {'mode': mode, 'source': saved['digest'], 'target': expected['digest'],
            'changed_fields': [key for key in saved if key != 'digest' and saved[key] != expected.get(key)]}
