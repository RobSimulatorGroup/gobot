import pytest
from gobot.rl.experiment_identity import make_run_identity, validate_transfer, freeze_protocol


def identity(**changes):
    config = dict(policy_contract={'obs': 99, 'joints': ['a','b']}, physics={'version': '3.11'},
                  data={'split': 'coverage40'}, algorithm={'horizon': 32}, software={'commit': 'a'}, seed=42)
    return make_run_identity(**(config | changes))


def test_changed_physics_requires_explicit_transfer():
    before, after = identity(), identity(physics={'version': '3.17'})
    with pytest.raises(ValueError, match='resume requires'):
        validate_transfer(before, after, 'resume')
    assert validate_transfer(before, after, 'eval-import')['changed_fields'] == ['physics']
    assert validate_transfer(before, after, 'fork')['mode'] == 'fork'
    with pytest.raises(ValueError, match='protocol changed'):
        validate_transfer(before, identity(policy_contract={'obs': 96, 'joints': ['b','a']}), 'fork')


def test_protocol_and_identity_cannot_be_silently_rewritten(tmp_path):
    path = tmp_path/'protocol.json'
    assert freeze_protocol(path, {'threshold': .02}) == freeze_protocol(path, {'threshold': .02})
    with pytest.raises(ValueError, match='frozen'):
        freeze_protocol(path, {'threshold': .2})
    tampered = identity(); tampered['seed'] = 43
    with pytest.raises(ValueError, match='digest mismatch'):
        validate_transfer(tampered, identity(), 'resume')
