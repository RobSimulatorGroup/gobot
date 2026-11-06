"""A green completion banner requires a complete, nonterminated replay."""
import json
from pathlib import Path

import pytest

from examples.g1.reproduction.record_tracking import replay_outcome


def test_success_requires_every_reference_frame():
    expected = {'frames': 504, 'failed': False, 'fail_step': -1}
    status = replay_outcome(expected, 'success', terminated=False, done=True, steps=504)
    assert status['success'] and status['replay_failure_step'] == -1
    assert status['completed_steps'] == 504


@pytest.mark.parametrize('terminated,done,steps', [
    (False, True, 503), (False, False, 504), (True, True, 504), (False, True, 505),
])
def test_truncated_or_failed_walk_cannot_be_labeled_success(terminated, done, steps):
    with pytest.raises(ValueError):
        replay_outcome({'frames': 504, 'failed': False, 'fail_step': -1}, 'success',
                       terminated=terminated, done=done, steps=steps)


def test_failure_replay_retains_different_failure_time():
    status = replay_outcome({'frames': 4346, 'failed': True, 'fail_step': 429}, 'failure',
                            terminated=True, done=True, steps=426)
    assert status['failure_reproduced'] and not status['same_failure_step']
    assert status['replay_failure_step'] == 426 and not status['success']


def test_success_cannot_be_labeled_as_reproduced_failure():
    with pytest.raises(ValueError, match='reproduce failure'):
        replay_outcome({'frames': 504, 'failed': True, 'fail_step': 250}, 'failure',
                       terminated=False, done=True, steps=504)


def test_changed_benchmark_label_is_rejected():
    with pytest.raises(ValueError, match='benchmark outcome'):
        replay_outcome({'frames': 504, 'failed': True, 'fail_step': 250}, 'success',
                       terminated=False, done=True, steps=504)


def test_capacity_recipe_preserves_all_other_training_parameters():
    root = Path(__file__).resolve().parents[2] / 'examples/g1/train'
    baseline = json.loads((root / 'reproduced_bones.json').read_text())
    candidate = json.loads((root / 'reproduced_bones_nj500.json').read_text())
    assert candidate.pop('njmax') == 500
    assert candidate == baseline
