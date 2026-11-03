"""Reference evaluation must distinguish timeout truncation from full tracking."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.locomotion.supermimic_reference import evaluation_summary


def test_full_clip_coverage_is_required_even_without_falls():
    motion = SimpleNamespace(names=['walk1', 'run1', 'sprint1'], clip_len=np.array([1001, 2001, 501]))
    result = {'per_clip': {
        'walk1': {'norm_return': .4, 'steps': 500, 'failed': False},
        'run1': {'norm_return': .8, 'steps': 2000, 'failed': False},
        'sprint1': {'norm_return': .01, 'steps': 10, 'failed': True},
    }}
    actual = evaluation_summary(result, motion)
    assert actual['n_full_clips'] == 1
    assert actual['frame_coverage'] == 2510 / 3500
    assert actual['gait_groups']['walk']['full_clips'] == 0
    assert actual['gait_groups']['run']['full_clips'] == 1
    assert actual['gait_groups']['sprint']['failed'] == 1
    assert not actual['admitted_velocity_terrain_policy']
