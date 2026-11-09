"""Public velocity parameters do not infer morphology from robot or joint names."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gobot.rl.batch import BatchEnvState
from gobot.rl.locomotion.config import NaturalGaitCfg, VelocityIllegalContactCfg, VelocityRewardCfg
from gobot.rl.locomotion.natural_gait import NaturalGaitEnv, desired_contact, gait_weights
from gobot.rl.locomotion.terms import posture_std
from gobot.rl.spec import ObservationSpec, SpecField
from gobot.rl.task_runtime import TaskRuntimeMetadata
from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg


def test_default_parameters_have_no_shape_or_joint_name_conventions():
    contact = VelocityIllegalContactCfg()
    assert contact.thigh_shape_patterns == contact.shank_shape_patterns == contact.trunk_head_shape_patterns == ()
    assert not contact.terminate_on_thigh
    reward = VelocityRewardCfg()
    assert reward.shank_collision == reward.trunk_head_collision == 0.
    names = ('drive_a', 'calf', 'shoulder', 'hinge_17')
    np.testing.assert_allclose(posture_std(names, reward.pose_std_standing), .05)
    np.testing.assert_allclose(posture_std(names, reward.pose_std_walking), .3)
    np.testing.assert_allclose(posture_std(names, reward.pose_std_running), .3)


def test_go1_keeps_explicit_task_parameters():
    cfg = go1_velocity_cfg()
    assert cfg.illegal_contact.terminate_on_thigh
    assert cfg.illegal_contact.thigh_shape_patterns == (r'.*_thigh_collision[123]',)
    assert cfg.rewards.shank_collision == cfg.rewards.trunk_head_collision == -.1
    for key, first, last in [('standing', .05, .1), ('walking', .3, .6), ('running', .3, .6)]:
        values = posture_std(cfg.joint_names, getattr(cfg.rewards, f'pose_std_{key}'))
        np.testing.assert_allclose(values, [last if name.endswith('_calf_joint') else first for name in cfg.joint_names])


@pytest.mark.parametrize('mapping', [{}, {'other': .1}, {'.*': .1, 'joint': .2},
                                     {'joint': 0.}, {'joint': float('nan')}, {'joint': float('inf')}])
def test_posture_mapping_requires_exactly_one_finite_positive_match(mapping):
    with pytest.raises(ValueError, match='finite positive'):
        posture_std(['joint'], mapping)


def test_contact_clock_uses_profile_foot_order_for_any_number_of_feet():
    phases = ((0., .5), (.25, .75), (.1, .6))
    cfg = NaturalGaitCfg(enabled=True, phase_offsets=phases)
    phase = torch.tensor([.0, .2, .4])
    weights = gait_weights(torch.tensor([.3, 1., 3.]), cfg)
    contact = desired_contact(phase, weights, cfg)
    assert contact.shape == (3, 2)
    swapped = replace(cfg, phase_offsets=tuple(row[::-1] for row in phases))
    torch.testing.assert_close(desired_contact(phase, weights, swapped), contact.flip(-1))

    schema = ObservationSpec('custom', (SpecField('input', 1),))
    task = SimpleNamespace(gait=cfg, robot=SimpleNamespace(feet=('pad_left', 'pad_right')),
                           capture_transitions=True)
    core = SimpleNamespace(cfg_obj=task, num_envs=3, device='cpu', observation_spec=schema,
                           critic_observation_spec=schema, num_obs=1, num_privileged_obs=1,
                           task_runtime_metadata=TaskRuntimeMetadata("custom", "v1"), cfg={})
    core.state = BatchEnvState(obs={'actor': torch.zeros(3, 1), 'critic': torch.zeros(3, 1)},
                               reward=None, terminated=None, truncated=None)
    core.locomotion_state = lambda: {'command': torch.zeros(3, 3)}
    env = NaturalGaitEnv(core)
    assert env.get_observations()['actor'].shape == (3, 6)


@pytest.mark.parametrize('phases', [(), ((0.,),), ((0.,), (0., .5), (0.,)),
                                   ((0.,), (0.,), (float('nan'),)), ((0.,), (0.,), (1.,))])
def test_gait_requires_explicit_valid_phase_matrix(phases):
    with pytest.raises(ValueError, match='phase_offsets'):
        NaturalGaitCfg(enabled=True, phase_offsets=phases)


@pytest.mark.parametrize('values', [{'weight': float('nan')}, {'blend_width': float('inf')},
                                    {'frequencies': (1., float('nan'), 2.)}])
def test_gait_rejects_nonfinite_parameters(values):
    with pytest.raises(ValueError, match='finite'):
        NaturalGaitCfg(**values)
