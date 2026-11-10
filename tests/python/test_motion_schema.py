"""Motion data layout follows named joints and links, independent of robot type."""
import json

import numpy as np
import pytest
import torch

from gobot.rl.locomotion.motion import MotionDataset, MotionDatasetCfg, MotionFeatureSpec, motion_features


def write_dataset(path, spec, *, schema='gobot_motion_v2', dt=.01):
    arrays, clips = {}, []
    for i, speed in enumerate((.6, 2.)):
        features = np.zeros((5, spec.dim), np.float32)
        features[:, spec.velocity_indices[0]] = speed
        features[:, 0] = i
        arrays[f'features_{i}'] = features
        arrays[f'valid_{i}'] = np.ones(5, dtype=bool)
        # A generic loader must trust the explicit split even if a name resembles
        # one particular source dataset's held-out subject naming convention.
        clips.append({'name': f'clip{i}_subject4', 'split': 'train', 'gait': i})
    arrays['metadata'] = np.array(json.dumps({'schema': schema, 'dt': dt,
        'feature_spec': spec.metadata(), 'clips': clips}))
    np.savez(path, **arrays)


@pytest.mark.parametrize('joint_count,link_count', [(3, 0), (12, 2), (29, 4)])
def test_motion_features_and_sampling_follow_named_layout(tmp_path, joint_count, link_count):
    spec = MotionFeatureSpec(tuple(f'joint_{i}' for i in range(joint_count)),
                             tuple(f'link_{i}' for i in range(link_count)))
    state = dict(base_position=np.zeros((2, 3)), base_quaternion=np.tile([1., 0., 0., 0.], (2, 1)),
                 joint_position=np.ones((2, joint_count)), joint_velocity=np.ones((2, joint_count)) * 2,
                 base_linear_velocity_body=np.tile([.6, 0., 0.], (2, 1)),
                 base_angular_velocity_body=np.tile([0., 0., .2], (2, 1)),
                 link_position=np.ones((2, link_count, 3)))
    features = motion_features(state, spec=spec)
    assert features.shape == (2, spec.dim)
    np.testing.assert_allclose(features[:, spec.velocity_indices], [[.6, 0., .2]] * 2)
    np.testing.assert_allclose(features[:, spec.section('projected_gravity')], [[0., 0., -1.]] * 2)
    path = tmp_path / 'references.npz'
    write_dataset(path, spec)
    data = MotionDataset(path)
    assert data.feature_spec == spec and data.dt == .01
    assert data.pairs.shape == (8, 2 * spec.dim)
    samples, supported = data.sample_matching(torch.tensor([0, 1]), torch.tensor([[.6, 0., 0.], [2., 0., 0.]]))
    assert supported.tolist() == [True, True]
    torch.testing.assert_close(samples[:, spec.velocity_indices[0]], torch.tensor([.6, 2.]))
    torch.testing.assert_close(samples[:, spec.dim + spec.velocity_indices[0]], torch.tensor([.6, 2.]))
    torch.testing.assert_close(data.speeds, torch.tensor([.6] * 4 + [2.] * 4))


def test_old_motion_data_requires_regeneration(tmp_path):
    path = tmp_path / 'references.npz'
    write_dataset(path, MotionFeatureSpec(('motor',), ('tip',)), schema='gobot_motion_v1')
    with pytest.raises(ValueError, match='regenerate'):
        MotionDataset(path)


def test_motion_layout_and_period_mismatches_fail_early(tmp_path):
    path = tmp_path / 'references.npz'
    spec = MotionFeatureSpec(('motor',), ('tip',))
    write_dataset(path, spec)
    with pytest.raises(ValueError, match='control period'):
        MotionDataset(MotionDatasetCfg(str(path), history_dt=.02))
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays['metadata']))
    metadata['feature_spec'] = MotionFeatureSpec(('motor', 'second'), ('tip',)).metadata()
    arrays['metadata'] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='invalid reference clip'):
        MotionDataset(path)
    with pytest.raises(ValueError, match='unique'):
        MotionFeatureSpec(('motor', 'motor'), ('tip',))
    with pytest.raises(ValueError, match='sequence'):
        MotionFeatureSpec('motor', ('tip',))
