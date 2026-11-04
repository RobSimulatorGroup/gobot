"""Same-action CPU control trace against the independent exported G1 model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from gobot.rl.articulation import ArticulationBatch
from gobot.sim.providers.base import RobotBatchSpec
from .reference import digest


def check(project, output, steps=32, backend='mujoco-cpu', reference_backend='mujoco-cpu'):
    if steps < 1 or reference_backend not in ('mujoco-cpu', 'mujoco-warp'):
        raise ValueError('positive steps and an explicit reference backend are required')
    project = Path(project).resolve()
    path = project/'assets/supermimic/robot_profile.json'
    profile = json.loads(path.read_text())
    model = mujoco.MjModel.from_xml_path(str(project/'assets/supermimic/world.xml'))
    joints = [model.joint(name) for name in profile['joint_names']]
    controls = [int(np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)[0]) for joint in joints]
    default = np.asarray(profile['default_joint_position'])
    pose = np.asarray(model.key('init_state').qpos[:7])
    data = [mujoco.MjData(model) for _ in range(2)]
    def reset_reference(value):
        mujoco.mj_resetData(model, value)
        value.qpos[:7] = pose
        for joint, position in zip(joints, default, strict=True):
            value.qpos[joint.qposadr[0]] = position
        value.ctrl[controls] = default
        mujoco.mj_forward(model, value)
    for value in data:
        reset_reference(value)
    batch = ArticulationBatch(
        project_path=project, scene_path='res://g1_29dof_flat_scene.jscn',
        spec=RobotBatchSpec(profile['robot_name'], profile['base_link'], tuple(profile['joint_names']),
                           tuple(profile['tracked_links'])),
        num_envs=2, backend=backend, physics_dt=profile['physics_dt'], deterministic=True,
        decimation=profile['decimation'], solver_settings=profile['solver_settings'],
    )
    def reset_candidate(mask):
        poses = np.tile(np.r_[pose[:3], pose[[4, 5, 6, 3]]], (2, 1))
        batch.reset(mask, base_pose=poses, base_velocity=np.zeros((2, 6)),
                    joint_position=np.tile(default, (2, 1)), joint_velocity=np.zeros((2, 29)),
                    targets=np.tile(default, (2, 1)))
    reference_warp = None
    try:
        if reference_backend == 'mujoco-warp':
            import warp as wp
            import mujoco_warp as mjw
            wp.init()
            with wp.ScopedDevice('cuda:0'):
                wm = mjw.put_model(model)
                wm.opt.deterministic = True
                wm.opt.warn_overflow = 0
                wd = mjw.put_data(model, data[0], nworld=2, nconmax=128, njmax=500)
                mjw.forward(wm, wd)
            reference_warp = (wm, wd)
            wp.synchronize()
        reset_candidate([True, True])
        maxima = {'base_position_m': 0., 'base_rotation_matrix': 0.,
                  'joint_position_rad': 0., 'joint_velocity_rad_s': 0.,
                  'base_linear_velocity_m_s': 0., 'base_angular_velocity_rad_s': 0.}
        checkpoints = {}
        for step in range(steps):
            targets = np.stack([default+.03*np.sin(.2*step+np.arange(29)*.1+lane*.3) for lane in range(2)])
            batch.set_joint_targets(torch.tensor(targets, dtype=torch.float32))
            batch.step_control()
            if reference_warp is None:
                for lane, value in enumerate(data):
                    # Quantize the input identically at the typed float32 boundary.
                    value.ctrl[controls] = targets[lane].astype(np.float32)
                    for _ in range(profile['decimation']):
                        mujoco.mj_step(model, value)
            else:
                wm, wd = reference_warp
                ctrl = np.zeros((2, model.nu), dtype=np.float32)
                ctrl[:, controls] = targets.astype(np.float32)
                wp.to_torch(wd.ctrl).copy_(torch.as_tensor(ctrl, device='cuda:0'))
                torch.cuda.synchronize()
                with wp.ScopedDevice('cuda:0'):
                    for _ in range(profile['decimation']):
                        mjw.step(wm, wd)
                wp.synchronize()
                positions = wp.to_torch(wd.qpos).cpu().numpy().astype(np.float64)
                velocities = wp.to_torch(wd.qvel).cpu().numpy().astype(np.float64)
                data = [SimpleNamespace(qpos=positions[lane], qvel=velocities[lane]) for lane in range(2)]
            state = batch.read_state()
            for lane, value in enumerate(data):
                reference_rotation = np.empty(9)
                candidate_rotation = np.empty(9)
                actual_pose = state.base_pose[lane].cpu().numpy()
                mujoco.mju_quat2Mat(reference_rotation, value.qpos[3:7])
                mujoco.mju_quat2Mat(candidate_rotation, actual_pose[[6, 3, 4, 5]].astype(np.float64))
                expected = {'base_position_m': value.qpos[:3], 'base_rotation_matrix': reference_rotation,
                            'joint_position_rad': np.array([value.qpos[j.qposadr[0]] for j in joints]),
                            'joint_velocity_rad_s': np.array([value.qvel[j.dofadr[0]] for j in joints]),
                            'base_linear_velocity_m_s': value.qvel[:3],
                            'base_angular_velocity_rad_s': reference_rotation.reshape(3, 3)@value.qvel[3:6]}
                actual = {'base_position_m': actual_pose[:3], 'base_rotation_matrix': candidate_rotation,
                          'joint_position_rad': state.joint_position[lane].cpu().numpy(),
                          'joint_velocity_rad_s': state.joint_velocity[lane].cpu().numpy(),
                          'base_linear_velocity_m_s': state.base_velocity[lane, :3].cpu().numpy(),
                          'base_angular_velocity_rad_s': state.base_velocity[lane, 3:].cpu().numpy()}
                for key in maxima:
                    maxima[key] = max(maxima[key], float(np.max(np.abs(expected[key]-actual[key]))))
            if step+1 in (1, 8, 32):
                checkpoints[str(step+1)] = dict(maxima)
        before = batch.read_state()
        kept = {name: getattr(before, name)[1].clone() for name in ('base_pose', 'base_velocity', 'joint_position', 'joint_velocity')}
        reset_candidate([True, False])
        after = batch.read_state()
        masked_reset = all(torch.equal(value, getattr(after, name)[1]) for name, value in kept.items())
        limits = {key: 1e-4 if 'velocity' not in key else 1e-3 for key in maxima}
        report = {'protocol': 'g1_scene_control_reference_v1', 'steps': steps, 'num_envs': 2, 'backend': backend, 'reference_backend': reference_backend,
                  'deterministic': True, 'diagnostic_capacity': {'nconmax': 128, 'njmax': 500},
                  'max_abs_error': maxima, 'limits': limits, 'at_control_step': checkpoints,
                  'masked_reset_preserves_other_lane': masked_reset,
                  'passed': masked_reset and all(maxima[key] <= limits[key] for key in maxima),
                  'profile_sha256': digest(path), 'mujoco_version': mujoco.__version__,
                  'scope': 'scene-compiled physics/control only; mjlab environment timing/rewards remain separate'}
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not report['passed']:
            raise RuntimeError(f'control trace differs: {output}')
        return report
    finally:
        batch.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('mujoco-cpu','mujoco-warp'), default='mujoco-cpu')
    parser.add_argument('--reference-backend', choices=('mujoco-cpu','mujoco-warp'), default='mujoco-cpu')
    args = parser.parse_args()
    print(json.dumps(check(args.project, args.output, backend=args.backend, reference_backend=args.reference_backend), indent=2))
