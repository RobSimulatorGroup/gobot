"""Compare scene-owned FK with the independently exported G1 model on random states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gobot
import mujoco
import numpy as np
import torch

from gobot.rl.kinematics import SceneKinematics
from gobot.rl.motion_tracking.kinematics import SceneStateKinematics
from gobot.rl.supermimic.rotation_primitives import matrix_from_quat
from .reference import digest


def check(project, output, samples=16):
    project = Path(project).resolve()
    profile_path = project/'assets/supermimic/robot_profile.json'
    profile = json.loads(profile_path.read_text())
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(project))
        context.load_scene('res://g1_29dof_flat_scene.jscn')
        snapshot = context.compile_physics_scene_snapshot()
        kin = SceneKinematics(snapshot, robot_name=profile['robot_name'], base_link=profile['base_link'],
                              joint_names=profile['joint_names'], dtype=torch.float64)
        state_kin = SceneStateKinematics(kin)
        model = mujoco.MjModel.from_xml_path(str(project/'assets/supermimic/world.xml'))
        bodies = [model.body(name).id for name in kin.body_names]
        joints = [model.joint(name) for name in kin.joint_names]
        rng = np.random.default_rng(1701)
        maxima = {'position_m': 0., 'rotation_matrix': 0., 'linear_velocity_m_s': 0., 'angular_velocity_rad_s': 0.}
        for _ in range(samples):
            data = mujoco.MjData(model)
            data.qpos[:3] = rng.normal(size=3)
            quat = rng.normal(size=4)
            data.qpos[3:7] = quat/np.linalg.norm(quat)
            for joint in joints:
                data.qpos[joint.qposadr[0]] = rng.uniform(joint.range[0], joint.range[1])
            data.qvel[:] = rng.normal(size=model.nv)
            mujoco.mj_forward(model, data)
            root = model.body(profile['base_link']).id
            rotation = data.xmat[root].reshape(3, 3).copy()
            q = np.array([data.qpos[j.qposadr[0]] for j in joints])
            qd = np.array([data.qvel[j.dofadr[0]] for j in joints])
            inputs = [data.xpos[root].copy(), rotation, data.qvel[:3].copy(), rotation@data.qvel[3:6], q, qd]
            tensors = [torch.as_tensor(value, dtype=torch.float64) for value in inputs]
            state_out = state_kin(torch.tensor(data.qpos, dtype=torch.float64),
                                  torch.tensor(data.qvel, dtype=torch.float64))
            actual = (state_out['pos'], matrix_from_quat(state_out['quat']),
                      state_out['lin_vel'], state_out['ang_vel'])
            velocities = np.empty((len(bodies), 6))
            for i, body in enumerate(bodies):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocities[i], 0)
            expected = [data.xpos[bodies], data.xmat[bodies].reshape(-1, 3, 3), velocities[:, 3:], velocities[:, :3]]
            for key, measured, target in zip(maxima, actual, expected, strict=True):
                maxima[key] = max(maxima[key], float(np.max(np.abs(measured.detach().numpy()-target))))
        torch.manual_seed(1701)
        weights = [torch.randn_like(value) for value in actual]
        def scalar(angles, rates):
            values = kin(*tensors[:4], angles, rates, all_bodies=True)
            return sum((value*weight).sum() for value, weight in zip(values, weights, strict=True))
        gradient_passed = torch.autograd.gradcheck(scalar, (tensors[4].clone().requires_grad_(True),
                                                           tensors[5].clone().requires_grad_(True)),
                                                  eps=1e-6, atol=1e-5, rtol=1e-4)
        limits = {'position_m': 2e-5, 'rotation_matrix': 2e-5,
                  'linear_velocity_m_s': 1e-4, 'angular_velocity_rad_s': 1e-4}
        report = {'protocol': 'g1_scene_fk_reference_v1', 'samples': samples, 'seed': 1701,
                  'max_abs_error': maxima, 'limits': limits, 'gradient_check': gradient_passed,
                  'passed': gradient_passed and all(maxima[key] <= limits[key] for key in maxima),
                  'profile_sha256': digest(profile_path), 'kinematics_digest': kin.digest,
                  'links': list(kin.body_names), 'mujoco_version': mujoco.__version__}
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not report['passed']:
            raise RuntimeError(f'G1 FK comparison failed: {output}')
        return report
    finally:
        context.clear_scene()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.project, args.output), indent=2))
