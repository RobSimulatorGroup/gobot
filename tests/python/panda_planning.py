"""Offline Panda scene and independent MuJoCo checks for motion-planning tests."""

from pathlib import Path

import mujoco
import pytest
import torch
from scipy.spatial.transform import Rotation

import gobot


PANDA_URDF = Path(__file__).resolve().parents[1] / "fixtures" / "planning" / "panda.urdf"


@pytest.fixture(scope="module")
def panda():
    context = gobot.app.context()
    previous = context.project_path
    try:
        context.set_project_path(str(PANDA_URDF.parent))
        context.load_scene(f"res://{PANDA_URDF.name}")
        return context.compile_physics_scene_snapshot()
    finally:
        context.clear_scene()
        if previous:
            context.set_project_path(previous)


def reference_pose(model, planner, q, link):
    data = mujoco.MjData(model)
    for (_, name), value in zip(planner.joint_names, q, strict=True):
        joint = model.joint(name)
        assert joint.range[0] <= value.item() <= joint.range[1]
        data.qpos[joint.qposadr[0]] = value.item()
    mujoco.mj_kinematics(model, data)
    body = data.body(link[1])
    return torch.from_numpy(body.xpos.copy()), torch.from_numpy(body.xmat.reshape(3, 3).copy())


def check_pose(model, planner, q, link, position, rotation):
    actual_position, actual_rotation = reference_pose(model, planner, q, link)
    assert torch.linalg.vector_norm(actual_position - position).item() <= 1e-4
    error = Rotation.from_matrix((rotation.T @ actual_rotation).numpy())
    assert error.magnitude() <= 1e-3
