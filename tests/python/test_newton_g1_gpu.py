from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "newton_g1"


def _walk_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _nodes_by_name(root, names, type_name):
    expected = set(names)
    result = {
        node.name: node
        for node in _walk_nodes(root)
        if node.name in expected and node.type_name == type_name
    }
    missing = [name for name in names if name not in result]
    if missing:
        raise AssertionError(f"scene is missing {type_name} nodes: {', '.join(missing)}")
    return result


def test_real_newton_g1_policy_smoke() -> None:
    if os.environ.get("GOBOT_RUN_NEWTON_G1_GPU_TEST") != "1":
        raise SystemExit(77)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Newton G1 GPU smoke requested but Torch cannot access CUDA")

    scripts = EXAMPLE / "scripts"
    sys.path.insert(0, str(EXAMPLE))
    try:
        contract = runpy.run_path(str(scripts / "g1_policy_contract.py"))
        playback = runpy.run_path(str(scripts / "g1_policy.py"))
    finally:
        sys.path.remove(str(EXAMPLE))

    import gobot
    from gobot.rl.providers import NewtonProvider

    context = gobot.app.create_context()
    provider = None
    try:
        context.set_project_path(str(EXAMPLE))
        root = context.load_scene("res://newton_g1.jscn")
        robots = [node for node in _walk_nodes(root) if node.type_name == "Robot3D"]
        assert len(robots) == 1
        robot = robots[0]
        links = _nodes_by_name(robot, contract["LINK_NAMES"], "Link3D")
        assert len(links) == 44
        joints = _nodes_by_name(robot, contract["JOINT_NAMES"], "Joint3D")
        assert sum(bool(joint.affine_actuator_enabled) for joint in joints.values()) == 43
        native_contract = contract["load_native_policy_contract"](
            EXAMPLE / "assets/unitree_g1/rl_policies/g1_29dof.yaml"
        )
        assert native_contract["mjw_joint_names"] == contract["JOINT_NAMES"]
        playback["_configure_native_g1_scene"](robot, joints, native_contract)

        artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        assert len(artifact["robot_names"]) == 1
        assert artifact["dimensions"]["nu"] == 86
        provider = NewtonProvider(
            artifact,
            num_envs=1,
            device="cuda:0",
            fixed_time_step=contract["PHYSICS_DT"],
            nconmax=30,
            njmax=100,
            use_mujoco_contacts=True,
            model_config=playback["NEWTON_MODEL_CONFIG"],
        )
        assert provider.use_mujoco_contacts
        assert provider.model_config == playback["NEWTON_MODEL_CONFIG"]
        model = provider._model
        limit_stiffness = np.asarray(model.joint_limit_ke.numpy())
        limit_damping = np.asarray(model.joint_limit_kd.numpy())
        active_limits = limit_stiffness > 0.0
        assert int(np.count_nonzero(active_limits)) == 43
        np.testing.assert_allclose(limit_stiffness[active_limits], 100.0)
        np.testing.assert_allclose(limit_damping[active_limits], 1.0)
        np.testing.assert_allclose(model.joint_friction.numpy(), 0.0)
        np.testing.assert_allclose(model.joint_effort_limit.numpy(), 1.0e6)
        np.testing.assert_allclose(model.shape_material_ke.numpy(), 5.0e4)
        np.testing.assert_allclose(model.shape_material_kd.numpy(), 5.0e2)
        np.testing.assert_allclose(model.shape_material_kf.numpy(), 1.0e3)
        layout = provider.resolve_robot_layout(
            artifact["robot_names"][0],
            base_link=contract["BASE_LINK"],
            joint_names=contract["JOINT_NAMES"],
            link_names=contract["LINK_NAMES"],
        )
        assert len(layout.base_joint_q_indices) == 7
        assert len(layout.base_joint_qd_indices) == 6
        assert len(layout.joint_q_indices) == 43
        assert len(layout.link_body_indices) == 44
        assert layout.actuator_modes == ("position",) * 43

        device = provider.arrays["joint_q"].device
        base_q_index = torch.as_tensor(layout.base_joint_q_indices, device=device)
        base_qd_index = torch.as_tensor(layout.base_joint_qd_indices, device=device)
        joint_q_index = torch.as_tensor(layout.joint_q_indices, device=device)
        joint_qd_index = torch.as_tensor(layout.joint_qd_indices, device=device)
        default_position = torch.tensor(
            native_contract["mjw_joint_pos"],
            dtype=torch.float32,
            device=device,
        ).reshape(1, 43)
        provider.reset_robot_state(
            layout,
            torch.ones(1, dtype=torch.bool, device=device),
            base_pose=torch.tensor(
                [contract["BASE_POSE_XYZW"]],
                dtype=torch.float32,
                device=device,
            ),
            base_velocity=torch.zeros((1, 6), device=device),
            joint_position=default_position,
            joint_velocity=torch.zeros((1, 43), device=device),
            controls=default_position,
        )

        arrays = provider.arrays
        rotate_inverse = playback["_quat_rotate_inverse"]
        policy = playback["WarpOnnxPolicy"](
            str(EXAMPLE / "assets/unitree_g1/rl_policies/mjw_g1_29DOF.onnx"),
            device="cuda:0",
            torch=torch,
        )
        gravity = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        command = torch.zeros((1, 3), device=device)
        previous_action = torch.zeros((1, 43), device=device)
        height_samples = []
        for policy_step in range(100):
            base_pose = arrays["joint_q"].index_select(1, base_q_index)
            base_velocity = arrays["joint_qd"].index_select(1, base_qd_index)
            orientation = base_pose[:, 3:7]
            observation = torch.cat(
                (
                    rotate_inverse(torch, orientation, base_velocity[:, :3]),
                    rotate_inverse(torch, orientation, base_velocity[:, 3:6]),
                    rotate_inverse(torch, orientation, gravity),
                    command,
                    arrays["joint_q"].index_select(1, joint_q_index) - default_position,
                    arrays["joint_qd"].index_select(1, joint_qd_index),
                    previous_action,
                ),
                dim=1,
            )
            assert tuple(observation.shape) == (1, 141)
            action = policy.action(observation)
            assert tuple(action.shape) == (1, 43)
            assert bool(torch.isfinite(action).all())
            provider.set_joint_position_targets(
                layout,
                default_position + float(native_contract["action_scale"]) * action,
            )
            provider.step(nsteps=contract["POLICY_DECIMATION"])
            previous_action = action
            if policy_step % 10 == 9:
                height_samples.append(
                    float(arrays["joint_q"][0, layout.base_joint_q_indices[2]].item())
                )

        provider.synchronize()
        provider.assert_no_overflow()
        provider.assert_finite(("joint_q", "joint_qd", "body_q", "body_qd"))
        # Headless baseline from Newton 1.4's example_robot_policy.py using
        # the same structured G1 USD, policy, zero command, and 5 ms step.
        native_height_samples = np.asarray(
            [
                0.80070990,
                0.80293155,
                0.80237371,
                0.79937446,
                0.78262889,
                0.71005958,
                0.44935423,
                0.37238929,
                0.37056434,
                0.36665466,
            ]
        )
        np.testing.assert_allclose(
            np.asarray(height_samples),
            native_height_samples,
            rtol=0.0,
            atol=1.0e-3,
        )
    finally:
        if provider is not None:
            provider.close()
        context.clear_world()
        context.clear_scene()


def main() -> int:
    test_real_newton_g1_policy_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
