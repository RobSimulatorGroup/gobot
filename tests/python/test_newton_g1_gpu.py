from __future__ import annotations

import os
import json
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
        native_contract = contract["load_native_policy_contract"](
            EXAMPLE / "assets/unitree_g1/rl_policies/g1_29dof.yaml"
        )
        assert native_contract["mjw_joint_names"] == contract["JOINT_NAMES"]
        playback["_validate_native_g1_scene"](robot, joints, native_contract)
        task_config = playback["_load_task_config"](str(EXAMPLE))

        artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        assert len(artifact["robot_names"]) == 1
        assert artifact["dimensions"]["nu"] == 43
        provider = NewtonProvider(
            artifact,
            num_envs=4,
            device="cuda:0",
            fixed_time_step=contract["PHYSICS_DT"],
            nconmax=30,
            njmax=100,
            use_mujoco_contacts=True,
            model_config=playback["NewtonModelConfig"](**task_config["newton_model"]),
        )
        assert provider.use_mujoco_contacts
        assert provider.graph_captured
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
        view = provider.create_robot_view(
            robot_name=artifact["robot_names"][0],
            base_link=contract["BASE_LINK"],
            joint_names=contract["JOINT_NAMES"],
            link_names=contract["LINK_NAMES"],
        )

        device = provider.arrays["joint_q"].device
        default_position = torch.tensor(
            native_contract["mjw_joint_pos"],
            dtype=torch.float32,
            device=device,
        ).reshape(1, 43)
        default_batch = default_position.repeat(4, 1)
        base_pose_batch = torch.tensor(
            [contract["BASE_POSE_XYZW"]], dtype=torch.float32, device=device
        ).repeat(4, 1)
        all_envs = torch.ones(4, dtype=torch.bool, device=device)
        view.reset(
            all_envs,
            base_pose=base_pose_batch,
            base_velocity=torch.zeros((4, 6), device=device),
            joint_position=default_batch,
            joint_velocity=torch.zeros((4, 43), device=device),
            controls=default_batch,
        )

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
            state = view.read_state()
            base_pose = state.base_pose[:1]
            base_velocity = state.base_velocity[:1]
            orientation = base_pose[:, 3:7]
            observation = torch.cat(
                (
                    rotate_inverse(torch, orientation, base_velocity[:, :3]),
                    rotate_inverse(torch, orientation, base_velocity[:, 3:6]),
                    rotate_inverse(torch, orientation, gravity),
                    command,
                    state.joint_position[:1] - default_position,
                    state.joint_velocity[:1],
                    previous_action,
                ),
                dim=1,
            )
            assert tuple(observation.shape) == (1, 141)
            action = policy.action(observation)
            assert tuple(action.shape) == (1, 43)
            assert bool(torch.isfinite(action).all())
            view.set_position_targets(
                (
                    default_position + float(native_contract["action_scale"]) * action
                ).repeat(4, 1),
            )
            provider.step(nsteps=contract["POLICY_DECIMATION"])
            previous_action = action
            if policy_step % 10 == 9:
                height_samples.append(
                    float(view.read_state().base_pose[0, 2].item())
                )

        provider.synchronize()
        provider.assert_no_overflow()
        provider.assert_finite(("joint_q", "joint_qd", "body_q", "body_qd"))
        # Headless baseline from Newton 1.4's unmodified G1 example using its
        # canonical g1_isaac.usd, zero command, and 5 ms step.
        reference = json.loads(
            (ROOT / "tests/fixtures/newton_g1_1_4_reference.json").read_text(encoding="utf-8")
        )
        native_height_samples = np.asarray(reference["zero_command_height_every_10_policy_steps"])
        assert min(height_samples) > 0.75
        np.testing.assert_allclose(
            np.asarray(height_samples),
            native_height_samples,
            rtol=0.0,
            atol=float(reference["absolute_tolerance"]),
        )

        def reset_all() -> None:
            view.reset(
                all_envs,
                base_pose=base_pose_batch,
                base_velocity=torch.zeros((4, 6), device=device),
                joint_position=default_batch,
                joint_velocity=torch.zeros((4, 43), device=device),
                controls=default_batch,
            )

        reset_all()
        view.set_position_targets(default_batch)
        provider.step(nsteps=20)
        first_replay = view.read_state().base_pose.clone()
        first_joints = view.read_state().joint_position.clone()
        reset_all()
        view.set_position_targets(default_batch)
        provider.step(nsteps=20)
        torch.testing.assert_close(view.read_state().base_pose, first_replay, rtol=0.0, atol=0.0)
        torch.testing.assert_close(view.read_state().joint_position, first_joints, rtol=0.0, atol=0.0)

        before_masked_reset = view.read_state().base_pose.clone()
        mask = torch.tensor([False, False, True, False], dtype=torch.bool, device=device)
        shifted_base = base_pose_batch.clone()
        shifted_base[2, 0] = 2.0
        view.reset(
            mask,
            base_pose=shifted_base,
            base_velocity=torch.zeros((4, 6), device=device),
            joint_position=default_batch,
            joint_velocity=torch.zeros((4, 43), device=device),
            controls=default_batch,
        )
        masked_state = view.read_state()
        torch.testing.assert_close(masked_state.base_pose[[0, 1, 3]], before_masked_reset[[0, 1, 3]])
        assert abs(float(masked_state.base_pose[2, 0]) - 2.0) < 1.0e-5
        assert provider.capacities.get("nconmax", 1) > 0

        reset_all()
        forward_command = torch.tensor(
            [reference["forward_command"]], dtype=torch.float32, device=device
        )
        previous_action.zero_()
        start_xy = view.read_state().base_pose[0, :2].clone()
        minimum_height = float("inf")
        for _ in range(int(reference["forward_policy_steps"])):
            state = view.read_state()
            orientation = state.base_pose[:1, 3:7]
            observation = torch.cat(
                (
                    rotate_inverse(torch, orientation, state.base_velocity[:1, :3]),
                    rotate_inverse(torch, orientation, state.base_velocity[:1, 3:6]),
                    rotate_inverse(torch, orientation, gravity),
                    forward_command,
                    state.joint_position[:1] - default_position,
                    state.joint_velocity[:1],
                    previous_action,
                ),
                dim=1,
            )
            action = policy.action(observation)
            view.set_position_targets(
                (
                    default_position + float(native_contract["action_scale"]) * action
                ).repeat(4, 1)
            )
            provider.step(nsteps=contract["POLICY_DECIMATION"])
            previous_action.copy_(action)
            minimum_height = min(
                minimum_height, float(view.read_state().base_pose[0, 2].item())
            )
        provider.synchronize()
        final_state = view.read_state()
        planar_displacement = float(torch.linalg.vector_norm(final_state.base_pose[0, :2] - start_xy))
        lower, upper = reference["forward_planar_displacement_range"]
        assert float(lower) <= planar_displacement <= float(upper)
        assert minimum_height >= float(reference["forward_minimum_height"])
        assert bool(torch.isfinite(final_state.joint_position).all())
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
