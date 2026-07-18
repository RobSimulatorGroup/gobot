import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GO1_PROJECT = REPO_ROOT / "examples/go1"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GO1_PROJECT))

from examples.go1.scripts import go1 as go1_playback
from examples.go1 import go1_profile
from examples.go1.go1_velocity_contract import GO1_TASK_VERSION
from examples.go1.train import go1_velocity_cfg as go1_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
from examples.go1.train.go1_training_state import (
    build_terrain_curriculum_state,
    restore_terrain_curriculum_assignments,
)
import gobot
from gobot.rl import (
    BatchEnvState,
    CompiledSceneArtifact,
    CpuBatchEnv,
    MuJoCoWarpProvider,
    TaskRuntimeMetadata,
)
from gobot.rl.locomotion import (
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)
from gobot.rl.rsl_rl import RslRlVecEnvWrapper


OPTIONAL_DEPENDENCY_SKIP_CODE = 77


class OptionalDependencyUnavailable(RuntimeError):
    pass


def _require_torch():
    try:
        return __import__("torch")
    except ImportError as error:
        raise OptionalDependencyUnavailable("torch is unavailable; skipping torch-backed integration test") from error


def _assert_runtime_error(pattern: str, fn) -> None:
    try:
        fn()
    except RuntimeError as error:
        assert pattern in str(error)
        return
    raise AssertionError("expected RuntimeError")


def _matrix_storage(value):
    return tuple(value["matrix_data"]["storage"])


def _node_by_name(nodes, name: str, type_name: str | None = None):
    for node in nodes:
        if node["name"] == name and (type_name is None or node["type"] == type_name):
            return node
    raise AssertionError(f"missing node {name!r}")


def test_repo_imports_use_one_gobot_package():
    import gobot
    import gobot._core
    import gobot.rl

    assert gobot.__version__ == gobot._core.__version__
    assert gobot.__package__ == "gobot"
    assert gobot.rl.__package__ == "gobot.rl"
    assert go1_profile.__package__ == "examples.go1"
    assert not (REPO_ROOT / "python/gobot/rl/tasks/go1.py").exists()


def test_batch_env_state_contract():
    state = BatchEnvState(
        obs={"actor": np.zeros((3, 2), dtype=np.float32), "critic": np.ones((3, 3), dtype=np.float32)},
        reward=np.zeros(3, dtype=np.float32),
        terminated=np.asarray([False, True, False]),
        truncated=np.asarray([False, False, True]),
        info={"steps": np.asarray([2, 3, 4], dtype=np.int64)},
    )
    assert state.done.tolist() == [False, True, True]
    assert isinstance(state.obs, dict)


def test_batch_env_clears_stale_final_observation_info():
    env = CpuBatchEnv(num_envs=2)
    env._state = env.make_empty_state({"actor": 2})
    env._state.obs["actor"][:] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    env.capture_final_observation(np.asarray([1], dtype=np.int64))
    assert "final_observation" in env._state.info
    assert env._state.info["_final_observation"].tolist() == [False, True]

    env.clear_step_final_observation()
    assert env._state.final_observation is None
    assert "final_observation" not in env._state.info
    assert "_final_observation" not in env._state.info


def test_task_runtime_metadata_is_public_task_summary():
    actor_spec = velocity_actor_observation_schema(1, 0)
    metadata = TaskRuntimeMetadata(
        name="dummy",
        version="dummy_numpy_v1",
        obs_groups_spec={"actor": actor_spec.dim},
        reward_names=("alive",),
        backend="gobot_native_cpu_batch_numpy",
        cache_info={"note": "unit-test"},
    )
    payload = metadata.metadata()
    assert payload["kind"] == "gobot_task_runtime_metadata"
    assert payload["obs_groups_spec"] == {"actor": actor_spec.dim}
    assert payload["reward_names"] == ["alive"]
    assert payload["cache_info"] == {"note": "unit-test"}


def test_go1_robot_scene_matches_mujoco_contract():
    scene = json.loads((REPO_ROOT / "examples/go1/go1.jscn").read_text(encoding="utf-8"))
    nodes = scene["__NODES__"]

    hip_joint = _node_by_name(nodes, "FR_hip_joint", "Joint3D")["properties"]
    assert hip_joint["damping"] == 0.0
    assert hip_joint["armature"] == np.float32(0.000111842 * 6.0**2)
    assert hip_joint["friction_loss"] == 0.0
    assert hip_joint["drive_mode"] == "Position"
    assert hip_joint["drive_stiffness"] == np.float32(15.895242654143557)
    assert hip_joint["drive_damping"] == np.float32(1.0119225760208341)
    assert hip_joint["velocity_limit"] == np.float32(30.1)
    assert hip_joint["force_lower_limit"] == np.float32(-23.7)
    assert hip_joint["force_upper_limit"] == np.float32(23.7)
    assert hip_joint["control_lower_limit"] == 0.0
    assert hip_joint["control_upper_limit"] == 0.0

    thigh_joint = _node_by_name(nodes, "FR_thigh_joint", "Joint3D")["properties"]
    calf_joint = _node_by_name(nodes, "FR_calf_joint", "Joint3D")["properties"]
    assert thigh_joint["damping"] == 0.0
    assert thigh_joint["armature"] == np.float32(0.000111842 * 6.0**2)
    assert thigh_joint["friction_loss"] == 0.0
    assert thigh_joint["drive_stiffness"] == np.float32(15.895242654143557)
    assert thigh_joint["drive_damping"] == np.float32(1.0119225760208341)
    assert thigh_joint["velocity_limit"] == np.float32(30.1)
    assert thigh_joint["force_lower_limit"] == np.float32(-23.7)
    assert thigh_joint["force_upper_limit"] == np.float32(23.7)
    assert calf_joint["damping"] == 0.0
    assert calf_joint["armature"] == np.float32(0.000111842 * 9.0**2)
    assert calf_joint["friction_loss"] == 0.0
    assert calf_joint["drive_stiffness"] == np.float32(35.764295971822996)
    assert calf_joint["drive_damping"] == np.float32(2.2768257960468765)
    assert calf_joint["velocity_limit"] == np.float32(20.06)
    assert calf_joint["force_lower_limit"] == np.float32(-35.55)
    assert calf_joint["force_upper_limit"] == np.float32(35.55)
    assert calf_joint["control_lower_limit"] == 0.0
    assert calf_joint["control_upper_limit"] == 0.0

    node_names = {node["name"] for node in nodes}
    assert {
        "trunk_collision",
        "head_collision",
        "FR_hip_collision",
        "FR_thigh_collision1",
        "FR_calf_collision1",
        "FR_foot_collision",
        "FR_foot_height_scan",
        "FR_foot_contact",
    } <= node_names

    base_collision = _node_by_name(nodes, "trunk_collision", "CollisionShape3D")["properties"]
    assert base_collision["condim"] == 1
    np.testing.assert_allclose(_matrix_storage(base_collision["solref"]), (0.01, 1.0))

    for collision_name in ("FR_thigh_collision1", "FR_calf_collision1"):
        collision = _node_by_name(nodes, collision_name, "CollisionShape3D")["properties"]
        assert collision["condim"] == 1
        np.testing.assert_allclose(_matrix_storage(collision["solref"]), (0.01, 1.0))

    foot_collision = _node_by_name(nodes, "FR_foot_collision", "CollisionShape3D")["properties"]
    np.testing.assert_allclose(_matrix_storage(foot_collision["friction"]), (1.0, 0.005, 0.0005))
    assert foot_collision["condim"] == 6
    assert foot_collision["priority"] == 1
    np.testing.assert_allclose(_matrix_storage(foot_collision["solref"]), (0.01, 1.0))

    trunk_index = nodes.index(_node_by_name(nodes, "trunk", "Link3D"))
    assert _node_by_name(nodes, "imu", "IMUSensor3D")["parent"] == trunk_index
    assert _node_by_name(nodes, "root_angmom", "AngularMomentumSensor3D")["parent"] == trunk_index
    terrain_scan = _node_by_name(nodes, "terrain_scan", "HeightScanner3D")["properties"]
    assert terrain_scan["pattern_mode"] == "Grid"
    assert terrain_scan["reduction_mode"] == "None"
    assert terrain_scan["visualize_debug"] is True
    for foot in ("FR", "FL", "RR", "RL"):
        foot_height_scan = _node_by_name(nodes, f"{foot}_foot_height_scan", "TerrainHeightSensor3D")[
            "properties"
        ]
        assert foot_height_scan["reduction_mode"] == "Min"
        assert foot_height_scan["visualize_debug"] is True
        assert len(foot_height_scan["sample_offsets"]) == 5
        np.testing.assert_allclose(_matrix_storage(foot_height_scan["sample_offsets"][0]), (0.0, 0.0, 0.0))
        foot_contact = _node_by_name(nodes, f"{foot}_foot_contact", "ContactSensor3D")["properties"]
        assert foot_contact["visualize_debug"] is True
        assert foot_contact["radius"] == np.float32(0.03)


def test_go1_env_applies_mujoco_solver_settings():
    cfg = go1_cfg.go1_velocity_cfg(project_path=str(REPO_ROOT / "examples/go1"))
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = False
    cfg.observations.actor_noise = False

    env = Go1VelocityEnv(num_envs=1, cfg=cfg, sim_workers=0, seed=3)
    try:
        solver_settings = env.context.get_mujoco_solver_settings()
        assert solver_settings["solver"] == gobot.PhysicsSolverType.Newton
        assert solver_settings["integrator"] == gobot.PhysicsIntegratorType.ImplicitFast
        assert solver_settings["cone"] == gobot.PhysicsFrictionConeType.Elliptic
        assert solver_settings["jacobian"] == gobot.PhysicsJacobianType.Auto
        assert solver_settings["iterations"] == 10
        assert solver_settings["line_search_iterations"] == 20
        assert solver_settings["convex_collision_iterations"] == 500
        assert solver_settings["impedance_ratio"] == 10.0
        assert env.cfg["mujoco_solver_settings"] == dict(go1_cfg.GO1_MUJOCO_SOLVER_SETTINGS)
        assert env.physics_dt == 0.005
        assert env.decimation == 4
        assert env.step_dt == 0.02
    finally:
        env.close()


def test_go1_rough_env_reset_step_shapes():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    env = Go1VelocityEnv(cfg, num_envs=2, device="cpu", seed=123, max_episode_length=4, sim_workers=1)
    try:
        obs = env.get_observations()
        assert obs["actor"].shape == (2, 235)
        assert obs["critic"].shape == (2, 259)
        assert env.obs_groups_spec == {"actor": 235, "critic": 259}
        assert env.cfg["action_clip"] is None
        assert np.all(env.action_spec.lower == -np.inf)
        assert np.all(env.action_spec.upper == np.inf)
        assert env.cfg["task_runtime"]["reward_names"] == (
            "track_linear_velocity",
            "track_angular_velocity",
            "upright",
            "pose",
            "body_ang_vel",
            "angular_momentum",
            "dof_pos_limits",
            "action_rate_l2",
            "air_time",
            "foot_clearance",
            "foot_swing_height",
            "foot_slip",
            "soft_landing",
            "self_collisions",
            "shank_collision",
            "trunk_head_collision",
        )
        assert env.task_runtime_metadata.version == GO1_TASK_VERSION
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["native_contact_detail"] == "substep_contact_history"
        assert env._runtime_terrain_node is not None
        assert env._runtime_terrain_node.box_count == 514
        assert env._runtime_terrain_node.heightfield_count == 40
        assert env._spawn_grid_shape == (10, 7)
        assert env._spawn_origins.shape == (70, 3)
        assert np.all(env._spawn_env_levels >= 0)
        assert np.all(env._spawn_env_levels <= 5)
        assert np.all(env._spawn_type_cols >= 0)
        assert np.all(env._spawn_type_cols < 7)
        assert "foot_friction" in env.cfg["task_runtime"]["array_names"]
        assert "foot_friction_enabled" in env.cfg["task_runtime"]["array_names"]
        np.testing.assert_allclose(
            env.backend.state.default_joint_position,
            go1_profile.GO1_DEFAULT_JOINT_POS,
        )
        np.testing.assert_allclose(env.backend.state.action_scale[[0, 1, 3, 4]], 0.3727530386870487)
        assert np.all(env.backend.state.foot_friction_enabled > 0.5)

        original_encoder_bias = env.backend.state.encoder_bias.copy()
        encoder_bias = np.full((env.num_envs, env.num_actions), 0.01, dtype=np.float32)
        np.copyto(env.backend.state.encoder_bias, encoder_bias)
        env.backend.step_task_inputs(
            np.zeros((env.num_envs, env.num_actions), dtype=np.float32),
            0,
            workers=1,
        )
        np.testing.assert_allclose(
            env.backend.state.target_position,
            np.asarray(go1_profile.GO1_DEFAULT_JOINT_POS, dtype=np.float32).reshape(1, -1)
            - encoder_bias,
        )
        actor_obs, _ = env._build_generic_velocity_observations(env.backend.state)
        np.testing.assert_allclose(
            actor_obs[:, 9 : 9 + env.num_actions],
            env.backend.state.joint_position
            - np.asarray(go1_profile.GO1_DEFAULT_JOINT_POS, dtype=np.float32).reshape(1, -1),
        )
        np.copyto(env.backend.state.encoder_bias, original_encoder_bias)

        assert env.cfg_obj.command.ranges.lin_vel_x == (-1.0, 1.0)
        env.set_training_progress(59_999)
        assert env.cfg_obj.command.ranges.lin_vel_x == (-1.0, 1.0)
        env.set_training_progress(60_000)
        assert env.cfg_obj.command.ranges.lin_vel_x == (-1.5, 2.0)
        env.set_training_progress(120_000)
        assert env.cfg_obj.command.ranges.lin_vel_x == (-2.0, 3.0)

        startup_encoder_bias = env.backend.state.encoder_bias.copy()
        startup_com_offset = env.backend.state.base_com_offset.copy()
        startup_foot_friction = env.backend.state.foot_friction.copy()
        env.reset()
        np.testing.assert_array_equal(env.backend.state.encoder_bias, startup_encoder_bias)
        np.testing.assert_array_equal(env.backend.state.base_com_offset, startup_com_offset)
        np.testing.assert_array_equal(env.backend.state.foot_friction, startup_foot_friction)

        state = env.step(np.zeros((env.num_envs, env.num_actions), dtype=np.float32))
        assert isinstance(state, BatchEnvState)
        assert state.obs["actor"].shape == (2, 235)
        assert state.obs["critic"].shape == (2, 259)
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        assert np.all(env.backend.state.illegal_contact_count <= env.decimation)
        assert np.all(env.backend.state.shank_collision_count <= env.decimation)
        assert np.all(env.backend.state.trunk_head_collision_count <= env.decimation)
    finally:
        env.close()


def _synthetic_go1_task_env():
    cfg = go1_cfg.go1_velocity_cfg(project_path="/tmp/go1")
    num_envs = 2
    num_actions = len(cfg.joint_names)
    foot_count = len(cfg.foot_names)
    height_scan_dim = 4
    actor_dim = velocity_actor_observation_schema(num_actions, height_scan_dim).dim
    critic_dim = velocity_critic_observation_schema(num_actions, height_scan_dim, foot_count).dim
    default_joint_position = np.asarray(cfg.default_joint_pos, dtype=np.float32)
    joint_position = np.broadcast_to(default_joint_position, (num_envs, num_actions)).copy()
    joint_position[0, 0] += 0.2
    joint_position[0, 2] -= 0.15
    joint_position[1] += 0.01
    action = np.asarray(
        [np.linspace(-0.3, 0.3, num_actions), np.full(num_actions, 0.2)],
        dtype=np.float32,
    )
    flat_scan_points = np.asarray(
        [[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    reward_weights = np.asarray(
        [
            cfg.rewards.track_linear_velocity,
            cfg.rewards.track_angular_velocity,
            cfg.rewards.upright,
            cfg.rewards.pose,
            cfg.rewards.body_ang_vel,
            cfg.rewards.angular_momentum,
            cfg.rewards.dof_pos_limits,
            cfg.rewards.action_rate_l2,
            cfg.rewards.air_time,
            cfg.rewards.foot_clearance,
            cfg.rewards.foot_swing_height,
            cfg.rewards.foot_slip,
            cfg.rewards.soft_landing,
            cfg.rewards.self_collisions,
            cfg.rewards.shank_collision,
            cfg.rewards.trunk_head_collision,
        ],
        dtype=np.float32,
    )
    state = SimpleNamespace(
        task_params=np.asarray([0.02, 0.25, 0.5, 0.2, 0.05, 5.0], dtype=np.float32),
        reward_weights=reward_weights,
        command=np.asarray([[0.5, -0.2, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32),
        base_quaternion=np.asarray([[1.0, 0.0, 0.0, 0.0]] * num_envs, dtype=np.float32),
        base_linear_velocity=np.asarray([[0.2, -0.1, 0.05], [0.1, 0.0, -0.2]], dtype=np.float32),
        base_angular_velocity=np.asarray([[0.1, -0.2, 0.25], [0.0, 0.0, 0.0]], dtype=np.float32),
        base_linear_velocity_body=np.asarray([[0.2, -0.1, 0.05], [0.1, 0.0, -0.2]], dtype=np.float32),
        base_angular_velocity_body=np.asarray([[0.1, -0.2, 0.25], [0.0, 0.0, 0.0]], dtype=np.float32),
        projected_gravity=np.asarray([[0.0, 0.0, -1.0]] * num_envs, dtype=np.float32),
        joint_position=joint_position,
        joint_velocity=np.zeros((num_envs, num_actions), dtype=np.float32),
        previous_action=np.zeros((num_envs, num_actions), dtype=np.float32),
        submitted_action=action.copy(),
        action=action.copy(),
        last_action=action.copy(),
        default_joint_position=default_joint_position.copy(),
        joint_lower_limit=default_joint_position - 0.1,
        joint_upper_limit=default_joint_position + 0.1,
        pose_std_standing=np.asarray([0.1 if "calf" in name else 0.05 for name in cfg.joint_names], dtype=np.float32),
        pose_std_walking=np.asarray([0.6 if "calf" in name else 0.3 for name in cfg.joint_names], dtype=np.float32),
        pose_std_running=np.asarray([0.6 if "calf" in name else 0.3 for name in cfg.joint_names], dtype=np.float32),
        foot_contact=np.asarray([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        foot_height=np.asarray([[0.04, 0.12, 0.08, 0.15], [0.03, 0.03, 0.03, 0.03]], dtype=np.float32),
        foot_velocity=np.asarray(
            [
                [[0.2, 0.1, 0.0], [0.3, 0.0, 0.0], [0.1, 0.2, 0.0], [0.0, 0.4, 0.0]],
                [[0.2, 0.0, 0.0]] * foot_count,
            ],
            dtype=np.float32,
        ),
        foot_peak_height=np.asarray([[0.08, 0.12, 0.10, 0.15], [0.2, 0.2, 0.2, 0.2]], dtype=np.float32),
        first_contact=np.asarray([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        landing_force=np.asarray([[20.0, 0.0, 30.0, 0.0], [100.0, 100.0, 100.0, 100.0]], dtype=np.float32),
        self_collision_count=np.asarray([2.0, 0.0], dtype=np.float32),
        shank_collision_count=np.asarray([1.0, 3.0], dtype=np.float32),
        trunk_head_collision_count=np.asarray([0.0, 1.0], dtype=np.float32),
        illegal_contact_count=np.asarray([0.0, 1.0], dtype=np.float32),
        height_scan=np.asarray([[0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32),
        height_scan_point=np.broadcast_to(flat_scan_points, (num_envs, height_scan_dim, 3)).copy(),
        height_scan_hit=np.ones((num_envs, height_scan_dim), dtype=bool),
        foot_air_time=np.asarray([[0.0, 0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        foot_contact_force=np.arange(num_envs * foot_count * 3, dtype=np.float32).reshape(num_envs, foot_count, 3),
        base_height=np.asarray([0.3, 0.25], dtype=np.float32),
        reward_terms=np.zeros((num_envs, 16), dtype=np.float32),
        reward=np.zeros(num_envs, dtype=np.float32),
        terminated=np.zeros(num_envs, dtype=np.uint8),
        velocity_error=np.zeros(num_envs, dtype=np.float32),
        foot_slip=np.zeros(num_envs, dtype=np.float32),
        base_clearance=np.zeros(num_envs, dtype=np.float32),
        terrain_normal_error=np.zeros(num_envs, dtype=np.float32),
        actor_obs=np.zeros((num_envs, actor_dim), dtype=np.float32),
        critic_obs=np.zeros((num_envs, critic_dim), dtype=np.float32),
    )
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = cfg
    env.num_envs = num_envs
    env.num_actions = num_actions
    env._foot_count = foot_count
    env._height_scan_dim = height_scan_dim
    env._reward_term_names = (
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "body_ang_vel",
        "angular_momentum",
        "dof_pos_limits",
        "action_rate_l2",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "self_collisions",
        "shank_collision",
        "trunk_head_collision",
    )
    env.backend = SimpleNamespace(state=state)
    return env, state


def test_go1_reward_and_observation_formulas_match_task_contract():
    env, state = _synthetic_go1_task_env()
    original_peak_height = state.foot_peak_height.copy()
    env._run_go1_rough_task_numpy()

    cfg = env.cfg_obj.rewards
    command = state.command
    lin_vel = state.base_linear_velocity_body
    ang_vel = state.base_angular_velocity_body
    command_speed = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])
    active = (command_speed > cfg.command_threshold).astype(np.float32)
    lin_error = np.sum(np.square(command[:, :2] - lin_vel[:, :2]), axis=1) + np.square(lin_vel[:, 2])
    ang_error = np.square(command[:, 2] - ang_vel[:, 2]) + np.sum(np.square(ang_vel[:, :2]), axis=1)
    pose_std = env._go1_pose_std(command_speed)
    pose_error = np.mean(
        np.square((state.joint_position - state.default_joint_position.reshape(1, -1)) / pose_std),
        axis=1,
    )
    foot_speed = np.linalg.norm(state.foot_velocity[:, :, :2], axis=2)
    expected = np.zeros_like(state.reward_terms)
    expected[:, 0] = cfg.track_linear_velocity * np.exp(-lin_error / cfg.lin_vel_std**2)
    expected[:, 1] = cfg.track_angular_velocity * np.exp(-ang_error / cfg.ang_vel_std**2)
    expected[:, 2] = cfg.upright
    expected[:, 3] = cfg.pose * np.exp(-pose_error)
    expected[:, 6] = cfg.dof_pos_limits * env._soft_joint_limit_penalty(state, cfg.soft_joint_pos_limit_factor)
    expected[:, 7] = cfg.action_rate_l2 * np.sum(np.square(state.action - state.previous_action), axis=1)
    expected[:, 9] = cfg.foot_clearance * np.sum(
        np.abs(state.foot_height - cfg.foot_clearance_target_height) * foot_speed,
        axis=1,
    ) * active
    expected[:, 10] = cfg.foot_swing_height * np.sum(
        np.square(original_peak_height / cfg.foot_clearance_target_height - 1.0) * state.first_contact,
        axis=1,
    ) * active
    expected[:, 11] = cfg.foot_slip * np.sum(
        np.square(foot_speed) * (state.foot_contact > 0.0),
        axis=1,
    ) * active
    expected[:, 12] = cfg.soft_landing * np.sum(state.landing_force, axis=1) * active
    expected[:, 13] = cfg.self_collisions * state.self_collision_count
    expected[:, 14] = cfg.shank_collision * state.shank_collision_count
    expected[:, 15] = cfg.trunk_head_collision * state.trunk_head_collision_count

    np.testing.assert_allclose(state.reward_terms, expected, rtol=2.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(state.reward, np.sum(expected, axis=1) * 0.02, rtol=2.0e-5, atol=1.0e-6)
    np.testing.assert_array_equal(state.terminated, [0, 1])
    np.testing.assert_allclose(state.actor_obs[:, -4:], state.height_scan / 5.0)
    np.testing.assert_allclose(state.critic_obs[:, : state.actor_obs.shape[1]], state.actor_obs)

    state.base_linear_velocity[0, 0] = np.nan
    env._run_go1_rough_task_numpy()
    assert np.isfinite(state.reward_terms).all()
    assert np.isfinite(state.reward).all()


def test_go1_run_progress_reward_does_not_reward_standing():
    from examples.go1.train.go1_gait import gait_foot_indices, gait_joint_indices

    env, state = _synthetic_go1_task_env()
    go1_cfg.apply_training_profile(env.cfg_obj, "run")
    env._reward_term_names = (*env._reward_term_names, "run_progress", "bound_gait")
    env._gait_foot_indices = gait_foot_indices(env.cfg_obj.foot_names)
    env._gait_joint_indices = gait_joint_indices(env.cfg_obj.joint_names)
    state.reward_weights = np.asarray(
        [getattr(env.cfg_obj.rewards, name) for name in env._reward_term_names],
        dtype=np.float32,
    )
    state.reward_terms = np.zeros((env.num_envs, len(env._reward_term_names)), dtype=np.float32)
    state.command_is_run_env = np.asarray([True, True])
    state.command[:, :] = 0.0
    state.command[:, 0] = 0.5
    state.base_linear_velocity[:, :] = 0.0
    state.base_linear_velocity[0, 0] = 0.2

    env._run_go1_rough_task_numpy()

    progress_index = env._reward_term_names.index("run_progress")
    np.testing.assert_allclose(
        state.reward_terms[:, progress_index],
        [env.cfg_obj.rewards.run_progress * 0.4, 0.0],
    )
    assert np.isfinite(state.reward_terms).all()
    assert np.isfinite(state.reward).all()


def test_go1_training_configuration_matches_task_contract():
    cfg = go1_cfg.go1_velocity_cfg(project_path="/tmp/go1")
    train_cfg = go1_cfg.rsl_rl_train_cfg()
    assert train_cfg["num_steps_per_env"] == 24
    assert train_cfg["max_iterations"] == 10_000
    assert train_cfg["save_interval"] == 50
    assert train_cfg["actor"]["hidden_dims"] == [512, 256, 128]
    assert train_cfg["critic"]["hidden_dims"] == [512, 256, 128]
    assert train_cfg["algorithm"] == {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "use_clipped_value_loss": True,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "learning_rate": 1.0e-3,
        "max_grad_norm": 1.0,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    }
    assert [stage.step for stage in cfg.command_curriculum] == [0, 2_500 * 24, 5_000 * 24]
    assert cfg.episode_length_s == 20.0
    assert cfg.physics_dt == 0.005
    assert cfg.decimation == 4
    assert cfg.training_profile == "balanced"
    assert cfg.command.rel_run_envs == 0.0


def test_go1_run_training_profile_reserves_high_speed_forward_commands():
    cfg = go1_cfg.go1_velocity_cfg(project_path="/tmp/go1")

    returned = go1_cfg.apply_training_profile(cfg, "run")

    assert returned is cfg
    assert cfg.training_profile == "run"
    assert cfg.command.rel_forward_envs == 0.3
    assert cfg.command.rel_run_envs == 0.6
    assert cfg.command.run_velocity_x == (1.5, 2.5)
    assert [stage.step for stage in cfg.command_curriculum] == [0, 2_500 * 24]
    assert [stage.step for stage in cfg.run_command_curriculum] == [
        0,
        100 * 24,
        250 * 24,
    ]
    assert [stage.run_velocity_x for stage in cfg.run_command_curriculum] == [
        (1.5, 2.5),
        (2.0, 3.0),
        (2.5, 3.5),
    ]
    assert cfg.rewards.track_linear_velocity == 4.0
    assert cfg.rewards.run_progress == 4.0
    assert cfg.rewards.bound_gait == 3.0


def test_go1_bound_gait_score_requires_pair_sync_speed_and_run_command():
    from examples.go1.train.go1_gait import bound_gait_score_numpy

    contacts = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    velocity = np.zeros((4, 4, 3), dtype=np.float32)
    velocity[1, 0, 0] = 1.0
    velocity[1, 1, 0] = -1.0
    velocity[1, 2, 2] = 1.0
    velocity[1, 3, 2] = -1.0
    height = np.asarray(
        [
            [0.1, 0.1, 0.0, 0.0],
            [0.1, 0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    action = np.zeros((4, 12), dtype=np.float32)
    action[1, 1:3] = 1.0
    action[1, 4:6] = -1.0
    action[1, 7:9] = 1.0
    action[1, 10:12] = -1.0
    score = bound_gait_score_numpy(
        contacts,
        velocity,
        height,
        action,
        base_velocity_x=np.asarray([2.0, 2.0, 0.0, 2.0], dtype=np.float32),
        command_x=np.full(4, 2.0, dtype=np.float32),
        run_mask=np.asarray([True, True, True, False]),
        foot_indices=(0, 1, 2, 3),
        joint_indices=((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)),
        motion_std=1.0,
        action_std=0.5,
        height_sync_std=0.04,
        height_separation_std=0.05,
        trot_penalty=0.5,
    )

    assert score[0] > 0.99
    assert score[1] < 0.01
    assert score[2] == 0.0
    assert score[3] == 0.0


def test_go1_bound_gait_score_matches_torch_backend():
    torch = _require_torch()
    from examples.go1.train.go1_gait import bound_gait_score_numpy
    from examples.go1.train.go1_warp_velocity_env import _bound_gait_score_torch

    rng = np.random.default_rng(47)
    contact = rng.integers(0, 2, size=(16, 4), dtype=np.int64).astype(np.float32)
    velocity = rng.normal(size=(16, 4, 3)).astype(np.float32)
    height = rng.uniform(0.0, 0.2, size=(16, 4)).astype(np.float32)
    action = rng.normal(size=(16, 12)).astype(np.float32)
    base_velocity_x = rng.uniform(-0.5, 3.5, size=16).astype(np.float32)
    command_x = rng.uniform(1.5, 3.5, size=16).astype(np.float32)
    run_mask = rng.integers(0, 2, size=16, dtype=np.int64).astype(bool)
    expected = bound_gait_score_numpy(
        contact,
        velocity,
        height,
        action,
        base_velocity_x,
        command_x,
        run_mask,
        foot_indices=(0, 1, 2, 3),
        joint_indices=((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)),
        motion_std=0.8,
        action_std=0.6,
        height_sync_std=0.04,
        height_separation_std=0.05,
        trot_penalty=0.4,
    )
    actual = _bound_gait_score_torch(
        torch.from_numpy(contact),
        torch.from_numpy(velocity),
        torch.from_numpy(height),
        torch.from_numpy(action),
        torch.from_numpy(base_velocity_x),
        torch.from_numpy(command_x),
        torch.from_numpy(run_mask),
        foot_indices=torch.arange(4, dtype=torch.long),
        joint_indices=torch.arange(12, dtype=torch.long).view(4, 3),
        motion_std=0.8,
        action_std=0.6,
        height_sync_std=0.04,
        height_separation_std=0.05,
        trot_penalty=0.4,
    )

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-6, atol=1.0e-7)


def test_go1_footfall_patterns_distinguish_bound_trot_and_flight():
    torch = _require_torch()
    from examples.go1.tools.evaluate_velocity_policy import _footfall_patterns

    contacts = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    patterns = _footfall_patterns(contacts)

    assert patterns["bound_support"].tolist() == [True, True, False, False, False, False]
    assert patterns["trot_support"].tolist() == [False, False, True, True, False, False]
    assert patterns["flight"].tolist() == [False, False, False, False, True, False]
    assert patterns["full_stance"].tolist() == [False, False, False, False, False, True]
    assert patterns["front_pair_sync"].tolist() == [True, True, False, False, True, True]
    assert patterns["rear_pair_sync"].tolist() == [True, True, False, False, True, True]


def test_go1_gait_metrics_are_defined_without_surviving_samples():
    from examples.go1.tools.evaluate_velocity_policy import _group_gait_metrics

    metrics = _group_gait_metrics(
        np.asarray([True]),
        samples=np.asarray([0]),
        pattern_counts={"front_pair_sync": np.asarray([0])},
        squared_error_sums={"front_action_pair_rmse": np.asarray([0.0])},
        scalar_sums={"fore_rear_height_separation_m": np.asarray([0.0])},
    )

    assert metrics == {
        "footfall_samples": 0,
        "front_pair_sync_ratio": 0.0,
        "front_action_pair_rmse": 0.0,
        "fore_rear_height_separation_m": 0.0,
        "bound_over_trot": 0.0,
    }


def test_go1_terrain_curriculum_checkpoint_restores_exact_assignments():
    levels = np.asarray([0, 1, 2, 0, 2, 2], dtype=np.int64)
    terrain_types = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    state = build_terrain_curriculum_state(levels, terrain_types, rows=3, cols=2)

    restored_levels, restored_types, exact = restore_terrain_curriculum_assignments(
        state,
        np.zeros_like(levels),
        np.zeros_like(terrain_types),
        rows=3,
        cols=2,
    )

    assert exact
    np.testing.assert_array_equal(restored_levels, levels)
    np.testing.assert_array_equal(restored_types, terrain_types)


def test_go1_terrain_curriculum_checkpoint_resamples_across_batch_sizes():
    levels = np.asarray([0, 1, 2, 0, 2, 2], dtype=np.int64)
    terrain_types = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    state = build_terrain_curriculum_state(levels, terrain_types, rows=3, cols=2)
    target_types = np.repeat(np.arange(2, dtype=np.int64), 6)

    restored_levels, restored_types, exact = restore_terrain_curriculum_assignments(
        state,
        np.zeros(12, dtype=np.int64),
        target_types,
        rows=3,
        cols=2,
    )

    assert not exact
    np.testing.assert_array_equal(restored_types, target_types)
    assert np.bincount(restored_levels[restored_types == 0], minlength=3).tolist() == [2, 2, 2]
    assert np.bincount(restored_levels[restored_types == 1], minlength=3).tolist() == [2, 0, 4]


def test_go1_checkpoint_admission_requires_survival_and_progress():
    _require_torch()
    from examples.go1.tools.evaluate_velocity_policy import _group_metrics

    metrics = _group_metrics(
        np.ones(3, dtype=bool),
        reward=np.zeros(3, dtype=np.float32),
        velocity_error=np.zeros(3, dtype=np.float32),
        body_velocity_x=np.asarray([100.0, 20.0, 80.0], dtype=np.float32),
        command_progress=np.asarray([100.0, 20.0, 80.0], dtype=np.float32),
        target_planar_speed=np.ones(3, dtype=np.float32),
        yaw_command_progress=np.asarray([100.0, 100.0, 20.0], dtype=np.float32),
        target_yaw_speed=np.ones(3, dtype=np.float32),
        velocity_samples=np.full(3, 100, dtype=np.int64),
        survival_steps=np.full(3, 100, dtype=np.int64),
        reset_reason=np.asarray([0, 0, 1], dtype=np.int64),
        max_steps=100,
        min_progress_ratio=0.5,
    )

    assert metrics["survival_rate"] == 2.0 / 3.0
    assert metrics["planar_progress_success_rate"] == 2.0 / 3.0
    assert metrics["yaw_progress_success_rate"] == 2.0 / 3.0
    assert metrics["progress_success_rate"] == 1.0 / 3.0
    assert metrics["admission_rate"] == 1.0 / 3.0
    assert metrics["mean_command_progress"] == 2.0 / 3.0


def test_go1_rough_first_contact_is_not_repeated_while_touching():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    cfg.terrain_curriculum = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=7, max_episode_length=20, sim_workers=1)
    try:
        state = env.backend.state
        assert not state.foot_air_time.flags.writeable
        assert not state.foot_peak_height.flags.writeable
        assert state.reward.flags.writeable
        assert state.actor_obs.flags.writeable
        action = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
        previous_contact = np.zeros_like(env.backend.state.foot_contact, dtype=bool)
        saw_persistent_contact = False
        for _ in range(8):
            env.step(action)
            state = env.backend.state
            contact = np.asarray(state.foot_contact, dtype=np.float32) > 0.0
            first_contact = np.asarray(state.first_contact, dtype=np.float32) > 0.0
            landing_force = np.asarray(state.landing_force, dtype=np.float32)
            persistent_contact = contact & previous_contact
            if np.any(persistent_contact):
                saw_persistent_contact = True
                assert not np.any(first_contact[persistent_contact])
                assert np.all(landing_force[persistent_contact] == 0.0)
            previous_contact = contact.copy()
        assert saw_persistent_contact
    finally:
        env.close()


def test_go1_velocity_push_adds_to_current_world_velocity():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = True
    cfg.push_mode = "velocity"
    cfg.push_interval_mode = "global"
    cfg.push_interval_steps = 1
    cfg.push_velocity_ranges = {
        "x": (0.2, 0.2),
        "y": (-0.3, -0.3),
        "z": (0.4, 0.4),
        "roll": (-0.1, -0.1),
        "pitch": (0.2, 0.2),
        "yaw": (0.3, 0.3),
    }
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=11, max_episode_length=20, sim_workers=1)
    try:
        initial_linear = np.asarray([[0.5, -0.25, 0.1]], dtype=np.float32)
        initial_angular = np.asarray([[0.15, 0.05, -0.2]], dtype=np.float32)
        env.backend.set_base_velocities([0], initial_linear, initial_angular)
        env._apply_pushes()
        np.testing.assert_allclose(env.backend.state.base_linear_velocity, initial_linear, atol=1.0e-5)
        np.testing.assert_allclose(env.backend.state.base_angular_velocity, initial_angular, atol=1.0e-5)
        env.backend.refresh()
        np.testing.assert_allclose(
            env.backend.state.base_linear_velocity,
            initial_linear + np.asarray([[0.2, -0.3, 0.4]], dtype=np.float32),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            env.backend.state.base_angular_velocity,
            initial_angular + np.asarray([[-0.1, 0.2, 0.3]], dtype=np.float32),
            atol=1.0e-5,
        )
        np.testing.assert_array_equal(env._push_count, np.asarray([1], dtype=np.int64))
    finally:
        env.close()


def test_go1_rough_play_uses_rough_terrain_grid():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1", play=True)
    cfg.domain_randomization.enabled = False
    env = Go1VelocityEnv(cfg, num_envs=4, device="cpu", seed=5, max_episode_length=4)
    try:
        assert env._runtime_terrain_node is not None
        assert env._runtime_terrain_node.box_count == 514
        assert env._runtime_terrain_node.heightfield_count == 40
        assert env._spawn_grid_shape == (10, 7)
        assert env._spawn_origins.shape == (70, 3)
        assert np.all(env._spawn_env_levels >= 0)
        assert np.all(env._spawn_env_levels < 10)
        assert np.all(env._spawn_type_cols >= 0)
        assert np.all(env._spawn_type_cols < 7)
    finally:
        env.close()


def test_go1_native_batch_arrays_track_actions_and_state():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=8)
    try:
        env.reset(seed=123)
        first_action = np.linspace(-0.2, 0.2, env.num_actions, dtype=np.float32).reshape(1, -1)
        first_state = env.step(first_action)
        np.testing.assert_allclose(env.backend.state.previous_action, 0.0)
        np.testing.assert_allclose(env.backend.state.last_action, first_action)
        np.testing.assert_allclose(first_state.obs["actor"][:, 33:45], first_action)

        second_action = -first_action
        env.step(second_action)
        np.testing.assert_allclose(env.backend.state.previous_action, first_action)
        np.testing.assert_allclose(env.backend.state.last_action, second_action)
        action_rate_index = env._reward_term_names.index("action_rate_l2")
        np.testing.assert_allclose(
            env.backend.state.reward_terms[:, action_rate_index],
            cfg.rewards.action_rate_l2 * np.sum(np.square(second_action - first_action), axis=1),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        batch_state = env.backend.state
        assert np.isfinite(batch_state.base_position).all()
        assert np.isfinite(batch_state.base_quaternion).all()
        assert np.isfinite(batch_state.joint_position).all()
        assert "link_position" in env.cfg["task_runtime"]["array_names"]
        assert "link_quaternion" in env.cfg["task_runtime"]["array_names"]
        assert batch_state.link_position.shape == (1, len(env._batch_link_names), 3)
        assert batch_state.link_quaternion.shape == (1, len(env._batch_link_names), 4)
    finally:
        env.close()


def test_go1_parallel_batch_extraction_matches_serial_results():
    def make_env(workers: int):
        cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
        cfg.observations.actor_noise = False
        cfg.push_enabled = False
        cfg.domain_randomization.enabled = False
        cfg.terrain_curriculum = False
        return Go1VelocityEnv(
            cfg,
            num_envs=4,
            device="cpu",
            seed=17,
            max_episode_length=32,
            sim_workers=workers,
            context=gobot.app.create_context(),
        )

    serial = make_env(1)
    parallel = make_env(4)
    try:
        actions = np.linspace(-0.25, 0.25, serial.num_envs * serial.num_actions, dtype=np.float32).reshape(
            serial.num_envs,
            serial.num_actions,
        )
        serial_state = serial.step(actions)
        parallel_state = parallel.step(actions)
        np.testing.assert_allclose(parallel_state.obs["actor"], serial_state.obs["actor"], rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(parallel_state.obs["critic"], serial_state.obs["critic"], rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(parallel_state.reward, serial_state.reward, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_array_equal(parallel_state.terminated, serial_state.terminated)
        np.testing.assert_array_equal(parallel_state.truncated, serial_state.truncated)
        np.testing.assert_allclose(
            parallel.backend.state.foot_contact_force,
            serial.backend.state.foot_contact_force,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            parallel.backend.state.height_scan,
            serial.backend.state.height_scan,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    finally:
        serial.close()
        parallel.close()


def test_go1_native_base_mass_randomization_preserves_reset_state_distribution():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = True
    cfg.domain_randomization.randomize_base_mass = True
    cfg.domain_randomization.random_com = False
    cfg.domain_randomization.randomize_kp = False
    cfg.domain_randomization.randomize_kd = False
    cfg.domain_randomization.encoder_bias_range = (0.0, 0.0)
    env = Go1VelocityEnv(cfg, num_envs=16, device="cpu", seed=42, max_episode_length=64, collect_step_extras=True)
    try:
        mass_delta = env.backend.state.base_mass_delta.copy()
        assert float(np.std(mass_delta)) > 0.1
        env.reset(seed=42)
        np.testing.assert_array_equal(env.backend.state.base_mass_delta, mass_delta)
        actions = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
        for _ in range(4):
            state = env.step(actions)
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_rsl_rl_wrapper_keeps_core_env_numpy():
    torch = _require_torch()

    class DummyEnv:
        num_envs = 2
        num_actions = 3
        num_obs = 4
        num_privileged_obs = 5
        cfg = {}
        cfg_obj = object()
        seed = 1
        max_episode_length = 10

        def __init__(self):
            self.common_step_counter = 0
            self.loaded_training_state = None
            self.episode_length_buf = np.zeros(2, dtype=np.int64)
            self.state = BatchEnvState(
                obs={"actor": np.zeros((2, 4), dtype=np.float32), "critic": np.ones((2, 5), dtype=np.float32)},
                reward=np.zeros(2, dtype=np.float32),
                terminated=np.zeros(2, dtype=bool),
                truncated=np.zeros(2, dtype=bool),
                info={"steps": self.episode_length_buf},
            )

        def reset(self, seed=None):
            assert seed == 123
            return self.state.obs, {}

        def step(self, actions):
            assert isinstance(actions, np.ndarray)
            self.state.reward[:] = 1.0
            self.state.info["log"] = {"x": np.asarray(1.0, dtype=np.float32)}
            self.state.info["reward_terms"] = {"term": np.ones(2, dtype=np.float32)}
            self.state.info["final_observation"] = {
                "actor": np.full((2, 4), 2.0, dtype=np.float32),
                "critic": np.full((2, 5), 3.0, dtype=np.float32),
            }
            self.state.info["_final_observation"] = np.asarray([False, True], dtype=bool)
            return self.state

        def training_state_dict(self):
            return {"common_step_counter": 123, "terrain_curriculum": {"version": 1}}

        def load_training_state_dict(self, state):
            self.loaded_training_state = dict(state)
            self.common_step_counter = int(state["common_step_counter"])
            return {"terrain_curriculum": "exact"}

        def close(self):
            pass

    env = DummyEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    wrapper.episode_length_buf = torch.asarray([3, 7], dtype=torch.long)
    np.testing.assert_array_equal(env.episode_length_buf, [3, 7])
    obs = wrapper.reset(seed=123)
    assert isinstance(obs["actor"], torch.Tensor)
    obs, reward, done, extras = wrapper.step(torch.zeros((2, 3), dtype=torch.float32))
    assert obs["critic"].shape == (2, 5)
    assert reward.tolist() == [1.0, 1.0]
    assert done.tolist() == [False, False]
    assert extras["log"]["x"] == 1.0
    assert "reward_terms" not in extras
    assert "final_observation" not in extras
    assert "_final_observation" not in extras
    assert isinstance(env.state.obs["actor"], np.ndarray)
    assert wrapper.training_state_dict()["common_step_counter"] == 123
    result = wrapper.load_training_state_dict({"common_step_counter": 456})
    assert result == {"terrain_curriculum": "exact"}
    assert env.loaded_training_state == {"common_step_counter": 456}
    assert env.common_step_counter == 456

    env.rsl_rl_include_reward_terms = True
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    _, _, _, extras = wrapper.step(torch.zeros((2, 3), dtype=torch.float32))
    assert extras["reward_terms"]["term"].tolist() == [1.0, 1.0]


def test_rsl_rl_wrapper_preserves_device_native_torch_buffers():
    torch = _require_torch()

    class TorchEnv:
        num_envs = 2
        num_actions = 3
        num_obs = 4
        num_privileged_obs = 5
        cfg = {}
        cfg_obj = object()
        seed = 1
        max_episode_length = 10
        accepts_device_actions = True

        def __init__(self):
            self.episode_length_buf = torch.zeros(2, dtype=torch.long)
            self.state = BatchEnvState(
                obs={
                    "actor": torch.zeros((2, 4), dtype=torch.float32),
                    "critic": torch.ones((2, 5), dtype=torch.float32),
                },
                reward=torch.zeros(2, dtype=torch.float32),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                info={"steps": self.episode_length_buf},
            )
            self.last_actions = None

        def reset(self, seed=None):
            return self.state.obs, {}

        def step(self, actions):
            self.last_actions = actions
            self.state.reward.fill_(2.0)
            self.state.truncated[1] = True
            self.state.info["time_outs"] = self.state.truncated
            return self.state

        def close(self):
            pass

    env = TorchEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    actions = torch.zeros((2, 3), dtype=torch.float32)
    obs, reward, done, extras = wrapper.step(actions)

    assert env.last_actions is actions
    assert obs["actor"].data_ptr() == env.state.obs["actor"].data_ptr()
    assert obs["policy"].data_ptr() == env.state.obs["actor"].data_ptr()
    assert reward.data_ptr() == env.state.reward.data_ptr()
    assert extras["time_outs"].data_ptr() == env.state.truncated.data_ptr()
    assert done.tolist() == [False, True]
    assert env.state.array_backend == "torch"
    assert env.state.device == "cpu"
    assert wrapper.memory_profile()["device_native_observations"] is True


def test_rsl_rl_wrapper_caches_alternating_device_native_observations():
    torch = _require_torch()

    class TorchEnv:
        num_envs = 2
        num_actions = 1
        num_obs = 2
        num_privileged_obs = 3
        cfg = {}
        cfg_obj = object()
        seed = 1
        max_episode_length = 10
        accepts_device_actions = True

        def __init__(self):
            self.episode_length_buf = torch.zeros(2, dtype=torch.long)
            self.buffers = (
                {
                    "actor": torch.zeros((2, 2), dtype=torch.float32),
                    "critic": torch.zeros((2, 3), dtype=torch.float32),
                },
                {
                    "actor": torch.ones((2, 2), dtype=torch.float32),
                    "critic": torch.ones((2, 3), dtype=torch.float32),
                },
            )
            self.index = 0
            self.state = BatchEnvState(
                obs=self.buffers[0],
                reward=torch.zeros(2),
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                info={"steps": self.episode_length_buf},
            )

        def reset(self, seed=None):
            return self.state.obs, {}

        def step(self, actions):
            del actions
            self.index = 1 - self.index
            self.state.obs = self.buffers[self.index]
            return self.state

        def close(self):
            pass

    env = TorchEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    actions = torch.zeros((2, 1), dtype=torch.float32)
    first = wrapper.reset()
    first_values = first["actor"].clone()
    second, _, _, _ = wrapper.step(actions)

    torch.testing.assert_close(first["actor"], first_values)
    assert second["actor"].data_ptr() != first["actor"].data_ptr()

    third, _, _, _ = wrapper.step(actions)
    assert third is first
    assert wrapper.memory_profile()["native_obs_tensordicts"] == 2


def test_compiled_scene_artifact_validates_robot_prefixes():
    content = "<mujoco/>"
    artifact_mapping = {
        "schema_version": 1,
        "backend": "MuJoCoCpu",
        "format": "mjcf",
        "content": content,
        "content_digest": "fnv1a64:f4053c9f5a9db5a9",
        "backend_version": "3.10.0",
        "dimensions": {"nq": 7, "nv": 6, "nu": 12},
        "robot_names": ["go1"],
        "robot_prefixes": ["go1_"],
        "terrain_geom_groups": [5],
    }
    artifact = CompiledSceneArtifact.from_mapping(artifact_mapping)
    assert artifact.robot_prefix("go1") == "go1_"
    assert artifact.dimensions["nq"] == 7
    assert artifact.content_digest == artifact.digest
    assert artifact.terrain_geom_groups == (5,)
    try:
        CompiledSceneArtifact.from_mapping(
            {"schema_version": 2, "format": "mjcf", "content": content}
        )
        raise AssertionError("unsupported artifact schema should fail")
    except ValueError as error:
        assert "schema" in str(error)
    try:
        CompiledSceneArtifact.from_mapping(
            {**artifact_mapping, "content": "<mujoco model='changed'/>"}
        )
        raise AssertionError("modified artifact content should fail digest validation")
    except ValueError as error:
        assert "digest mismatch" in str(error)
    availability = MuJoCoWarpProvider.availability()
    assert isinstance(availability.available, bool)
    assert availability.available or availability.reason


def test_go1_current_policy_contract():
    cfg = go1_cfg.go1_velocity_cfg(project_path="/tmp/go1")
    actor_schema = velocity_actor_observation_schema(
        len(cfg.joint_names),
        go1_playback.TERRAIN_SCAN_DIM,
    )
    critic_schema = velocity_critic_observation_schema(
        len(cfg.joint_names),
        go1_playback.TERRAIN_SCAN_DIM,
        len(cfg.foot_names),
    )
    assert cfg.name == go1_playback.TASK_NAME
    assert actor_schema.dim == 235
    assert critic_schema.dim == 259
    assert go1_playback.ACTOR_OBS_SPEC.names == actor_schema.names
    assert go1_playback.ACTION_SPEC.names == tuple(cfg.joint_names)
    assert cfg.physics_dt == 0.005
    assert cfg.decimation == 4
    assert cfg.action_clip is None
    np.testing.assert_allclose(cfg.default_joint_pos, go1_profile.GO1_DEFAULT_JOINT_POS)


def test_go1_playback_prefers_onnx_then_falls_back_to_torch_checkpoint():
    with tempfile.TemporaryDirectory() as temporary_directory:
        policies = Path(temporary_directory) / "policies"
        policies.mkdir()
        (policies / "go1_velocity.pt").touch()

        sentinel = object()
        original_torch_policy = go1_playback.TorchPolicy
        original_policy_env = os.environ.pop(go1_playback.POLICY_ENV, None)
        try:
            go1_playback.TorchPolicy = lambda path: sentinel
            script = object.__new__(go1_playback.Script)
            script.context = type("FakeContext", (), {"project_path": temporary_directory})()
            assert script._load_policy() is sentinel
        finally:
            go1_playback.TorchPolicy = original_torch_policy
            if original_policy_env is not None:
                os.environ[go1_playback.POLICY_ENV] = original_policy_env


def test_go1_playback_loads_optional_run_policy_and_switches_only_when_requested():
    with tempfile.TemporaryDirectory() as temporary_directory:
        policies = Path(temporary_directory) / "policies"
        policies.mkdir()
        run_path = policies / "go1_velocity_run.pt"
        run_path.touch()

        base_policy = object()
        run_policy = object()
        original_torch_policy = go1_playback.TorchPolicy
        original_run_policy_env = os.environ.pop(go1_playback.RUN_POLICY_ENV, None)
        try:
            go1_playback.TorchPolicy = lambda path: run_policy
            script = object.__new__(go1_playback.Script)
            script.context = type("FakeContext", (), {"project_path": temporary_directory})()
            script.policy = base_policy
            script.run_policy = script._load_run_policy()

            script.run_requested = False
            assert script._active_policy() is base_policy
            script.run_requested = True
            assert script._active_policy() is run_policy
        finally:
            go1_playback.TorchPolicy = original_torch_policy
            if original_run_policy_env is not None:
                os.environ[go1_playback.RUN_POLICY_ENV] = original_run_policy_env


def test_go1_playback_builds_current_observation():
    cfg = go1_cfg.go1_velocity_cfg(project_path="/tmp/go1")
    script = object.__new__(go1_playback.Script)
    script.command = [0.3, -0.2, 0.1]
    script.last_action = [0.0] * len(go1_playback.JOINT_NAMES)
    script.default_pos = [float(value) for value in cfg.default_joint_pos]
    script.height_scan_max_distance = 5.0
    state = {
        "links": [
            {
                "name": go1_playback.BASE_LINK,
                "global_transform": {
                    "position": [0.0, 0.0, 0.5],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
                "linear_velocity": [0.1, 0.2, 0.3],
                "angular_velocity": [0.4, 0.5, 0.6],
            }
        ],
        "joints": [
            {"name": name, "position": script.default_pos[index], "velocity": 0.0}
            for index, name in enumerate(go1_playback.JOINT_NAMES)
        ],
        "sensors": [
            {
                "name": go1_playback.IMU_SENSOR,
                "values": [1.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            },
            {
                "name": go1_playback.TERRAIN_SCAN_SENSOR,
                "values": [0.0] * go1_playback.TERRAIN_SCAN_DIM,
            }
        ],
    }
    observation = np.asarray(script._observation(state), dtype=np.float32)
    assert observation.shape == (go1_playback.ACTOR_OBS_SPEC.dim,)
    np.testing.assert_allclose(observation[0:3], (0.1, 0.2, 0.3), atol=1.0e-6)
    np.testing.assert_allclose(observation[3:6], (0.4, 0.5, 0.6), atol=1.0e-6)
    np.testing.assert_allclose(observation[9:21], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(observation[45:48], script.command, atol=1.0e-6)


def test_go1_playback_reads_authored_terrain_origins():
    terrain = gobot.create_node("Terrain3D", "terrain")
    terrain.spawn_origins = [(4.0, -2.0, 0.3), (0.0, 0.0, 0.1)]

    class FakeTerrainWorld:
        def __init__(self):
            self.name = "terrain_world"
            self.children = [terrain]

        def find(self, name):
            return next((child for child in self.children if child.name == name), None)

    class FakeRoot:
        def __init__(self):
            self.name = "root"
            self.children = [FakeTerrainWorld()]

        def find(self, name):
            return next((child for child in self.children if child.name == name), None)

    root = FakeRoot()
    script = object.__new__(go1_playback.Script)
    script.reset_base_height = 0.278
    script.get_root = lambda: root

    np.testing.assert_allclose(
        script._terrain_spawn_origins(),
        [[4.0, -2.0, 0.3], [0.0, 0.0, 0.1]],
    )
    np.testing.assert_allclose(script._resolve_reset_base_position(), (0.0, 0.0, 0.378))


def test_go1_playback_reports_nearest_terrain_cell():
    script = object.__new__(go1_playback.Script)
    script.terrain_rows = 2
    script.terrain_type_names = ["flat", "slope"]
    script.terrain_origins = [
        [-4.0, -4.0, 0.0],
        [-4.0, 4.0, 0.0],
        [4.0, -4.0, 0.0],
        [4.0, 4.0, 0.0],
    ]

    assert script._terrain_cell([3.5, 4.5, 0.2]) == (1, "slope", 1.0)


def test_go1_playback_fall_detection_uses_terrain_relative_clearance():
    def state(*, world_z: float, clearance: float, roll_degrees: float = 0.0):
        half_roll = np.deg2rad(roll_degrees) * 0.5
        scan = [clearance] * go1_playback.TERRAIN_SCAN_DIM
        return {
            "links": [
                {
                    "name": go1_playback.BASE_LINK,
                    "global_transform": {
                        "position": [0.0, 0.0, world_z],
                        "quaternion": [
                            float(np.cos(half_roll)),
                            float(np.sin(half_roll)),
                            0.0,
                            0.0,
                        ],
                    },
                }
            ],
            "sensors": [
                {
                    "name": go1_playback.TERRAIN_SCAN_SENSOR,
                    "values": scan,
                }
            ],
        }

    script = object.__new__(go1_playback.Script)
    reset_count = 0

    def reset():
        nonlocal reset_count
        reset_count += 1

    script.reset = reset

    assert not script._reset_if_fallen(state(world_z=-0.4, clearance=0.28))
    assert reset_count == 0
    assert script._reset_if_fallen(state(world_z=0.3, clearance=0.1))
    assert reset_count == 1
    assert script._reset_if_fallen(
        state(world_z=0.3, clearance=0.28, roll_degrees=75.0)
    )
    assert reset_count == 2


def test_go1_stale_policy_rejects_playback():
    script = object.__new__(go1_playback.Script)

    def reject_policy():
        raise RuntimeError("checkpoint has no policy manifest")

    script._load_policy = reject_policy
    try:
        script._load_and_validate_policy()
    except RuntimeError as error:
        assert "Play requires a current manifest-backed policy" in str(error)
    else:
        raise AssertionError("stale policy should reject Go1 Play")

    assert script.policy is None
    assert script.manifest is None
    assert "no policy manifest" in script.policy_error


def test_go1_w_key_updates_forward_velocity_command():
    class FakeInput:
        has_control_focus = True

        @staticmethod
        def is_key_pressed(_key):
            return False

        @staticmethod
        def is_key_held(key):
            return key == "W"

    script = object.__new__(go1_playback.Script)
    script.context = type("FakeContext", (), {"input": FakeInput()})()
    script.command = [0.0, 0.0, 0.0]
    script.command_target = [0.0, 0.0, 0.0]

    assert script._update_keyboard_command(0.02) is False
    assert script.command[0] > 0.0
    assert script.command[1] == 0.0
    assert script.command[2] == 0.0


def test_go1_diagonal_keyboard_command_preserves_planar_speed_limit():
    class FakeInput:
        has_control_focus = True

        @staticmethod
        def is_key_pressed(_key):
            return False

        @staticmethod
        def is_key_held(key):
            return key in {"W", "E"}

    script = object.__new__(go1_playback.Script)
    script.context = type("FakeContext", (), {"input": FakeInput()})()
    script.command = [0.0, 0.0, 0.0]
    script.command_target = [0.0, 0.0, 0.0]

    assert script._update_keyboard_command(1.0) is False
    np.testing.assert_allclose(
        script.command,
        [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
        atol=1.0e-7,
    )
    assert np.linalg.norm(script.command[:2]) <= 1.0


def test_go1_sprint_keyboard_command_uses_trained_run_limit():
    class FakeInput:
        has_control_focus = True

        @staticmethod
        def is_key_pressed(_key):
            return False

        @staticmethod
        def is_key_held(key):
            return key in {"W", "LeftShift"}

    script = object.__new__(go1_playback.Script)
    script.context = type("FakeContext", (), {"input": FakeInput()})()
    script.command = [0.0, 0.0, 0.0]
    script.command_target = [0.0, 0.0, 0.0]

    assert script._update_keyboard_command(1.0) is False
    assert script.command == [3.0, 0.0, 0.0]
    assert script.run_requested is True


def test_go1_zero_command_still_runs_manifest_policy():
    cfg = go1_cfg.go1_velocity_cfg(project_path="/tmp/go1")

    class FakePolicy:
        obs_dim = go1_playback.ACTOR_OBS_SPEC.dim
        action_dim = len(go1_playback.JOINT_NAMES)

        def __init__(self):
            self.calls = 0

        def action(self, observation):
            self.calls += 1
            assert len(observation) == self.obs_dim
            return [0.5] * self.action_dim

    class FakeContext:
        has_world = True
        simulation_time = 0.0
        input = None

    class FakeJoint:
        lower_limit = -3.0
        upper_limit = 3.0

        def __init__(self, name, position):
            self.name = name
            self.position = position
            self.target = None

        def get_runtime_state(self):
            return {"name": self.name, "position": self.position, "velocity": 0.0}

        def set_position_target(self, value):
            self.target = float(value)

    class FakeLink:
        def get_runtime_state(self):
            return {
                "name": go1_playback.BASE_LINK,
                "global_transform": {
                    "position": [0.0, 0.0, 0.3],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }

    class FakeTerrainSensor:
        def get_runtime_state(self):
            return {
                "name": go1_playback.TERRAIN_SCAN_SENSOR,
                "values": [0.28] * go1_playback.TERRAIN_SCAN_DIM,
            }

    class FakeImuSensor:
        def get_runtime_state(self):
            return {
                "name": go1_playback.IMU_SENSOR,
                "values": [1.0, 0.0, 0.0, 0.0] + [0.0] * 9,
            }

    script = object.__new__(go1_playback.Script)
    script.context = FakeContext()
    script.playing = True
    script.world_controls_ready = True
    script.ticks = 0
    script.decimation = 4
    script.print_every_ticks = 1000
    script.command = [0.0, 0.0, 0.0]
    script.command_target = [0.0, 0.0, 0.0]
    script.policy = FakePolicy()
    script.action_scale = [
        0.24850202579136574 if "_calf_" in name else 0.3727530386870487
        for name in go1_playback.JOINT_NAMES
    ]
    script.action_clip = None
    script.default_pos = [float(value) for value in cfg.default_joint_pos]
    script.height_scan_max_distance = 5.0
    script.last_action = [0.0] * len(go1_playback.JOINT_NAMES)
    script.last_targets = list(script.default_pos)
    script.base_link = FakeLink()
    script.imu = FakeImuSensor()
    script.terrain_scan = FakeTerrainSensor()
    script.robot = type("FakeRobot", (), {"name": go1_playback.ROBOT})()
    script.joints = [
        FakeJoint(name, script.default_pos[index])
        for index, name in enumerate(go1_playback.JOINT_NAMES)
    ]
    script._physics_process(0.005)
    assert script.policy.calls == 1
    assert script.last_action == [0.5] * len(go1_playback.JOINT_NAMES)
    assert all(joint.target is not None for joint in script.joints)


def main():
    tests = [
        test_repo_imports_use_one_gobot_package,
        test_batch_env_state_contract,
        test_batch_env_clears_stale_final_observation_info,
        test_task_runtime_metadata_is_public_task_summary,
        test_go1_current_policy_contract,
        test_go1_playback_prefers_onnx_then_falls_back_to_torch_checkpoint,
        test_go1_playback_loads_optional_run_policy_and_switches_only_when_requested,
        test_go1_playback_builds_current_observation,
        test_go1_playback_reads_authored_terrain_origins,
        test_go1_playback_reports_nearest_terrain_cell,
        test_go1_playback_fall_detection_uses_terrain_relative_clearance,
        test_go1_stale_policy_rejects_playback,
        test_go1_w_key_updates_forward_velocity_command,
        test_go1_diagonal_keyboard_command_preserves_planar_speed_limit,
        test_go1_sprint_keyboard_command_uses_trained_run_limit,
        test_go1_zero_command_still_runs_manifest_policy,
        test_go1_env_applies_mujoco_solver_settings,
        test_go1_robot_scene_matches_mujoco_contract,
        test_go1_rough_env_reset_step_shapes,
        test_go1_reward_and_observation_formulas_match_task_contract,
        test_go1_run_progress_reward_does_not_reward_standing,
        test_go1_training_configuration_matches_task_contract,
        test_go1_run_training_profile_reserves_high_speed_forward_commands,
        test_go1_bound_gait_score_requires_pair_sync_speed_and_run_command,
        test_go1_bound_gait_score_matches_torch_backend,
        test_go1_footfall_patterns_distinguish_bound_trot_and_flight,
        test_go1_gait_metrics_are_defined_without_surviving_samples,
        test_go1_terrain_curriculum_checkpoint_restores_exact_assignments,
        test_go1_terrain_curriculum_checkpoint_resamples_across_batch_sizes,
        test_go1_checkpoint_admission_requires_survival_and_progress,
        test_go1_rough_first_contact_is_not_repeated_while_touching,
        test_go1_rough_play_uses_rough_terrain_grid,
        test_go1_native_batch_arrays_track_actions_and_state,
        test_go1_parallel_batch_extraction_matches_serial_results,
        test_go1_native_base_mass_randomization_preserves_reset_state_distribution,
        test_go1_velocity_push_adds_to_current_world_velocity,
        test_rsl_rl_wrapper_keeps_core_env_numpy,
        test_rsl_rl_wrapper_preserves_device_native_torch_buffers,
        test_rsl_rl_wrapper_caches_alternating_device_native_observations,
        test_compiled_scene_artifact_validates_robot_prefixes,
    ]
    skipped = 0
    for test in tests:
        try:
            test()
        except OptionalDependencyUnavailable as error:
            skipped += 1
            print(error)
    if skipped == len(tests):
        sys.exit(OPTIONAL_DEPENDENCY_SKIP_CODE)


if __name__ == "__main__":
    main()
