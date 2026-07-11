import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.go1.scripts import go1 as go1_playback
from examples.go1.train import go1_velocity_cfg as go1_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
import gobot
from gobot.rl import (
    BatchEnvState,
    CpuBatchEnv,
    TaskRuntimeMetadata,
)
from gobot.rl.locomotion import (
    HeightScan,
    LocomotionDomainRandomization,
    LocomotionDomainRandomizationCfg,
    LocomotionRewardContext,
    TerrainSpawn,
    action_rate_l2,
    dispatch_reward_terms,
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


def test_locomotion_common_helpers():
    scan = HeightScan(dim=4, max_distance=2.0)
    assert np.allclose(scan.normalize(np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32)), [0.0, 0.5, 1.0, 2.0])
    _assert_runtime_error("expected trailing dimension 4", lambda: scan.normalize(np.zeros(3, dtype=np.float32)))

    spawn = TerrainSpawn(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        levels=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )
    assert spawn.update_limit(0.5, reset_reason=2, survival=0.2, distance=0.0, expected_distance=1.0) > 0.5
    assert spawn.update_limit(0.5, reset_reason=1, survival=0.1, distance=0.0, expected_distance=1.0) < 0.5

    dr = LocomotionDomainRandomization(
        LocomotionDomainRandomizationCfg(encoder_bias_range=(-0.1, 0.1), reset_lin_vel_ranges={"x": (-1.0, 1.0)}),
        num_actions=2,
    )
    payload = dr.reset_payload(np.asarray([0, 2], dtype=np.int64), np.random.default_rng(1))
    assert payload["base_linear_velocity"].shape == (2, 3)
    assert payload["encoder_bias"].shape == (2, 2)

    ctx = LocomotionRewardContext(
        dt=0.02,
        command=np.zeros((2, 3), dtype=np.float32),
        base_lin_vel_b=np.zeros((2, 3), dtype=np.float32),
        base_ang_vel_b=np.zeros((2, 3), dtype=np.float32),
        projected_gravity=np.asarray([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]], dtype=np.float32),
        joint_pos=np.zeros((2, 2), dtype=np.float32),
        joint_vel=np.zeros((2, 2), dtype=np.float32),
        actions=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        previous_actions=np.zeros((2, 2), dtype=np.float32),
    )
    reward, terms = dispatch_reward_terms(ctx, {"action_rate_l2": action_rate_l2}, {"action_rate_l2": -0.1})
    assert np.allclose(reward, [-0.002, -0.002])
    assert "action_rate_l2" in terms


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
    assert terrain_scan["visualize_debug"] is False
    for foot in ("FR", "FL", "RR", "RL"):
        foot_height_scan = _node_by_name(nodes, f"{foot}_foot_height_scan", "TerrainHeightSensor3D")[
            "properties"
        ]
        assert foot_height_scan["reduction_mode"] == "Min"
        assert len(foot_height_scan["sample_offsets"]) == 5
        np.testing.assert_allclose(_matrix_storage(foot_height_scan["sample_offsets"][0]), (0.0, 0.0, 0.0))
        assert _node_by_name(nodes, f"{foot}_foot_contact", "ContactSensor3D")["properties"]["radius"] == np.float32(
            0.03
        )


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
        assert env.task_runtime_metadata.version == go1_cfg.GO1_TASK_VERSION
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
        np.testing.assert_allclose(env.backend.state.default_joint_position, go1_cfg.GO1_DEFAULT_JOINT_POS)
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
            np.asarray(go1_cfg.GO1_DEFAULT_JOINT_POS, dtype=np.float32).reshape(1, -1) - encoder_bias,
        )
        actor_obs, _ = env._build_generic_velocity_observations(env.backend.state)
        np.testing.assert_allclose(
            actor_obs[:, 9 : 9 + env.num_actions],
            env.backend.state.joint_position
            - np.asarray(go1_cfg.GO1_DEFAULT_JOINT_POS, dtype=np.float32).reshape(1, -1),
        )
        np.copyto(env.backend.state.encoder_bias, original_encoder_bias)

        assert env.cfg_obj.command.ranges.lin_vel_x == (-1.0, 1.0)
        env.set_training_progress(119_999)
        assert env.cfg_obj.command.ranges.lin_vel_x == (-1.0, 1.0)
        env.set_training_progress(120_000)
        assert env.cfg_obj.command.ranges.lin_vel_x == (-1.5, 2.0)
        env.set_training_progress(240_000)
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


def test_go1_rough_first_contact_is_not_repeated_while_touching():
    cfg = go1_cfg.go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    cfg.terrain_curriculum = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=7, max_episode_length=20, sim_workers=1)
    try:
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
            self.state = BatchEnvState(
                obs={"actor": np.zeros((2, 4), dtype=np.float32), "critic": np.ones((2, 5), dtype=np.float32)},
                reward=np.zeros(2, dtype=np.float32),
                terminated=np.zeros(2, dtype=bool),
                truncated=np.zeros(2, dtype=bool),
                info={"steps": np.zeros(2, dtype=np.int64)},
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

        def close(self):
            pass

    env = DummyEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
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

    env.rsl_rl_include_reward_terms = True
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    _, _, _, extras = wrapper.step(torch.zeros((2, 3), dtype=torch.float32))
    assert extras["reward_terms"]["term"].tolist() == [1.0, 1.0]


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
    np.testing.assert_allclose(cfg.default_joint_pos, go1_cfg.GO1_DEFAULT_JOINT_POS)


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


def test_go1_stale_policy_falls_back_to_terrain_preview():
    script = object.__new__(go1_playback.Script)

    def reject_policy():
        raise RuntimeError("checkpoint has no policy manifest")

    script._load_policy = reject_policy
    script._load_policy_or_enable_preview()
    script._configure_runtime_profile()

    assert script.policy is None
    assert script.manifest is None
    assert "no policy manifest" in script.policy_error
    assert script.physics_dt == go1_playback.GO1_PHYSICS_DT
    assert script.decimation == go1_playback.GO1_DECIMATION
    np.testing.assert_allclose(script.default_pos, go1_cfg.GO1_DEFAULT_JOINT_POS)


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
                "values": [0.0] * go1_playback.TERRAIN_SCAN_DIM,
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
        test_locomotion_common_helpers,
        test_go1_current_policy_contract,
        test_go1_playback_prefers_onnx_then_falls_back_to_torch_checkpoint,
        test_go1_playback_builds_current_observation,
        test_go1_playback_reads_authored_terrain_origins,
        test_go1_stale_policy_falls_back_to_terrain_preview,
        test_go1_zero_command_still_runs_manifest_policy,
        test_go1_env_applies_mujoco_solver_settings,
        test_go1_robot_scene_matches_mujoco_contract,
        test_go1_rough_env_reset_step_shapes,
        test_go1_rough_first_contact_is_not_repeated_while_touching,
        test_go1_rough_play_uses_rough_terrain_grid,
        test_go1_native_batch_arrays_track_actions_and_state,
        test_go1_native_base_mass_randomization_preserves_reset_state_distribution,
        test_go1_velocity_push_adds_to_current_world_velocity,
        test_rsl_rl_wrapper_keeps_core_env_numpy,
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
