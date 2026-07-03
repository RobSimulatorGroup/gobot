import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from _gobot_test_import import prefer_build_gobot

prefer_build_gobot()

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "build/python", REPO_ROOT / "python", REPO_ROOT):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
    sys.path.insert(0, path_string)

from benchmark import go1_velocity_benchmark
from examples.go1.scripts import compare_unilab_gobot_go1_rough as go1_compare
from examples.go1.scripts import go1 as go1_playback
from examples.go1.train import go1_velocity_cfg as go1_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
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


def _require_go1_velocity_train():
    try:
        from examples.go1.train import go1_velocity_train
    except ImportError as error:
        missing = getattr(error, "name", "")
        if missing in {"torch", "rsl_rl"}:
            raise OptionalDependencyUnavailable("Go1 training dependencies are unavailable; skipping train cfg test") from error
        raise
    return go1_velocity_train


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


def test_repo_imports_use_source_python_and_build_native_extension():
    import gobot
    import gobot._core
    import gobot.rl

    assert Path(gobot.__file__).resolve().is_relative_to((REPO_ROOT / "python").resolve())
    assert Path(gobot.rl.__file__).resolve().is_relative_to((REPO_ROOT / "python").resolve())
    assert Path(gobot._core.__file__).resolve().is_relative_to((REPO_ROOT / "build/python").resolve())


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


def test_go1_unilab_cfg_profiles_and_playback_schemas():
    cfg = go1_cfg.go1_velocity_cfg("go1_rough", project_path="/tmp/go1")
    assert cfg.name == "gobot_go1_unilab_rough"
    assert cfg.task_profile == "unilab_rough"
    assert len(cfg.joint_names) == 12
    assert cfg.foot_names == go1_cfg.GO1_UNILAB_FOOT_NAMES
    assert cfg.foot_link_names == go1_cfg.GO1_UNILAB_FOOT_LINK_NAMES
    assert cfg.observations.height_scan_sensor == "terrain_scan"
    assert cfg.action_clip == 100.0
    assert cfg.physics_dt == 0.005
    assert cfg.decimation == 4
    assert cfg.mujoco_solver_settings == go1_cfg.GO1_UNILAB_MUJOCO_SOLVER_SETTINGS
    assert cfg.kp == 35.0
    assert cfg.kd == 0.5
    assert cfg.base_clearance == 0.32
    assert cfg.illegal_contact.ground_force_threshold == 1.0
    assert cfg.randomize_rough_reset_pose
    assert cfg.use_unilab_reset_rng
    assert cfg.command.heading_command
    assert cfg.command.resampling_time_range == (10.0, 10.0)
    assert cfg.command.ranges.lin_vel_x == (-1.0, 1.0)
    assert cfg.command.ranges.lin_vel_y == (-1.0, 1.0)
    assert cfg.command.ranges.ang_vel_z == (-1.0, 1.0)
    assert cfg.command.zero_small_xy_threshold == 0.08
    assert cfg.domain_randomization.enabled
    assert cfg.domain_randomization.randomize_base_mass
    assert cfg.domain_randomization.added_mass_range == (-1.0, 3.0)
    assert cfg.domain_randomization.random_com
    assert cfg.domain_randomization.randomize_kp
    assert cfg.domain_randomization.kp_multiplier_range == (0.9, 1.1)
    assert cfg.domain_randomization.randomize_kd
    assert cfg.domain_randomization.kd_multiplier_range == (0.9, 1.1)
    assert cfg.domain_randomization.encoder_bias_range == (0.0, 0.0)
    assert cfg.push_enabled
    assert cfg.push_interval_steps == 625
    assert cfg.push_interval_mode == "global"
    assert cfg.terrain_out_of_bounds
    assert cfg.unilab_rewards.scales["feet_gait"] == 0.5
    assert cfg.unilab_rewards.scales["tracking_lin_vel"] == 3.0
    np.testing.assert_allclose(
        np.asarray(cfg.default_joint_pos, dtype=np.float32),
        np.asarray(
            [
                0.0,
                0.9,
                -1.8,
                0.0,
                0.9,
                -1.8,
                0.0,
                1.0,
                -1.8,
                0.0,
                1.0,
                -1.8,
            ],
            dtype=np.float32,
        ),
    )

    actor_schema = velocity_actor_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM)
    critic_schema = velocity_critic_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM, len(cfg.foot_names))
    assert actor_schema.dim == 235
    assert critic_schema.dim == 259
    assert go1_playback.ACTOR_OBS_SCHEMA.names == actor_schema.names
    assert go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim == 45
    assert go1_playback.UNILAB_FLAT_ACTOR_OBS_SCHEMA.dim == 49
    assert go1_playback.FIXED_TIME_STEP == cfg.physics_dt
    assert go1_playback.DECIMATION == cfg.decimation
    assert go1_playback.RESET_BASE_POSITION[2] == cfg.base_clearance
    assert go1_playback.UNILAB_MUJOCO_SOLVER_SETTINGS == dict(go1_cfg.GO1_UNILAB_MUJOCO_SOLVER_SETTINGS)
    assert go1_playback._validate_checkpoint_schema(
        {
            "infos": {
                "gobot_go1_unilab_rough": {
                    "obs_schema_version": go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.version,
                    "obs_names": go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.names,
                    "num_obs": 45,
                }
            }
        },
        45,
    ).version == go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.version
    assert np.allclose(go1_playback.DEFAULT_POS, cfg.default_joint_pos)

    flat = go1_cfg.go1_velocity_cfg("go1_flat", project_path="/tmp/go1")
    assert flat.name == "gobot_go1_unilab_flat"
    assert flat.task_profile == "unilab_flat"
    assert flat.scene_path == "res://go1_flat_scene.jscn"
    assert flat.base_clearance == 0.45
    assert not flat.randomize_rough_reset_pose
    assert flat.observations.height_scan_sensor is None
    assert not flat.terrain_curriculum
    assert flat.action_clip == 1.0
    assert flat.physics_dt == 0.01
    assert flat.decimation == 2
    assert flat.command.resampling_time_range == (1_000_000_000.0, 1_000_000_000.0)
    assert flat.command.ranges.lin_vel_x == (-0.6, 1.0)
    assert flat.command.ranges.lin_vel_y == (-0.4, 0.4)
    assert flat.command.ranges.ang_vel_z == (-0.8, 0.8)
    assert flat.command.zero_small_xy_threshold == 0.0
    assert flat.domain_randomization.enabled
    assert flat.domain_randomization.randomize_base_mass
    assert flat.domain_randomization.random_com
    assert not flat.domain_randomization.randomize_kp
    assert not flat.domain_randomization.randomize_kd
    assert flat.push_enabled
    assert flat.unilab_rewards.scales["base_height"] == -100.0


def test_go1_robot_scene_matches_unilab_mujoco_contract():
    scene = json.loads((REPO_ROOT / "examples/go1/go1.jscn").read_text(encoding="utf-8"))
    nodes = scene["__NODES__"]

    hip_joint = _node_by_name(nodes, "FR_hip_joint", "Joint3D")["properties"]
    assert hip_joint["damping"] == 1.0
    assert hip_joint["armature"] == np.float32(0.01)
    assert hip_joint["friction_loss"] == np.float32(0.2)
    assert hip_joint["drive_mode"] == "Position"
    assert hip_joint["drive_stiffness"] == 100.0
    assert hip_joint["force_lower_limit"] == np.float32(-23.7)
    assert hip_joint["force_upper_limit"] == np.float32(23.7)
    assert hip_joint["control_lower_limit"] == np.float32(-0.863)
    assert hip_joint["control_upper_limit"] == np.float32(0.863)

    thigh_joint = _node_by_name(nodes, "FR_thigh_joint", "Joint3D")["properties"]
    calf_joint = _node_by_name(nodes, "FR_calf_joint", "Joint3D")["properties"]
    assert thigh_joint["damping"] == 2.0
    assert thigh_joint["armature"] == np.float32(0.01)
    assert thigh_joint["friction_loss"] == np.float32(0.2)
    assert thigh_joint["drive_stiffness"] == 100.0
    assert thigh_joint["force_lower_limit"] == np.float32(-23.7)
    assert thigh_joint["force_upper_limit"] == np.float32(23.7)
    assert calf_joint["damping"] == 2.0
    assert calf_joint["armature"] == np.float32(0.01)
    assert calf_joint["friction_loss"] == np.float32(0.2)
    assert calf_joint["drive_stiffness"] == 100.0
    assert calf_joint["force_lower_limit"] == np.float32(-35.55)
    assert calf_joint["force_upper_limit"] == np.float32(35.55)

    node_names = {node["name"] for node in nodes}
    assert {"base1", "base2", "base3", "FR_hip_geom", "FR_thigh_geom", "FR_calf_geom1", "FR"} <= node_names

    base_collision = _node_by_name(nodes, "base1", "CollisionShape3D")["properties"]
    assert _matrix_storage(base_collision["friction"]) == (0.0, 0.0, 0.0)
    assert base_collision["condim"] == 1
    assert base_collision["margin"] == np.float32(0.001)

    foot_collision = _node_by_name(nodes, "FR", "CollisionShape3D")["properties"]
    np.testing.assert_allclose(_matrix_storage(foot_collision["friction"]), (0.8, 0.02, 0.01))
    assert foot_collision["condim"] == 6
    np.testing.assert_allclose(foot_collision["solimp"][:3], (0.015, 1.0, 0.023))
    assert foot_collision["margin"] == np.float32(0.005)

    trunk_index = nodes.index(_node_by_name(nodes, "trunk", "Link3D"))
    assert _node_by_name(nodes, "imu", "IMUSensor3D")["parent"] == trunk_index
    assert _node_by_name(nodes, "root_angmom", "AngularMomentumSensor3D")["parent"] == trunk_index
    terrain_scan = _node_by_name(nodes, "terrain_scan", "HeightScanner3D")["properties"]
    assert terrain_scan["pattern_mode"] == "Grid"
    assert terrain_scan["reduction_mode"] == "None"
    for foot in ("FR", "FL", "RR", "RL"):
        assert _node_by_name(nodes, f"{foot}_foot_height_scan", "TerrainHeightSensor3D")["properties"][
            "reduction_mode"
        ] == "Min"
        assert _node_by_name(nodes, f"{foot}_foot_contact", "ContactSensor3D")["properties"]["radius"] == np.float32(
            0.03
        )


def test_go1_benchmark_uses_unilab_env_step_accounting():
    class Args:
        steps = 20
        warmup_steps = 5
        actions = "zero"
        sim_workers = 0

    class Env:
        num_envs = 2048
        decimation = 10
        resolved_sim_workers = 8
        physics_dt = 0.002
        step_dt = 0.02

    records = {
        "env_step_total_ms": [10.0] * 20,
        "apply_action_ms": [1.0] * 20,
        "backend_physics_ms": [6.0] * 20,
        "backend_refresh_cache_ms": [1.0] * 20,
        "update_state_ms": [2.0] * 20,
        "reset_done_ms": [0.5] * 20,
    }
    metrics = go1_velocity_benchmark.build_benchmark_metrics(
        cfg_name="gobot_go1_unilab_flat",
        env=Env(),
        args=Args(),
        elapsed=0.25,
        timing_records=records,
    )
    assert metrics["throughput_env_steps_per_s"] == 2048 * 20 / 0.2
    assert metrics["timing_median_ms"]["env_step_total_ms"] == 10.0


def test_go1_env_applies_unilab_mujoco_solver_settings():
    cfg = go1_cfg.go1_velocity_cfg("go1_rough", project_path=str(REPO_ROOT / "examples/go1"))
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = False
    cfg.observations.actor_noise = False
    cfg.randomize_rough_reset_pose = False

    env = Go1VelocityEnv(num_envs=1, cfg=cfg, sim_workers=0, seed=3)
    try:
        solver_settings = env.context.get_mujoco_solver_settings()
        assert solver_settings["cone"] == 1
        assert solver_settings["convex_collision_iterations"] == 500
        assert solver_settings["impedance_ratio"] == 100.0
        assert env.metadata["mujoco_solver_settings"] == dict(go1_cfg.GO1_UNILAB_MUJOCO_SOLVER_SETTINGS)
        assert env.physics_dt == 0.005
        assert env.decimation == 4
        assert env.step_dt == 0.02
    finally:
        env.close()


def test_go1_env_rough_reset_uses_unilab_random_stream_for_commands():
    cfg = go1_cfg.go1_velocity_cfg("go1_rough", project_path=str(REPO_ROOT / "examples/go1"))
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = False
    cfg.observations.actor_noise = False
    cfg.randomize_rough_reset_pose = True
    cfg.use_unilab_reset_rng = True

    env = Go1VelocityEnv(num_envs=4, cfg=cfg, sim_workers=0, seed=11)
    try:
        command = np.asarray(env.backend.state.command, dtype=np.float32)
        assert command.shape == (4, 3)
        np.testing.assert_allclose(command[:, 2], 0.0)
        assert np.all(np.abs(command[:, :2]) <= 1.0)
        np.testing.assert_allclose(env.backend.state.command_time_left, 10.0)
        assert env.metadata["use_unilab_reset_rng"] is True

        before = command.copy()
        env.reset(seed=11)
        np.testing.assert_allclose(env.backend.state.command, before)
    finally:
        env.close()


def test_go1_playback_profile_observation_shapes():
    script = object.__new__(go1_playback.Script)
    state = {
        "links": [
            {
                "name": go1_playback.BASE_LINK,
                "global_transform": {"position": [0.0, 0.0, 0.5], "quaternion": [1.0, 0.0, 0.0, 0.0]},
                "linear_velocity": [0.1, 0.2, 0.3],
                "angular_velocity": [0.4, 0.5, 0.6],
            }
        ],
        "joints": [
            {"name": name, "position": go1_playback.DEFAULT_POS[index], "velocity": 0.0}
            for index, name in enumerate(go1_playback.JOINT_NAMES)
        ],
        "sensors": [
            {
                "name": go1_playback.TERRAIN_SCAN_SENSOR,
                "values": [0.0] * go1_playback.TERRAIN_SCAN_DIM,
            }
        ],
    }
    script.command = [0.1, 0.0, 0.0]
    script.last_action = [0.0] * len(go1_playback.JOINT_NAMES)
    script.feet_phase = [0.0] * len(go1_playback.LEG_ORDER)
    script.phase = 0.0

    script.policy_profile = "legacy"
    script.policy_obs_dim = go1_playback.ACTOR_OBS_SCHEMA.dim
    script.height_scan_dim = go1_playback.TERRAIN_SCAN_DIM
    assert len(script._observation(state)) == go1_playback.ACTOR_OBS_SCHEMA.dim

    script.policy_profile = "unilab_flat"
    script.policy_obs_dim = go1_playback.UNILAB_FLAT_ACTOR_OBS_SCHEMA.dim
    assert len(script._observation(state)) == go1_playback.UNILAB_FLAT_ACTOR_OBS_SCHEMA.dim

    script.policy_profile = "unilab_rough"
    script.policy_obs_dim = go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim
    assert len(script._observation(state)) == go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim
    script._update_feet_phase()
    assert script.phase > 0.0


def test_go1_playback_unilab_rough_uses_unilab_upvector_observation():
    script = object.__new__(go1_playback.Script)
    script.command = [0.0, 0.0, 0.0]
    script.last_action = [0.0] * len(go1_playback.JOINT_NAMES)
    script.feet_phase = [0.0] * len(go1_playback.LEG_ORDER)
    script.phase = 0.0

    angle = np.pi / 4.0
    quat = [float(np.cos(angle / 2.0)), float(np.sin(angle / 2.0)), 0.0, 0.0]
    state = {
        "links": [
            {
                "name": go1_playback.BASE_LINK,
                "global_transform": {"position": [0.0, 0.0, 0.5], "quaternion": quat},
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }
        ],
        "joints": [
            {"name": name, "position": go1_playback.DEFAULT_POS[index], "velocity": 0.0}
            for index, name in enumerate(go1_playback.JOINT_NAMES)
        ],
        "sensors": [
            {
                "name": go1_playback.TERRAIN_SCAN_SENSOR,
                "values": [0.0] * go1_playback.TERRAIN_SCAN_DIM,
            }
        ],
    }

    script.policy_profile = "unilab_rough"
    script.policy_obs_dim = go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim
    obs = np.asarray(script._observation(state), dtype=np.float32)
    expected = -np.asarray(go1_playback._quat_rotate([0.0, 0.0, 1.0], quat), dtype=np.float32)
    np.testing.assert_allclose(obs[3:6], expected, atol=1.0e-6)


def test_go1_playback_unilab_rough_uses_heading_feedback_command():
    script = object.__new__(go1_playback.Script)
    script.command = [0.3, -0.2, 0.0]
    script.command_target = [0.3, -0.2, 0.0]
    script.heading_target = 1.0
    script.heading_target_initialized = True
    script.last_action = [0.0] * len(go1_playback.JOINT_NAMES)
    script.feet_phase = [0.0] * len(go1_playback.LEG_ORDER)
    script.phase = 0.0

    yaw = 0.25
    quat = [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]
    state = {
        "links": [
            {
                "name": go1_playback.BASE_LINK,
                "global_transform": {"position": [0.0, 0.0, 0.5], "quaternion": quat},
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }
        ],
        "joints": [
            {"name": name, "position": go1_playback.DEFAULT_POS[index], "velocity": 0.0}
            for index, name in enumerate(go1_playback.JOINT_NAMES)
        ],
        "sensors": [
            {
                "name": go1_playback.TERRAIN_SCAN_SENSOR,
                "values": [0.0] * go1_playback.TERRAIN_SCAN_DIM,
            }
        ],
    }

    script.policy_profile = "unilab_rough"
    script.policy_obs_dim = go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim
    obs = np.asarray(script._observation(state), dtype=np.float32)
    np.testing.assert_allclose(obs[6:9], [0.3, -0.2, 0.375], atol=1.0e-6)


def test_go1_playback_zero_command_still_runs_policy():
    class FakePolicy:
        obs_dim = go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim
        schema = go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA

        def __init__(self):
            self.calls = 0

        def action(self, observation):
            self.calls += 1
            assert len(observation) == self.obs_dim
            return [0.5] * len(go1_playback.JOINT_NAMES)

    class FakeContext:
        has_world = True
        simulation_time = 0.0
        input = None

    class FakeRobot:
        name = go1_playback.ROBOT

    class FakeJoint:
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

    class FakeSensor:
        def get_runtime_state(self):
            return {"name": go1_playback.TERRAIN_SCAN_SENSOR, "values": []}

    script = object.__new__(go1_playback.Script)
    script.context = FakeContext()
    script.playing = True
    script.world_controls_ready = True
    script.ticks = 0
    script.command = [0.0, 0.0, 0.0]
    script.policy = FakePolicy()
    script.policy_profile = "unilab_rough"
    script.policy_obs_dim = go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim
    script.action_scale = list(go1_playback.UNILAB_ROUGH_ACTION_SCALE)
    script.action_clip = 100.0
    script.last_action = [0.0] * len(go1_playback.JOINT_NAMES)
    script.last_targets = list(go1_playback.DEFAULT_POS)
    script.phase = 0.0
    script.feet_phase = [0.0] * len(go1_playback.LEG_ORDER)
    script.base_link = FakeLink()
    script.terrain_scan = FakeSensor()
    script.robot = FakeRobot()
    script.joints = [
        FakeJoint(name, go1_playback.DEFAULT_POS[index])
        for index, name in enumerate(go1_playback.JOINT_NAMES)
    ]

    script._physics_process(1.0 / go1_playback.PHYSICS_HZ)

    assert script.policy.calls == 1
    assert script.last_action == [0.5] * len(go1_playback.JOINT_NAMES)
    assert all(joint.target is not None for joint in script.joints)


def test_go1_compare_dump_path_splits_both_backends():
    class Args:
        backend = "both"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        Args.dump_npz = tmp_path / "parity.npz"
        assert go1_compare._dump_path(Args, "gobot") == tmp_path / "parity_gobot.npz"
        assert go1_compare._dump_path(Args, "unilab") == tmp_path / "parity_unilab.npz"

    Args.backend = "gobot"
    Args.dump_npz = Path("/tmp/parity.npz")
    assert go1_compare._dump_path(Args, "gobot") == Path("/tmp/parity.npz")


def test_go1_compare_fixed_reset_disables_randomized_terms():
    args = go1_compare.build_parser().parse_args(["--fixed-reset"])
    assert not args.no_domain_rand
    assert not args.no_push
    assert not args.no_obs_noise

    go1_compare.normalize_args(args)

    assert args.no_domain_rand
    assert args.no_push
    assert args.no_obs_noise


def test_go1_rough_playback_reset_uses_nearest_terrain_spawn():
    class FakeTerrain:
        spawn_origins = [
            [-20.0, -20.0, 0.0],
            [-4.0, -4.0, 0.525],
            [12.0, 4.0, -0.2],
        ]

    class FakeTerrainWorld:
        def find(self, name):
            return FakeTerrain() if name == "terrain" else None

    class FakeRoot:
        def find(self, name):
            return FakeTerrainWorld() if name == "terrain_world" else None

    script = object.__new__(go1_playback.Script)
    script.policy_profile = "unilab_rough"
    script.get_root = lambda: FakeRoot()

    np.testing.assert_allclose(
        script._resolve_reset_base_position(),
        [-4.0, -4.0, 0.845],
    )


def test_go1_train_cfg_preserves_unilab_task_contract():
    go1_velocity_train = _require_go1_velocity_train()

    args = go1_velocity_train.parse_args(["--task", "go1_rough", "--iterations", "2", "--num-envs", "3"])
    cfg = go1_velocity_train.build_velocity_cfg(args, REPO_ROOT / "examples/go1")
    assert cfg.task_profile == "unilab_rough"
    assert not cfg.terrain_curriculum

    train_cfg = go1_velocity_train.build_train_cfg(args, cfg)
    assert train_cfg["experiment_name"] == "gobot_go1_unilab_rough"
    assert train_cfg["seed"] == 42
    assert train_cfg["clip_actions"] == 100.0
    assert train_cfg["algorithm"]["class_name"] == "gobot.rl.rsl_rl.FinalObservationAwarePPO"
    assert train_cfg["algorithm"]["entropy_coef"] == 0.01

    args = go1_velocity_train.parse_args(["--task", "go1_rough", "--terrain-curriculum"])
    cfg = go1_velocity_train.build_velocity_cfg(args, REPO_ROOT / "examples/go1")
    assert cfg.terrain_curriculum

    flat_args = go1_velocity_train.parse_args(["--task", "go1_flat"])
    flat_cfg = go1_velocity_train.build_velocity_cfg(flat_args, REPO_ROOT / "examples/go1")
    flat_train_cfg = go1_velocity_train.build_train_cfg(flat_args, flat_cfg)
    assert flat_cfg.task_profile == "unilab_flat"
    assert flat_train_cfg["clip_actions"] == 1.0


def test_go1_train_checkpoint_path_prefers_cwd_relative_path(tmp_path):
    go1_velocity_train = _require_go1_velocity_train()

    checkpoint = tmp_path / "model_7.pt"
    checkpoint.write_bytes(b"checkpoint")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    assert go1_velocity_train._resolve_checkpoint(str(checkpoint), log_dir) == checkpoint


def test_go1_unilab_rough_long_eval_episode_reset_does_not_overflow_uint32():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1", play=True)
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=10_000_000_000)
    try:
        obs = env.get_observations()
        assert obs["actor"].shape == (1, 45)
        assert np.isfinite(obs["actor"]).all()
    finally:
        env.close()


def test_go1_unilab_rough_env_reset_step_shapes():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=4)
    try:
        expected_critic_dim = 48 + env.cfg["height_scan_dim"]
        obs = env.get_observations()
        assert obs["actor"].shape == (1, 45)
        assert obs["critic"].shape == (1, expected_critic_dim)
        assert env.obs_groups_spec == {"actor": 45, "critic": expected_critic_dim}
        assert env.cfg["task_profile"] == "unilab_rough"
        assert env.cfg["action_clip"] == 100.0
        assert env.cfg["task_runtime"]["mode"] == "numpy"
        assert not env.cfg["task_runtime"]["compiled"]
        assert env.cfg["task_runtime"]["backend"] == "gobot_native_cpu_batch_numpy"
        assert env.cfg["task_runtime"]["reward_names"] == (
            "lin_vel_z",
            "ang_vel_xy",
            "joint_torques_l2",
            "joint_acc_l2",
            "joint_power",
            "stand_still",
            "hip_pos",
            "joint_pos_penalty",
            "joint_mirror",
            "action_rate",
            "undesired_contacts",
            "contact_forces",
            "tracking_lin_vel",
            "tracking_ang_vel",
            "feet_air_time",
            "feet_air_time_variance",
            "feet_contact_without_cmd",
            "feet_slide",
            "feet_height_body",
            "feet_gait",
            "upward",
        )
        assert env.cfg["task_runtime"]["obs_groups_spec"] == {"actor": 45, "critic": expected_critic_dim}
        assert "action_clip" in env.cfg["task_runtime"]["array_names"]
        assert "push_force" in env.cfg["task_runtime"]["array_names"]
        assert "base_mass_delta" in env.cfg["task_runtime"]["array_names"]
        assert "joint_kp" in env.cfg["task_runtime"]["array_names"]
        assert "undesired_contact_count" in env.cfg["task_runtime"]["array_names"]
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["scene_source"] == "jscn"
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["unilab_reference"]
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["native_contact_detail"] == "unilab_geom_sensor"
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["domain_randomization_backend"] == "per_env_mjmodel_pool"
        go1_env_source = Path(__file__).resolve().parents[2].joinpath(
            "examples/go1/train/go1_velocity_env.py"
        ).read_text(encoding="utf-8")
        assert "TaskExpression" not in go1_env_source
        assert "RewardTermSpec" not in go1_env_source
        assert "task_buffer(" not in go1_env_source
        assert "TaskJitCompiler" not in go1_env_source
        assert "install_kernel(" not in go1_env_source
        assert "step_task_kernel(" not in go1_env_source
        assert "set_push_forces(" in go1_env_source
        assert "_maybe_apply_push" not in go1_env_source
        assert not REPO_ROOT.joinpath("examples/go1/train/go1_task_kernels.py").exists()

        state = env.step(np.zeros((1, env.num_actions), dtype=np.float32))
        assert isinstance(state, BatchEnvState)
        assert state.obs["actor"].shape == (1, 45)
        assert state.obs["critic"].shape == (1, expected_critic_dim)
        assert state.reward.shape == (1,)
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        batch_state = env.backend.state
        assert "local_linvel" in env.cfg["task_runtime"]["array_names"]
        assert "upvector" in env.cfg["task_runtime"]["array_names"]
        np.testing.assert_allclose(batch_state.local_linvel, batch_state.base_linear_velocity_body)
        np.testing.assert_allclose(batch_state.gyro, batch_state.base_angular_velocity_body)
        assert np.isfinite(batch_state.upvector).all()
        for key in ("env_step_total_ms", "backend_physics_ms", "numpy_task_ms", "update_state_ms", "reset_done_ms"):
            assert key in state.info["timing"]
    finally:
        env.close()


def test_go1_unilab_rough_native_command_sampling_matches_unilab_reset():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    env = Go1VelocityEnv(cfg, num_envs=128, device="cpu", seed=7, max_episode_length=8)
    try:
        env.reset(seed=7)
        command = np.asarray(env.command_b, dtype=np.float32)
        standing = np.asarray(env.backend.state.command_is_standing_env, dtype=bool)
        speed_xy = np.linalg.norm(command[:, :2], axis=1)

        np.testing.assert_allclose(command[:, 2], 0.0)
        if np.any(standing):
            np.testing.assert_allclose(command[standing], 0.0)
        moving_or_zeroed = (~standing) & (speed_xy > 1.0e-7)
        assert np.all(speed_xy[moving_or_zeroed] > cfg.command.zero_small_xy_threshold)

        env.step(np.zeros((env.num_envs, env.num_actions), dtype=np.float32))
        stepped_command = np.asarray(env.command_b, dtype=np.float32)
        if np.any(standing):
            np.testing.assert_allclose(stepped_command[standing, :2], 0.0, atol=1.0e-6)
            assert np.any(np.abs(stepped_command[standing, 2]) > 1.0e-5)
    finally:
        env.close()


def test_go1_native_runtime_state_tracks_batch_arrays_after_step():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=8)
    try:
        env.reset(seed=123)
        env.step(np.zeros((1, env.num_actions), dtype=np.float32))
        runtime_state = env._runtime_state(0)
        batch_state = env.backend.state

        np.testing.assert_allclose(
            runtime_state.base["global_transform"]["position"],
            batch_state.base_position[0],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            runtime_state.base["global_transform"]["quaternion"],
            batch_state.base_quaternion[0],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            [runtime_state.joints[name]["position"] for name in env.joint_names],
            batch_state.joint_position[0],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert "link_position" in env.cfg["task_runtime"]["array_names"]
        assert "link_quaternion" in env.cfg["task_runtime"]["array_names"]
        assert env.cfg_obj.base_link in runtime_state.links
        assert len(runtime_state.links) == len(env._batch_link_names)
    finally:
        env.close()


def test_go1_native_base_mass_randomization_preserves_reset_state_distribution():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.observations.critic_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = True
    cfg.domain_randomization.randomize_base_mass = True
    cfg.domain_randomization.random_com = False
    cfg.domain_randomization.randomize_kp = False
    cfg.domain_randomization.randomize_kd = False
    cfg.domain_randomization.encoder_bias_range = (0.0, 0.0)
    env = Go1VelocityEnv(cfg, num_envs=16, device="cpu", seed=42, max_episode_length=64, collect_step_extras=True)
    try:
        env.reset(seed=42)
        actions = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
        for _ in range(24):
            state = env.step(actions)
        batch_state = env.backend.state
        upright = -np.asarray(batch_state.projected_gravity, dtype=np.float32)[:, 2]
        assert float(np.mean(upright > 0.9)) < 0.75
        assert float(np.std(batch_state.base_position[:, 2])) > 0.1
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_go1_unilab_flat_env_reset_step_shapes():
    cfg = go1_cfg.go1_flat_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.domain_randomization.enabled = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=4)
    try:
        obs = env.get_observations()
        assert obs["actor"].shape == (1, 49)
        assert obs["critic"].shape == (1, 52)
        assert env.obs_groups_spec == {"actor": 49, "critic": 52}
        assert env.cfg["task_profile"] == "unilab_flat"
        assert env.cfg["task_runtime"]["reward_names"] == (
            "tracking_lin_vel",
            "tracking_ang_vel",
            "lin_vel_z",
            "ang_vel_xy",
            "base_height",
            "action_rate",
            "similar_to_default",
            "contact",
            "swing_feet_z",
        )

        state = env.step(np.zeros((1, env.num_actions), dtype=np.float32))
        assert state.obs["actor"].shape == (1, 49)
        assert state.obs["critic"].shape == (1, 52)
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.obs["critic"]).all()
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
    env.rsl_rl_include_final_observation = True
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    _, _, _, extras = wrapper.step(torch.zeros((2, 3), dtype=torch.float32))
    assert extras["reward_terms"]["term"].tolist() == [1.0, 1.0]
    assert extras["final_observation"]["actor"].shape == (2, 4)
    assert extras["time_out_bootstrap_obs"]["actor"].shape == (2, 4)
    assert extras["_final_observation"].tolist() == [False, True]


def main():
    tests = [
        test_batch_env_state_contract,
        test_batch_env_clears_stale_final_observation_info,
        test_task_runtime_metadata_is_public_task_summary,
        test_locomotion_common_helpers,
        test_go1_unilab_cfg_profiles_and_playback_schemas,
        test_go1_benchmark_uses_unilab_env_step_accounting,
        test_go1_playback_profile_observation_shapes,
        test_go1_playback_unilab_rough_uses_unilab_upvector_observation,
        test_go1_playback_unilab_rough_uses_heading_feedback_command,
        test_go1_playback_zero_command_still_runs_policy,
        test_go1_compare_dump_path_splits_both_backends,
        test_go1_compare_fixed_reset_disables_randomized_terms,
        test_go1_rough_playback_reset_uses_nearest_terrain_spawn,
        test_go1_train_cfg_preserves_unilab_task_contract,
        test_go1_unilab_rough_env_reset_step_shapes,
        test_go1_unilab_rough_native_command_sampling_matches_unilab_reset,
        test_go1_native_runtime_state_tracks_batch_arrays_after_step,
        test_go1_native_base_mass_randomization_preserves_reset_state_distribution,
        test_go1_unilab_flat_env_reset_step_shapes,
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
