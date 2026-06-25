from pathlib import Path
import sys

import numpy as np

from _gobot_test_import import prefer_build_gobot

prefer_build_gobot()

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "python", REPO_ROOT / "build/python"):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
    sys.path.insert(0, path_string)

from benchmark import go1_velocity_benchmark
from examples.go1.scripts import go1 as go1_playback
from examples.go1.train import go1_velocity_cfg as go1_cfg
from examples.go1.train import go1_velocity_train
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
from gobot.rl import (
    BatchEnvState,
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
    assert cfg.kp == 35.0
    assert cfg.kd == 0.5
    assert cfg.command.heading_command
    assert cfg.command.resampling_time_range == (10.0, 10.0)
    assert cfg.command.ranges.lin_vel_x == (-1.0, 1.0)
    assert cfg.command.ranges.lin_vel_y == (-1.0, 1.0)
    assert cfg.command.ranges.ang_vel_z == (-1.0, 1.0)
    assert cfg.domain_randomization.enabled
    assert cfg.domain_randomization.randomize_base_mass
    assert cfg.domain_randomization.random_com
    assert cfg.domain_randomization.randomize_kp
    assert cfg.domain_randomization.randomize_kd
    assert cfg.push_enabled
    assert cfg.push_interval_steps == 750
    assert cfg.terrain_out_of_bounds
    assert cfg.unilab_rewards.scales["feet_gait"] == 0.5
    assert cfg.unilab_rewards.scales["tracking_lin_vel"] == 3.0
    assert np.allclose(np.asarray(cfg.default_joint_pos, dtype=np.float32)[[0, 3, 6, 9]], [0.1, -0.1, 0.1, -0.1])

    actor_schema = velocity_actor_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM)
    critic_schema = velocity_critic_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM, len(cfg.foot_names))
    assert actor_schema.dim == 235
    assert critic_schema.dim == 259
    assert go1_playback.ACTOR_OBS_SCHEMA.names == actor_schema.names
    assert go1_playback.UNILAB_ROUGH_ACTOR_OBS_SCHEMA.dim == 45
    assert go1_playback.UNILAB_FLAT_ACTOR_OBS_SCHEMA.dim == 49
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
    assert flat.observations.height_scan_sensor is None
    assert not flat.terrain_curriculum
    assert flat.action_clip == 1.0
    assert flat.physics_dt == 0.01
    assert flat.decimation == 2
    assert flat.command.resampling_time_range == (1_000_000_000.0, 1_000_000_000.0)
    assert flat.command.ranges.lin_vel_x == (-0.6, 1.0)
    assert flat.command.ranges.lin_vel_y == (-0.4, 0.4)
    assert flat.command.ranges.ang_vel_z == (-0.8, 0.8)
    assert flat.domain_randomization.enabled
    assert flat.domain_randomization.randomize_base_mass
    assert flat.domain_randomization.random_com
    assert not flat.domain_randomization.randomize_kp
    assert not flat.domain_randomization.randomize_kd
    assert flat.push_enabled
    assert flat.unilab_rewards.scales["base_height"] == -100.0


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


def test_go1_train_cfg_preserves_unilab_task_contract():
    args = go1_velocity_train.parse_args(["--task", "go1_rough", "--iterations", "2", "--num-envs", "3"])
    cfg = go1_velocity_train.build_velocity_cfg(args, REPO_ROOT / "examples/go1")
    assert cfg.task_profile == "unilab_rough"
    assert not cfg.terrain_curriculum

    train_cfg = go1_velocity_train.build_train_cfg(args, cfg)
    assert train_cfg["experiment_name"] == "gobot_go1_unilab_rough"
    assert train_cfg["clip_actions"] == 100.0

    args = go1_velocity_train.parse_args(["--task", "go1_rough", "--terrain-curriculum"])
    cfg = go1_velocity_train.build_velocity_cfg(args, REPO_ROOT / "examples/go1")
    assert cfg.terrain_curriculum

    flat_args = go1_velocity_train.parse_args(["--task", "go1_flat"])
    flat_cfg = go1_velocity_train.build_velocity_cfg(flat_args, REPO_ROOT / "examples/go1")
    flat_train_cfg = go1_velocity_train.build_train_cfg(flat_args, flat_cfg)
    assert flat_cfg.task_profile == "unilab_flat"
    assert flat_train_cfg["clip_actions"] == 1.0


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
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["scene_source"] == "jscn"
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["unilab_reference"]
        assert env.cfg["task_runtime"]["metadata"]["cache_info"]["native_contact_detail"] == "categorized"
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
        for key in ("env_step_total_ms", "backend_physics_ms", "numpy_task_ms", "update_state_ms", "reset_done_ms"):
            assert key in state.info["timing"]
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
    assert extras["log"]["x"].item() == 1.0
    assert isinstance(env.state.obs["actor"], np.ndarray)


def main():
    tests = [
        test_batch_env_state_contract,
        test_task_runtime_metadata_is_public_task_summary,
        test_locomotion_common_helpers,
        test_go1_unilab_cfg_profiles_and_playback_schemas,
        test_go1_benchmark_uses_unilab_env_step_accounting,
        test_go1_playback_profile_observation_shapes,
        test_go1_train_cfg_preserves_unilab_task_contract,
        test_go1_unilab_rough_env_reset_step_shapes,
        test_go1_unilab_flat_env_reset_step_shapes,
        test_rsl_rl_wrapper_keeps_core_env_numpy,
    ]
    try:
        for test in tests:
            test()
    except OptionalDependencyUnavailable as error:
        print(error)
        sys.exit(OPTIONAL_DEPENDENCY_SKIP_CODE)


if __name__ == "__main__":
    main()
