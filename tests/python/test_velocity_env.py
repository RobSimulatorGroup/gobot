from pathlib import Path
import sys
import types

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
from examples.go1.train.go1_task_kernels import go1_velocity_task
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
from gobot.rl import (
    ActionSpec,
    BatchEnvState,
    RewardTermSpec,
    SpecField,
    TaskExpression,
    TaskJitCompiler,
    TaskLayout,
    task_buffer,
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


def test_task_ir_metadata_and_array_validation():
    actor_spec = velocity_actor_observation_schema(1, 0)
    task = TaskLayout(
        name="dummy",
        version="dummy_v1",
        action_spec=ActionSpec(version="action_v1", fields=(SpecField("joint", 1),)),
        obs_groups={"actor": actor_spec},
        buffers=(task_buffer("obs", "env", actor_spec.dim), task_buffer("flag", "env", dtype="uint8")),
        reward_terms=(RewardTermSpec("alive", 1.0, TaskExpression("constant", params={"value": 1.0})),),
    )
    arrays = types.SimpleNamespace(
        obs=np.zeros((2, actor_spec.dim), dtype=np.float32),
        flag=np.zeros((2,), dtype=np.uint8),
    )
    assert task.metadata()["kind"] == "gobot_task_ir"
    assert task.obs_groups_spec == {"actor": actor_spec.dim}
    assert task.reward_names == ("alive",)
    task.validate_native_arrays(arrays)

    arrays.obs = np.zeros((2, actor_spec.dim + 1), dtype=np.float32)
    _assert_runtime_error("shape mismatch", lambda: task.validate_native_arrays(arrays))


def test_go1_task_jit_compiles_and_runs():
    env = _dummy_go1_env()
    task = env._make_task_ir()
    arrays = _dummy_go1_arrays(env)

    task.validate_native_arrays(arrays)
    compiled = TaskJitCompiler(cache_dir="/tmp/gobot-task-jit-test").compile(task, arrays, kernel=go1_velocity_task)
    compiled.run(arrays)

    assert compiled.function_address != 0
    assert compiled.build_info.backend == "llvm"
    assert compiled.build_info.library_path.exists()
    assert arrays.actor_obs.shape == (2, env.num_obs)
    assert arrays.critic_obs.shape == (2, env.num_privileged_obs)
    assert arrays.reward_terms.shape == (2, 7)
    assert np.isfinite(arrays.actor_obs).all()
    assert np.isfinite(arrays.critic_obs).all()
    assert np.isfinite(arrays.reward).all()


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


def test_go1_cfg_and_playback_schema_match():
    cfg = go1_cfg.go1_velocity_cfg("go1_rough", project_path="/tmp/go1")
    assert cfg.name == "gobot_go1_velocity"
    assert len(cfg.joint_names) == 12
    assert cfg.observations.height_scan_sensor == "terrain_scan"
    assert np.allclose(np.asarray(cfg.default_joint_pos, dtype=np.float32)[[0, 3, 6, 9]], [0.1, -0.1, 0.1, -0.1])

    actor_schema = velocity_actor_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM)
    critic_schema = velocity_critic_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM, len(cfg.foot_names))
    assert actor_schema.dim == 235
    assert critic_schema.dim == 259
    assert go1_playback.ACTOR_OBS_SCHEMA.names == actor_schema.names
    assert np.allclose(go1_playback.DEFAULT_POS, cfg.default_joint_pos)

    flat = go1_cfg.go1_velocity_cfg("go1_flat", project_path="/tmp/go1")
    assert flat.name == "gobot_go1_flat_velocity"
    assert flat.observations.height_scan_sensor is None
    assert not flat.terrain_curriculum


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
        cfg_name="gobot_go1_flat_velocity",
        env=Env(),
        args=Args(),
        elapsed=0.25,
        timing_records=records,
    )
    assert metrics["throughput_env_steps_per_s"] == 2048 * 20 / 0.2
    assert metrics["timing_median_ms"]["env_step_total_ms"] == 10.0


def test_go1_velocity_env_reset_step_shapes():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=4)
    try:
        obs = env.get_observations()
        assert obs["actor"].shape == (1, 235)
        assert obs["critic"].shape == (1, 259)
        assert env.obs_groups_spec == {"actor": 235, "critic": 259}
        assert env.cfg["task_kernel"]["mode"] == "jit"
        assert env.cfg["task_kernel"]["compiled"]
        assert env.task_ir.reward_names == (
            "track_linear_velocity",
            "track_angular_velocity",
            "upright",
            "action_rate_l2",
            "air_time",
            "foot_clearance",
            "foot_slip",
        )
        env.task_ir.validate_native_arrays(env.backend.state)

        state = env.step(np.zeros((1, env.num_actions), dtype=np.float32))
        assert isinstance(state, BatchEnvState)
        assert state.obs["actor"].shape == (1, 235)
        assert state.obs["critic"].shape == (1, 259)
        assert state.reward.shape == (1,)
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        for key in ("env_step_total_ms", "backend_physics_ms", "update_state_ms", "reset_done_ms"):
            assert key in state.info["timing"]
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


def _dummy_go1_env():
    cfg = go1_cfg.go1_flat_velocity_cfg(project_path="/tmp/go1")
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = cfg
    env.num_envs = 2
    env.num_actions = len(cfg.joint_names)
    env.joint_names = tuple(cfg.joint_names)
    env.default_joint_pos = np.asarray(cfg.default_joint_pos, dtype=np.float32)
    env.action_scale = np.ones(env.num_actions, dtype=np.float32)
    env.actor_obs_schema = velocity_actor_observation_schema(env.num_actions, 0)
    env.critic_obs_schema = velocity_critic_observation_schema(env.num_actions, 0, len(cfg.foot_names))
    env.action_spec = env._make_action_spec()
    env.num_obs = env.actor_obs_schema.dim
    env.num_privileged_obs = env.critic_obs_schema.dim
    env._height_scan_dim = 0
    env._foot_count = len(cfg.foot_names)
    return env


def _dummy_go1_arrays(env):
    num_envs = int(env.num_envs)
    num_dof = int(env.num_actions)
    num_feet = int(env._foot_count)
    height_scan_dim = int(env._height_scan_dim)
    arrays = types.SimpleNamespace()
    arrays.base_linear_velocity_body = np.zeros((num_envs, 3), dtype=np.float32)
    arrays.base_angular_velocity_body = np.zeros((num_envs, 3), dtype=np.float32)
    arrays.projected_gravity = np.tile(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (num_envs, 1))
    arrays.joint_position = np.tile(np.asarray(env.default_joint_pos, dtype=np.float32), (num_envs, 1))
    arrays.joint_velocity = np.zeros((num_envs, num_dof), dtype=np.float32)
    arrays.default_joint_position = np.asarray(env.default_joint_pos, dtype=np.float32).copy()
    arrays.encoder_bias = np.zeros((num_envs, num_dof), dtype=np.float32)
    arrays.previous_action = np.zeros((num_envs, num_dof), dtype=np.float32)
    arrays.last_action = np.zeros((num_envs, num_dof), dtype=np.float32)
    arrays.submitted_action = np.zeros((num_envs, num_dof), dtype=np.float32)
    arrays.command = np.zeros((num_envs, 3), dtype=np.float32)
    arrays.foot_height = np.zeros((num_envs, num_feet), dtype=np.float32)
    arrays.foot_air_time = np.zeros((num_envs, num_feet), dtype=np.float32)
    arrays.foot_contact = np.ones((num_envs, num_feet), dtype=np.float32)
    arrays.foot_contact_force = np.zeros((num_envs, num_feet, 3), dtype=np.float32)
    arrays.base_height = np.full((num_envs,), 0.35, dtype=np.float32)
    arrays.base_clearance = np.zeros((num_envs,), dtype=np.float32)
    arrays.velocity_error = np.zeros((num_envs,), dtype=np.float32)
    arrays.foot_slip = np.zeros((num_envs,), dtype=np.float32)
    arrays.reward_weights = np.asarray(env._reward_weights(), dtype=np.float32)
    arrays.task_params = np.asarray([0.02, 0.25, 0.25, 0.25, 0.05, 0.16, np.deg2rad(70.0), 5.0], dtype=np.float32)
    arrays.task_flags = np.zeros((1,), dtype=np.float32)
    arrays.reward_terms = np.zeros((num_envs, 7), dtype=np.float32)
    arrays.reward = np.zeros((num_envs,), dtype=np.float32)
    arrays.terminated = np.zeros((num_envs,), dtype=np.uint8)
    arrays.actor_obs = np.zeros((num_envs, env.num_obs), dtype=np.float32)
    arrays.critic_obs = np.zeros((num_envs, env.num_privileged_obs), dtype=np.float32)
    arrays.height_scan = np.zeros((num_envs, height_scan_dim), dtype=np.float32)
    return arrays


def main():
    tests = [
        test_batch_env_state_contract,
        test_task_ir_metadata_and_array_validation,
        test_go1_task_jit_compiles_and_runs,
        test_locomotion_common_helpers,
        test_go1_cfg_and_playback_schema_match,
        test_go1_benchmark_uses_unilab_env_step_accounting,
        test_go1_velocity_env_reset_step_shapes,
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
