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

from examples.go1.scripts import go1 as go1_playback
from benchmark import go1_velocity_benchmark
from examples.go1.train import go1_velocity_cfg as go1_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv, VelocityRuntimeState
from examples.go1.train import go1_velocity_video
from examples.go1.train.go1_velocity_video import Go1TrainingVideoCfg, Go1TrainingVideoRecorder
from gobot.rl import ActionSpec, BatchEnvState, RewardTermSpec, SpecField, TaskExpression, TaskLayout, task_buffer
from gobot.rl.rsl_rl import RslRlVecEnvWrapper
from gobot.rl.locomotion import (
    HeightScan,
    LocomotionDomainRandomization,
    LocomotionDomainRandomizationCfg,
    LocomotionRewardContext,
    TerrainSpawn,
    UniformVelocityCommand,
    action_rate_l2,
    dispatch_reward_terms,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)


OPTIONAL_DEPENDENCY_SKIP_CODE = 77


class OptionalDependencyUnavailable(RuntimeError):
    pass


def _require_torch():
    try:
        return __import__("torch")
    except ImportError as error:
        raise OptionalDependencyUnavailable("torch is unavailable; skipping Go1 velocity env integration test") from error


def _assert_raises_runtime_error(pattern: str, fn) -> None:
    try:
        fn()
    except RuntimeError as error:
        assert pattern in str(error)
        return
    raise AssertionError("expected RuntimeError")


def test_batch_env_state_contract_and_terminal_observation():
    obs = {"actor": np.zeros((3, 2), dtype=np.float32), "critic": np.ones((3, 3), dtype=np.float32)}
    state = BatchEnvState(
        obs=obs,
        reward=np.zeros(3, dtype=np.float32),
        terminated=np.asarray([False, True, False]),
        truncated=np.asarray([False, False, True]),
        info={"steps": np.asarray([2, 3, 4], dtype=np.int64)},
    )
    assert state.done.tolist() == [False, True, True]
    assert isinstance(state.obs, dict)


def test_task_ir_metadata_and_native_array_validation():
    actor_spec = velocity_actor_observation_schema(1, 0)
    task = TaskLayout(
        name="dummy",
        version="dummy_v1",
        action_spec=ActionSpec(
            version="action_v1",
            fields=(SpecField("joint", 1),),
        ),
        obs_groups={"actor": actor_spec},
        buffers=(task_buffer("obs", "env", actor_spec.dim), task_buffer("flag", "env", dtype="uint8")),
        reward_terms=(
            RewardTermSpec("alive", 1.0, TaskExpression("constant", params={"value": 1.0})),
        ),
    )
    assert task.metadata()["kind"] == "gobot_task_ir"
    assert task.obs_groups_spec == {"actor": actor_spec.dim}
    assert task.reward_names == ("alive",)

    arrays = types.SimpleNamespace(
        obs=np.zeros((2, actor_spec.dim), dtype=np.float32),
        flag=np.zeros((2,), dtype=np.uint8),
    )
    task.validate_native_arrays(arrays)
    arrays.obs = np.zeros((2, actor_spec.dim + 1), dtype=np.float32)
    _assert_raises_runtime_error("shape mismatch", lambda: task.validate_native_arrays(arrays))


def test_locomotion_unit_helpers():
    scan = HeightScan(dim=4, max_distance=2.0)
    normalized = scan.normalize(np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32))
    assert np.allclose(normalized, [0.0, 0.5, 1.0, 2.0])
    _assert_raises_runtime_error("expected trailing dimension 4", lambda: scan.normalize(np.zeros(3, dtype=np.float32)))
    normal = scan.terrain_normal(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.1],
            ],
            dtype=np.float32,
        )
    )
    expected = np.array([-0.1, 0.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(normal, expected, atol=1.0e-5)

    spawn = TerrainSpawn(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        levels=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )
    rng_a = np.random.default_rng(5)
    rng_b = np.random.default_rng(5)
    assert [spawn.sample_index(rng_a, progress=0.6, limit=0.0) for _ in range(5)] == [
        spawn.sample_index(rng_b, progress=0.6, limit=0.0) for _ in range(5)
    ]
    assert spawn.update_limit(0.5, reset_reason=2, survival=0.2, distance=0.0, expected_distance=1.0) > 0.5
    assert spawn.update_limit(0.5, reset_reason=1, survival=0.1, distance=0.0, expected_distance=1.0) < 0.5

    dr = LocomotionDomainRandomization(
        LocomotionDomainRandomizationCfg(
            encoder_bias_range=(-0.1, 0.1),
            reset_lin_vel_ranges={"x": (-1.0, 1.0)},
        ),
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


def test_go1_benchmark_metrics_match_unilab_env_step_accounting():
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
    assert metrics["num_envs"] == 2048
    assert metrics["num_steps"] == 20
    assert metrics["timing_median_ms"]["env_step_total_ms"] == 10.0
    assert metrics["throughput_env_steps_per_s"] == 2048 * 20 / 0.2


def test_go1_velocity_cfg_dimensions():
    cfg = go1_cfg.go1_velocity_cfg("go1_rough", project_path="/tmp/go1")
    assert cfg.name == "gobot_go1_velocity"
    assert cfg.robot_name == "go1"
    assert cfg.observations.height_scan_sensor == "terrain_scan"
    assert cfg.terrain_normal_upright.enabled
    assert cfg.illegal_contact.enabled
    assert cfg.domain_randomization.enabled
    assert cfg.push_enabled
    assert np.allclose(np.asarray(cfg.default_joint_pos, dtype=np.float32)[[0, 3, 6, 9]], [0.1, -0.1, 0.1, -0.1])
    assert cfg.spawn_jitter == 0.5
    assert cfg.push_interval_range_s == (1.0, 3.0)
    assert cfg.push_velocity_ranges["z"] == (-0.4, 0.4)
    assert cfg.push_velocity_ranges["roll"] == (-0.52, 0.52)
    assert cfg.domain_randomization.encoder_bias_range == (-0.015, 0.015)
    assert np.isclose(cfg.action_scale[r".*_(hip|thigh)_joint"], 0.3727530386870487)
    assert np.isclose(cfg.action_scale[r".*_calf_joint"], 0.24850202579136574)

    actor_schema = velocity_actor_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM)
    critic_schema = velocity_critic_observation_schema(
        len(cfg.joint_names),
        go1_playback.TERRAIN_SCAN_DIM,
        len(cfg.foot_names),
    )
    assert actor_schema.dim == 235
    assert critic_schema.dim == 259
    assert len(cfg.joint_names) == 12

    flat = go1_cfg.go1_velocity_cfg("go1_flat", project_path="/tmp/go1")
    assert flat.name == "gobot_go1_flat_velocity"
    assert flat.observations.height_scan_sensor is None
    assert not flat.terrain_curriculum
    assert not flat.terrain_normal_upright.enabled
    assert not flat.illegal_contact.enabled


def test_go1_playback_schema_matches_training_schema():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    training_schema = velocity_actor_observation_schema(len(cfg.joint_names), go1_playback.TERRAIN_SCAN_DIM)
    assert go1_playback.ACTOR_OBS_SCHEMA.names == training_schema.names
    assert go1_playback.ACTOR_OBS_SCHEMA.dim == 235
    assert np.allclose(go1_playback.DEFAULT_POS, cfg.default_joint_pos)
    assert np.allclose(
        go1_playback.ACTION_SCALE,
        [
            cfg.action_scale[r".*_(hip|thigh)_joint"],
            cfg.action_scale[r".*_(hip|thigh)_joint"],
            cfg.action_scale[r".*_calf_joint"],
        ]
        * 4,
    )


def test_go1_playback_height_scan_requires_runtime_sensor():
    script = object.__new__(go1_playback.Script)

    _assert_raises_runtime_error("missing required sensor", lambda: script._height_scan({"sensors": []}))


def test_go1_velocity_env_reset_step_shapes():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=4)
    try:
        observations = env.get_observations()
        assert observations["actor"].shape == (1, env.num_obs)
        assert observations["critic"].shape == (1, env.num_privileged_obs)
        assert isinstance(observations["actor"], np.ndarray)
        assert env.num_obs == 235
        assert env.num_privileged_obs == 259
        assert env.obs_groups_spec == {"actor": 235, "critic": 259}
        assert env.cfg["task"] == "gobot_go1_velocity"
        assert env.cfg["obs_schema_version"] == env.actor_obs_schema.version
        assert env.cfg["obs_names"] == env.actor_obs_schema.names
        assert env.task_ir.obs_groups_spec == env.obs_groups_spec
        assert env.task_ir.reward_names == tuple(term["name"] for term in env.cfg["task_ir"]["reward_terms"])
        assert env.cfg["task_ir"]["kind"] == "gobot_task_ir"
        assert env.cfg["task_ir"]["backend"] == "gobot_native_cpu_fused"
        assert env.cfg["task_ir"]["obs_groups_spec"] == env.obs_groups_spec
        assert env.cfg["task_ir"]["reward_terms"][0]["name"] == "track_linear_velocity"
        env.task_ir.validate_native_arrays(env.backend.state)

        state = env.step(np.zeros((1, env.num_actions), dtype=np.float32))
        assert isinstance(state, BatchEnvState)
        assert state.obs["actor"].shape == (1, env.num_obs)
        assert state.obs["critic"].shape == (1, env.num_privileged_obs)
        assert state.reward.shape == (1,)
        assert state.done.shape == (1,)
        assert np.isfinite(state.obs["actor"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
        assert "/velocity/terrain_level" in state.info["log"]
        assert "/velocity/velocity_error" in state.info["log"]
        assert "/velocity/terrain_normal_error" in state.info["log"]
        assert "/velocity/illegal_contact_count" in state.info["log"]
        assert "/velocity/terrain_curriculum_limit" in state.info["log"]
        assert "/velocity/push_count" in state.info["log"]
        for key in (
            "env_step_total_ms",
            "apply_action_ms",
            "step_core_ms",
            "backend_physics_ms",
            "backend_refresh_cache_ms",
            "update_state_ms",
            "reset_done_ms",
        ):
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
                obs={
                    "actor": np.zeros((2, 4), dtype=np.float32),
                    "critic": np.ones((2, 5), dtype=np.float32),
                },
                reward=np.zeros(2, dtype=np.float32),
                terminated=np.zeros(2, dtype=bool),
                truncated=np.zeros(2, dtype=bool),
                info={"steps": np.zeros(2, dtype=np.int64)},
            )
            self.closed = False

        def reset(self, seed=None):
            assert seed == 123
            return self.state.obs, {}

        def step(self, actions):
            assert isinstance(actions, np.ndarray)
            self.state.reward[:] = 1.0
            self.state.info["log"] = {"x": np.asarray(1.0, dtype=np.float32)}
            return self.state

        def close(self):
            self.closed = True

    env = DummyEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu")
    obs = wrapper.reset(seed=123)
    assert obs["actor"].shape == (2, 4)
    assert isinstance(obs["actor"], torch.Tensor)
    obs, reward, done, extras = wrapper.step(torch.zeros((2, 3), dtype=torch.float32))
    assert obs["critic"].shape == (2, 5)
    assert reward.tolist() == [1.0, 1.0]
    assert done.tolist() == [False, False]
    assert extras["log"]["x"].item() == 1.0
    assert isinstance(env.state.obs["actor"], np.ndarray)


def test_go1_velocity_terrain_normal_plane_fit():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env._height_scan_dim = 4
    state = VelocityRuntimeState(
        robot={},
        base={},
        joints={},
        links={},
        sensors={
            "terrain_scan": {
                "hits": [
                    {"hit": True, "point": [0.0, 0.0, 0.0]},
                    {"hit": True, "point": [1.0, 0.0, 0.1]},
                    {"hit": True, "point": [0.0, 1.0, 0.0]},
                    {"hit": True, "point": [1.0, 1.0, 0.1]},
                ]
            }
        },
        contacts=[],
    )

    normal = env._terrain_normal_from_scan(state)
    expected = np.array([-0.1, 0.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(normal, expected, atol=1.0e-5)


def test_go1_velocity_height_scan_requires_configured_sensor():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env._height_scan_dim = 4
    state = VelocityRuntimeState(robot={}, base={}, joints={}, links={}, sensors={}, contacts=[])

    _assert_raises_runtime_error("missing configured height scan sensor", lambda: env._height_scan(state))


def test_go1_velocity_height_scan_requires_valid_shape():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env._height_scan_dim = 4
    state = VelocityRuntimeState(
        robot={},
        base={},
        joints={},
        links={},
        sensors={"terrain_scan": {"hits": [], "values": [1.0, 2.0, 3.0, 4.0]}},
        contacts=[],
    )

    _assert_raises_runtime_error("expected 4", lambda: env._height_scan(state))


def test_go1_velocity_height_scan_allows_explicitly_disabled_sensor():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_flat_velocity_cfg(project_path="/tmp/go1")
    env._height_scan_dim = 0
    state = VelocityRuntimeState(robot={}, base={}, joints={}, links={}, sensors={}, contacts=[])

    assert env._height_scan(state).shape == (0,)


def test_go1_velocity_contact_summary_classifies_illegal_contacts():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env._foot_count = 4
    state = VelocityRuntimeState(
        robot={},
        base={},
        joints={},
        links={
            "FR_calf": {"global_transform": {"position": [10.0, 0.0, 0.0]}},
            "FL_calf": {"global_transform": {"position": [11.0, 0.0, 0.0]}},
            "RR_calf": {"global_transform": {"position": [12.0, 0.0, 0.0]}},
            "RL_calf": {"global_transform": {"position": [13.0, 0.0, 0.0]}},
        },
        sensors={},
        contacts=[
            {
                "robot_name": "go1",
                "link_name": "FR_thigh",
                "other_robot_name": "",
                "other_link_name": "",
                "position": [0.0, 0.0, 0.0],
                "normal_force": 10.0,
                "force": [0.0, 0.0, 10.0],
            },
            {
                "robot_name": "go1",
                "link_name": "trunk",
                "other_robot_name": "",
                "other_link_name": "",
                "position": [0.0, 0.0, 0.2],
                "normal_force": 3.0,
                "force": [0.0, 0.0, 3.0],
            },
            {
                "robot_name": "go1",
                "link_name": "FR_calf",
                "other_robot_name": "go1",
                "other_link_name": "FL_calf",
                "position": [0.0, 0.0, 0.4],
                "normal_force": 2.0,
                "force": [0.0, 0.0, 2.0],
            },
        ],
    )

    summary = env._contact_summary(state)
    assert summary["illegal"] == 1.0
    assert summary["trunk_head"] == 1.0
    assert summary["self_collision"] == 1.0


def test_go1_command_sampling_is_seed_reproducible():
    class DummyEnv:
        def __init__(self, seed):
            self.num_envs = 4
            self.step_dt = 0.02
            self._rng = np.random.default_rng(seed)

    cfg = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1").command
    first = UniformVelocityCommand(cfg, DummyEnv(123))
    second = UniformVelocityCommand(cfg, DummyEnv(123))
    env_ids = np.arange(4, dtype=np.int64)
    first.reset(env_ids)
    second.reset(env_ids)

    assert np.allclose(first.command_b, second.command_b)
    assert np.allclose(first.time_left, second.time_left)
    assert np.array_equal(first.is_heading_env, second.is_heading_env)


def test_go1_spawn_curriculum_is_seed_reproducible():
    first = _curriculum_env(seed=7)
    second = _curriculum_env(seed=7)
    assert [first._sample_spawn_index(0) for _ in range(8)] == [second._sample_spawn_index(0) for _ in range(8)]

    state = VelocityRuntimeState(
        robot={},
        base={"global_transform": {"position": [1.0, 0.0, 0.5]}},
        joints={},
        links={},
        sensors={},
        contacts=[],
    )
    first.episode_length_buf = np.array([80])
    second.episode_length_buf = np.array([80])
    first._update_terrain_curriculum_limit(0, state, reset_reason=2)
    second._update_terrain_curriculum_limit(0, state, reset_reason=2)
    assert np.allclose(first._terrain_curriculum_limits, second._terrain_curriculum_limits)


def test_go1_apply_actions_uses_batch_joint_api():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env.num_envs = 2
    env.num_actions = 3
    env.joint_names = ("j0", "j1", "j2")
    env.default_joint_pos = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    env.action_scale = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    calls = []

    class Runtime:
        def set_joint_position_targets(self, targets):
            calls.append(np.asarray(targets, dtype=np.float32).copy())

    env.runtime = Runtime()
    env._apply_actions(np.asarray([[1.0, -1.0, 0.0], [0.0, 0.5, -0.5]], dtype=np.float32))

    assert len(calls) == 1
    targets = calls[0]
    assert np.allclose(targets, [[0.6, -0.8, 0.3], [0.1, 0.7, -0.45]])


def test_go1_action_spec_matches_joint_order():
    env = object.__new__(Go1VelocityEnv)
    env.joint_names = ("j0", "j1", "j2")
    env.actor_obs_schema = velocity_actor_observation_schema(3, 0)

    spec = env._make_action_spec()

    assert spec.dim == 3
    assert spec.names == env.joint_names
    assert spec.metadata()["names"] == env.joint_names
    assert np.allclose(spec.clip([[2.0, -2.0, 0.5]]), [[1.0, -1.0, 0.5]])


def test_go1_policy_steps_count_environment_samples():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env.cfg_obj.terrain_curriculum_steps = 40
    env.num_envs = 4
    env._total_policy_steps = 0

    env.set_training_progress(12)
    assert env._total_policy_steps == 12
    assert np.isclose(env._curriculum_progress, 0.3)

    env._total_policy_steps += env.num_envs
    env._update_curriculum_progress()
    assert env._total_policy_steps == 16
    assert np.isclose(env._curriculum_progress, 0.4)


def test_go1_video_recorder_steps_eval_env_not_training_env(monkeypatch, tmp_path):
    torch = _require_torch()
    written: dict[str, object] = {}
    captures: list[dict[str, object]] = []

    class DummyImageIo:
        @staticmethod
        def imwrite(path, frames, fps, codec):
            written["path"] = path
            written["frames"] = frames.copy()
            written["fps"] = fps
            written["codec"] = codec

    class DummyPolicy:
        training = True

        def __init__(self):
            self.eval_calls = 0
            self.train_calls = 0

        def eval(self):
            self.eval_calls += 1
            self.training = False

        def train(self):
            self.train_calls += 1
            self.training = True

        def __call__(self, obs):
            batch = obs["actor"].shape[0]
            return torch.zeros((batch, 1), dtype=torch.float32)

    class DummyContext:
        root = None

        def clear_scene(self):
            pass

    class DummyEvalEnv:
        instances: list["DummyEvalEnv"] = []

        def __init__(
            self,
            cfg,
            *,
            num_envs,
            device,
            seed,
            max_episode_length,
            sim_workers,
            profile_step,
            collect_step_extras=True,
            context,
        ):
            self.cfg_obj = cfg
            self.num_envs = int(num_envs)
            self.device = device
            self.seed = seed
            self.max_episode_length = max_episode_length
            self.sim_workers = sim_workers
            self.profile_step = profile_step
            self.collect_step_extras = collect_step_extras
            self.context = context
            self.step_dt = 0.02
            self.joint_names = ("joint",)
            self.command_manager = type("Command", (), {"command_b": np.zeros((self.num_envs, 3), dtype=np.float32)})()
            self.reset_seeds: list[int | None] = []
            self.step_calls = 0
            self.closed = False
            self.state = BatchEnvState(
                obs=_obs_np(self.num_envs),
                reward=np.zeros(self.num_envs, dtype=np.float32),
                terminated=np.zeros(self.num_envs, dtype=bool),
                truncated=np.zeros(self.num_envs, dtype=bool),
                info={"steps": np.zeros(self.num_envs, dtype=np.int64)},
            )
            DummyEvalEnv.instances.append(self)

        def reset(self, seed=None):
            self.reset_seeds.append(seed)
            self.state.obs = _obs_np(self.num_envs)
            return self.state.obs, {}

        def step(self, actions):
            self.step_calls += 1
            self.state.obs = _obs_np(self.num_envs)
            self.state.info["steps"] += 1
            return self.state

        def _runtime_state(self, env_id):
            return VelocityRuntimeState(
                robot={},
                base={
                    "global_transform": {"position": [float(self.step_calls), 0.0, 0.4]},
                    "linear_velocity": [0.5, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                },
                joints={"joint": {"position": float(self.step_calls), "velocity": 0.0}},
                links={},
                sensors={},
                contacts=[],
            )

        def close(self):
            self.closed = True

    class TrainingEnv:
        def __init__(self):
            self.cfg_obj = go1_cfg.go1_flat_velocity_cfg(project_path="/tmp/go1")
            self.num_envs = 3
            self.device = "cpu"
            self.seed = 123
            self.max_episode_length = 99
            self.episode_length_buf = np.asarray([4, 5, 6], dtype=np.int64)
            self._total_policy_steps = 17
            self.get_observations_calls = 0
            self.step_calls = 0

        def get_observations(self):
            self.get_observations_calls += 1
            return _obs_np(self.num_envs)

        def step(self, actions):
            self.step_calls += 1
            raise AssertionError("training env must not be stepped by video recorder")

    monkeypatch.setattr(go1_velocity_video.gobot.app, "create_context", lambda: DummyContext())
    monkeypatch.setattr(go1_velocity_video, "Go1VelocityEnv", DummyEvalEnv)
    imageio_module = types.ModuleType("imageio")
    imageio_module.v3 = DummyImageIo
    monkeypatch.setitem(sys.modules, "imageio", imageio_module)
    monkeypatch.setitem(sys.modules, "imageio.v3", DummyImageIo)
    monkeypatch.setattr(
        go1_velocity_video.gobot.render,
        "capture_rgb",
        lambda **kwargs: captures.append(kwargs) or np.full((2, 3, 3), 127, dtype=np.uint8),
    )

    train_env = TrainingEnv()
    episode_lengths = train_env.episode_length_buf.copy()
    recorder = Go1TrainingVideoRecorder(
        train_env,
        Go1TrainingVideoCfg(
            interval=1,
            env_id=0,
            num_envs=2,
            seed=999,
            steps=3,
            fps=12,
            width=3,
            height=2,
            directory=tmp_path,
        ),
    )
    policy = DummyPolicy()

    path = recorder.record(2, policy)

    assert path == tmp_path / "go1_velocity_iter_000002.mp4"
    assert train_env.step_calls == 0
    assert train_env.get_observations_calls == 0
    assert train_env._total_policy_steps == 17
    assert np.array_equal(train_env.episode_length_buf, episode_lengths)
    assert len(DummyEvalEnv.instances) == 1
    assert DummyEvalEnv.instances[0].num_envs == 2
    assert DummyEvalEnv.instances[0].sim_workers == 1
    assert DummyEvalEnv.instances[0].profile_step is False
    assert DummyEvalEnv.instances[0].reset_seeds == [999]
    assert DummyEvalEnv.instances[0].step_calls == 3
    assert policy.eval_calls == 1
    assert policy.train_calls == 1
    assert written["fps"] == 12
    assert np.asarray(written["frames"]).shape == (3, 2, 3, 3)
    assert len(captures) == 3
    assert all("debug_arrows" in capture for capture in captures)
    assert all(len(capture["debug_arrows"]) == 1 for capture in captures)
    assert captures[0]["debug_arrows"][0].label == "actual_velocity"
    assert (tmp_path / "go1_velocity_iter_000002.replay.json").exists()
    assert (tmp_path / "go1_velocity_iter_000002.replay.npz").exists()
    replay = np.load(tmp_path / "go1_velocity_iter_000002.replay.npz")
    assert replay["debug_arrow_start"].shape == (3, 2, 3)
    assert replay["debug_arrow_vector"].shape == (3, 2, 3)
    assert replay["debug_arrow_color"].shape == (3, 2, 4)
    assert replay["debug_arrow_visible"].shape == (3, 2)
    assert not bool(replay["debug_arrow_visible"][0, 0])
    assert bool(replay["debug_arrow_visible"][0, 1])


def test_go1_video_recorder_invalid_eval_env_id_skips(monkeypatch, tmp_path):
    torch = _require_torch()

    class DummyPolicy:
        training = False

        def eval(self):
            pass

        def __call__(self, obs):
            return torch.zeros((obs["actor"].shape[0], 1), dtype=torch.float32)

    class TrainingEnv:
        cfg_obj = go1_cfg.go1_flat_velocity_cfg(project_path="/tmp/go1")
        num_envs = 4
        device = "cpu"
        seed = 1
        max_episode_length = 8
        step_calls = 0

        def step(self, actions):
            self.step_calls += 1
            raise AssertionError("training env must not be stepped")

    class EvalEnv:
        def __init__(self):
            self.num_envs = 1

    recorder = Go1TrainingVideoRecorder(
        TrainingEnv(),
        Go1TrainingVideoCfg(interval=1, env_id=2, num_envs=1, steps=1, directory=tmp_path),
    )
    recorder._eval_env = EvalEnv()

    assert recorder.record(0, DummyPolicy()) is None
    assert recorder.env.step_calls == 0


def test_go1_video_debug_arrows_rotate_command_to_world():
    cfg = go1_cfg.go1_flat_velocity_cfg(project_path="/tmp/go1")
    env = type(
        "Env",
        (),
        {"command_manager": type("Command", (), {"command_b": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)})()},
    )()
    recorder = Go1TrainingVideoRecorder(
        type("TrainingEnv", (), {"cfg_obj": cfg, "seed": 1})(),
        Go1TrainingVideoCfg(interval=1, env_id=0, steps=1),
    )
    half = float(np.sqrt(0.5))
    state = VelocityRuntimeState(
        robot={},
        base={
            "global_transform": {
                "position": [1.0, 2.0, 0.4],
                "quaternion": [float(half), 0.0, 0.0, float(half)],
            },
            "linear_velocity": [0.0, -0.5, 0.0],
        },
        joints={},
        links={},
        sensors={},
        contacts=[],
    )

    start, command_world, actual_world = recorder._velocity_arrow_vectors(env, state)
    arrows = recorder._debug_arrows(start, command_world, actual_world)

    assert np.allclose(start, [1.0, 2.0, 0.70])
    assert np.allclose(command_world, [0.0, 1.0, 0.0], atol=1.0e-5)
    assert np.allclose(actual_world, [0.0, -0.5, 0.0])
    assert [arrow.label for arrow in arrows] == ["command_velocity", "actual_velocity"]


def _obs(num_envs: int, device: str):
    torch = _require_torch()
    return {
        "actor": torch.zeros((num_envs, 1), dtype=torch.float32, device=device),
        "critic": torch.zeros((num_envs, 1), dtype=torch.float32, device=device),
        "policy": torch.zeros((num_envs, 1), dtype=torch.float32, device=device),
    }


def _obs_np(num_envs: int):
    return {
        "actor": np.zeros((num_envs, 1), dtype=np.float32),
        "critic": np.zeros((num_envs, 1), dtype=np.float32),
    }


def _curriculum_env(seed: int):
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env.cfg_obj.terrain_curriculum = True
    env._rng = np.random.default_rng(seed)
    env._curriculum_progress = 0.5
    env._spawn_origins = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    env._warmup_spawn_index = 0
    env._spawn_levels = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    env._spawn_order = np.arange(4, dtype=np.int64)
    env._terrain_curriculum_limits = np.zeros(1, dtype=np.float32)
    env._episode_start_xy = np.zeros((1, 2), dtype=np.float32)
    env.max_episode_length = 100
    env.step_dt = 0.02
    env.command_manager = type("Command", (), {"command_b": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)})()
    return env


def main():
    test_go1_velocity_cfg_dimensions()
    test_batch_env_state_contract_and_terminal_observation()
    test_task_ir_metadata_and_native_array_validation()
    test_go1_benchmark_metrics_match_unilab_env_step_accounting()
    test_go1_playback_schema_matches_training_schema()
    test_go1_velocity_terrain_normal_plane_fit()
    test_locomotion_unit_helpers()
    test_go1_velocity_contact_summary_classifies_illegal_contacts()
    test_go1_command_sampling_is_seed_reproducible()
    test_go1_spawn_curriculum_is_seed_reproducible()
    test_go1_apply_actions_uses_batch_joint_api()
    test_go1_policy_steps_count_environment_samples()
    try:
        test_rsl_rl_wrapper_keeps_core_env_numpy()
        test_go1_velocity_env_reset_step_shapes()
    except OptionalDependencyUnavailable as error:
        print(error)
        sys.exit(OPTIONAL_DEPENDENCY_SKIP_CODE)


if __name__ == "__main__":
    main()
