"""Scene-authored Cartpole position tracking on Gobot's MuJoCo CPU batch."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import gobot
from gobot.rl import BatchSimulationRuntime, CpuBatchEnv

from ..cartpole_task import (
    BASE_LINK, ROBOT, JOINT_NAMES, FIXED_DT, FORCE_LIMIT,
    DISTURBANCE_STD, DISTURBANCE_CLIP, DISTURBANCE_INTERVAL_TICKS,
    DISTURBANCE_DURATION_TICKS, DISTURBANCE_START_TICK,
    configure_robot, observation,
)

PROJECT_PATH = Path(__file__).resolve().parents[1]


class CartPoleEnv(CpuBatchEnv):
    """NumPy task; learner integration lives in gobot.rl.rsl_rl."""

    num_obs = 7
    num_actions = 1

    def __init__(
        self, num_envs=64, max_episode_length=1000, *,
        scene_path=PROJECT_PATH / "cartpole.jscn", sim_workers=0,
        disturbance_interval=DISTURBANCE_INTERVAL_TICKS,
        disturbance_duration=DISTURBANCE_DURATION_TICKS,
        disturbance_start=DISTURBANCE_START_TICK,
        disturbance_std=DISTURBANCE_STD, disturbance_clip=DISTURBANCE_CLIP,
        target_range=0.8, action_limit=FORCE_LIMIT, seed=42,
    ):
        super().__init__(num_envs=num_envs, seed=seed)
        if int(max_episode_length) <= 0 or int(sim_workers) < 0:
            raise ValueError("episode length must be positive and workers non-negative")
        values = (disturbance_std, disturbance_clip, target_range, action_limit)
        if not np.isfinite(values).all() or min(values) < 0 or action_limit == 0:
            raise ValueError("task limits must be finite and non-negative; action_limit must be positive")
        self.max_episode_length = int(max_episode_length)
        self.sim_workers = int(sim_workers)
        self.scene_path = Path(scene_path).expanduser().resolve()
        self.disturbance_interval = int(disturbance_interval)
        self.disturbance_duration = int(disturbance_duration)
        self.disturbance_start = int(disturbance_start)
        self.disturbance_std = float(disturbance_std)
        self.disturbance_clip = float(disturbance_clip)
        self.target_range = float(target_range)
        self.action_limit = float(action_limit)
        self.episode_length_buf = np.zeros(self.num_envs, dtype=np.int64)
        self._targets = np.zeros(self.num_envs, dtype=np.float32)
        self._efforts = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.cfg = dict(num_envs=self.num_envs, max_episode_length=self.max_episode_length,
                        scene_path=str(self.scene_path), physics_dt=FIXED_DT,
                        disturbance_interval=self.disturbance_interval,
                        disturbance_duration=self.disturbance_duration,
                        disturbance_start=self.disturbance_start,
                        disturbance_std=self.disturbance_std, disturbance_clip=self.disturbance_clip,
                        target_range=self.target_range, action_limit=self.action_limit)
        self.context = gobot.app.create_context()
        self._closed = False
        try:
            self.context.set_project_path(str(self.scene_path.parent))
            root = self.context.load_scene("res://" + self.scene_path.name)
            robot = root if root.name == ROBOT else root.find(ROBOT)
            if robot is None:
                raise ValueError(f"scene has no {ROBOT!r} robot")
            configure_robot(robot, action_limit=self.action_limit, disturbance_clip=self.disturbance_clip)
            self.context.fixed_time_step = FIXED_DT
            self.context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
            self.runtime = BatchSimulationRuntime(self.context, robot=ROBOT,
                                                  base_link=BASE_LINK, joint_names=JOINT_NAMES)
            self.runtime.configure(self.num_envs)
            self._state = self.make_empty_state(self.obs_groups_spec)
            self.reset(seed=seed)
        except Exception:
            self.close()
            raise

    @property
    def obs_groups_spec(self):
        return {"policy": self.num_obs}

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Cartpole environment is closed")

    def reset(self, *, seed=None):
        self._ensure_open()
        self.reset_seed(seed)
        self._reset_envs(np.arange(self.num_envs))
        self._state.reward.fill(0)
        self._state.terminated.fill(False)
        self._state.truncated.fill(False)
        self._state.info = {"steps": self.episode_length_buf}
        self.clear_step_final_observation()
        self._refresh_observations()
        return {key: value.copy() for key, value in self._state.obs.items()}, dict(self._state.info)

    def _reset_envs(self, env_ids):
        for env_id in env_ids:
            self.runtime.reset_env(int(env_id))
            positions = self._rng.uniform((-0.5, -0.1), (0.5, 0.1))
            velocities = self._rng.uniform(-0.1, 0.1, size=2)
            for index, joint in enumerate(JOINT_NAMES):
                self.runtime.reset_joint_state(int(env_id), joint, positions[index], velocities[index])
            self._targets[env_id] = self._rng.uniform(-self.target_range, self.target_range)
        self._efforts[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _refresh_observations(self):
        state = self.runtime.robot_state()
        positions = np.asarray(state["joint_position"])
        velocities = np.asarray(state["joint_velocity"])
        self._state.obs["policy"][:] = observation(positions, velocities, self._targets)
        return positions, velocities

    def _sample_disturbances(self):
        active = self.episode_length_buf >= self.disturbance_start
        if self.disturbance_interval > 0:
            offset = (self.episode_length_buf - self.disturbance_start) % self.disturbance_interval
            active &= offset < max(1, self.disturbance_duration)
        noise = np.zeros(self.num_envs)
        noise[active] = self._rng.normal(0.0, self.disturbance_std, size=np.count_nonzero(active))
        if self.disturbance_clip > 0:
            np.clip(noise, -self.disturbance_clip, self.disturbance_clip, out=noise)
        return noise

    def step(self, actions):
        self._ensure_open()
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_envs, 1) or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite with shape ({self.num_envs}, 1)")
        self.clear_step_final_observation()
        self._efforts[:, 0] = np.clip(actions[:, 0], -self.action_limit, self.action_limit)
        self._efforts[:, 1] = self._sample_disturbances()
        self.runtime.set_joint_effort_targets(self._efforts)
        self.runtime.step(1, workers=self.sim_workers)
        positions, velocities = self._refresh_observations()
        x, theta = positions[:, 0], positions[:, 1]
        state = self._state
        state.reward[:] = np.cos(theta) - np.abs(x - self._targets) - 0.01 * np.square(velocities).sum(axis=1)
        state.terminated[:] = (np.abs(x) > 0.95) | (np.abs(np.arctan2(np.sin(theta), np.cos(theta))) > 0.5)
        self.episode_length_buf += 1
        state.truncated[:] = (self.episode_length_buf >= self.max_episode_length) & ~state.terminated
        env_ids = np.flatnonzero(state.done)
        if self.autoreset and env_ids.size:
            self.capture_final_observation(env_ids)
            self._reset_envs(env_ids)
            self._refresh_observations()
        state.info["time_outs"] = state.truncated.copy()
        state.info["log"] = {
            "/cartpole/target_abs_mean": float(np.abs(self._targets).mean()),
            "/cartpole/pos_error_abs_mean": float(np.abs(state.obs["policy"][:, 6]).mean()),
        }
        self.step_counter += 1
        return state

    def close(self):
        if not self._closed:
            self.context.clear_scene()
            self._closed = True
