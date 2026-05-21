"""Gobot-backed Go1 velocity-tracking environment for rsl_rl.

This intentionally uses Gobot's scene and simulation APIs instead of importing
MuJoCo directly. The editable source of truth is ``examples/go1/go1_scene.jscn``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import gobot
from rsl_rl.env import VecEnv


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = "res://go1_scene.jscn"
ROBOT = "go1"
BASE_LINK = "trunk"

JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

DEFAULT_POS = np.array(
    [
        0.0,
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
    ],
    dtype=np.float64,
)

ACTION_SCALE = 0.25
KP = 40.0
KD = 1.0
DECIMATION = 10
FIXED_TIME_STEP = 0.002


class Go1VecEnv(VecEnv):
    """Vectorized Go1 task backed by one Gobot MuJoCo model and many runtime states."""

    num_obs: int = 48

    def __init__(
        self,
        num_envs: int = 64,
        max_episode_length: int = 1000,
        device: str = "cpu",
        project_path: str | Path = EXAMPLE_ROOT,
        scene_path: str = SCENE_PATH,
        decimation: int = DECIMATION,
        seed: int = 42,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.device = device
        self.project_path = Path(project_path)
        self.scene_path = scene_path
        self.decimation = int(decimation)
        self.num_actions = len(JOINT_NAMES)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        self._rng = np.random.default_rng(seed)
        self._cmds = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)

        self.context = gobot.app.context()
        self.context.set_project_path(str(self.project_path))
        self.context.load_scene(self.scene_path)
        self.context.fixed_time_step = FIXED_TIME_STEP
        self.context.set_default_joint_gains(
            {
                "position_stiffness": KP,
                "velocity_damping": KD,
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self.context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
        self.context.configure_batch_world(self.num_envs)

        self.cfg = {
            "num_envs": self.num_envs,
            "max_episode_length": self.max_episode_length,
            "project_path": str(self.project_path),
            "scene_path": self.scene_path,
            "backend": "MuJoCoCpu",
            "robot": ROBOT,
            "base_link": BASE_LINK,
            "joint_names": JOINT_NAMES,
            "action_scale": ACTION_SCALE,
            "kp": KP,
            "kd": KD,
            "decimation": self.decimation,
            "fixed_time_step": FIXED_TIME_STEP,
            "seed": seed,
        }
        self._obs = self._reset_all()

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self, seed: int | None = None) -> TensorDict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_length_buf.zero_()
        self._obs = self._reset_all()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().reshape(self.num_envs, self.num_actions)
        actions_np = np.clip(actions_np, -1.0, 1.0)

        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)

        base_heights = np.zeros(self.num_envs, dtype=np.float32)
        for env_id in range(self.num_envs):
            action = actions_np[env_id]
            self._apply_action(env_id, action)
            self.context.step_batch_env(env_id, self.decimation)

            state = self._robot_state(env_id)
            rewards[env_id] = self._compute_reward(state, self._cmds[env_id], action)
            terminated = self._terminated(state)

            self.episode_length_buf[env_id] += 1
            timed_out = bool(self.episode_length_buf[env_id] >= self.max_episode_length)
            done = terminated or timed_out
            dones[env_id] = done
            time_outs[env_id] = timed_out and not terminated

            self._last_actions[env_id] = action
            if done:
                self._reset_env(env_id)
                self.episode_length_buf[env_id] = 0
                state = self._robot_state(env_id)

            base_heights[env_id] = self._base_link(state)["position"][2]
            obs_list.append(self._get_obs(state, self._cmds[env_id], self._last_actions[env_id]))

        self._obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        obs_td = TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        extras = {
            "time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": {
                "/go1/command_vx": torch.as_tensor(float(np.mean(self._cmds[:, 0])), device=self.device),
                "/go1/base_height": torch.as_tensor(float(np.mean(base_heights)), device=self.device),
            },
        }
        return (
            obs_td,
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            extras,
        )

    def close(self) -> None:
        self.context.clear_world()

    def _apply_action(self, env_id: int, action: np.ndarray) -> None:
        target_pos = DEFAULT_POS + ACTION_SCALE * action
        for joint_name, target in zip(JOINT_NAMES, target_pos, strict=True):
            self.context.set_batch_joint_position_target(env_id, ROBOT, joint_name, float(target))

    def _compute_reward(self, state: dict, cmd: np.ndarray, action: np.ndarray) -> float:
        base = self._base_link(state)
        world_lin_vel = np.asarray(base["linear_velocity"], dtype=np.float64)
        base_lin_vel = _rotate_vec_by_quat_inv(world_lin_vel, base["quaternion"])
        base_ang_vel = np.asarray(base["angular_velocity"], dtype=np.float64)
        vx_cmd, vy_cmd, yaw_cmd = cmd

        lin_vel_err = (base_lin_vel[0] - vx_cmd) ** 2 + (base_lin_vel[1] - vy_cmd) ** 2
        ang_vel_err = (base_ang_vel[2] - yaw_cmd) ** 2
        r_lin = np.exp(-lin_vel_err / 0.25)
        r_ang = np.exp(-ang_vel_err / 0.25)
        r_z_vel = -1.5 * world_lin_vel[2] ** 2
        r_ang_xy = -0.05 * float(np.sum(base_ang_vel[:2] ** 2))
        r_action = -0.002 * float(np.sum(action**2))
        joint_vel = np.asarray(self._joint_values(state)[1], dtype=np.float64)
        r_joint_vel = -0.001 * float(np.sum(joint_vel**2))
        r_height = -2.0 * max(0.0, 0.27 - base["position"][2])
        return float(r_lin + 0.5 * r_ang + r_z_vel + r_ang_xy + r_action + r_joint_vel + r_height)

    def _terminated(self, state: dict) -> bool:
        base = self._base_link(state)
        roll, pitch = _quat_to_rp(base["quaternion"])
        return bool(base["position"][2] < 0.15 or abs(roll) > 0.8 or abs(pitch) > 0.8)

    def _get_obs(self, state: dict, cmd: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        base = self._base_link(state)
        quat = base["quaternion"]
        base_lin_vel = _rotate_vec_by_quat_inv(np.asarray(base["linear_velocity"], dtype=np.float64), quat)
        base_ang_vel = np.asarray(base["angular_velocity"], dtype=np.float64)
        projected_gravity = _rotate_vec_by_quat_inv(np.array([0.0, 0.0, -1.0], dtype=np.float64), quat)
        joint_pos, joint_vel = self._joint_values(state)
        joint_pos_rel = np.asarray(joint_pos, dtype=np.float32) - DEFAULT_POS.astype(np.float32)

        return np.concatenate(
            [
                base_lin_vel.astype(np.float32),
                base_ang_vel.astype(np.float32),
                projected_gravity.astype(np.float32),
                cmd.astype(np.float32),
                joint_pos_rel.astype(np.float32),
                np.asarray(joint_vel, dtype=np.float32),
                last_action.astype(np.float32),
            ]
        )

    def _reset_env(self, env_id: int) -> None:
        self.context.reset_batch_env(env_id)
        self.context.reset_batch_link_state(
            env_id,
            ROBOT,
            BASE_LINK,
            [0.0, 0.0, 0.27],
            [1.0, 0.0, 0.0, 0.0],
            self._rng.uniform(-0.05, 0.05, 3).tolist(),
            self._rng.uniform(-0.05, 0.05, 3).tolist(),
        )
        for joint_name, default_pos in zip(JOINT_NAMES, DEFAULT_POS, strict=True):
            position = float(default_pos + self._rng.uniform(-0.05, 0.05))
            velocity = float(self._rng.uniform(-0.05, 0.05))
            self.context.reset_batch_joint_state(env_id, ROBOT, joint_name, position, velocity)
            self.context.set_batch_joint_position_target(env_id, ROBOT, joint_name, float(default_pos))

        self._cmds[env_id] = _sample_cmd(self._rng)
        self._last_actions[env_id] = 0.0

    def _reset_all(self) -> torch.Tensor:
        obs_list = []
        for env_id in range(self.num_envs):
            self._reset_env(env_id)
            obs_list.append(self._get_obs(self._robot_state(env_id), self._cmds[env_id], self._last_actions[env_id]))
        return torch.as_tensor(
            np.stack(obs_list),
            dtype=torch.float32,
            device=self.device,
        )

    def _robot_state(self, env_id: int) -> dict:
        runtime = self.context.get_batch_runtime_state(env_id)
        for robot in runtime.get("robots", []):
            if robot.get("name") == ROBOT:
                return robot
        raise RuntimeError(f"Gobot runtime state has no robot '{ROBOT}'")

    def _base_link(self, state: dict) -> dict:
        for link in state.get("links", []):
            if link.get("name") == BASE_LINK:
                transform = link.get("global_transform", {})
                return {
                    "position": np.asarray(transform.get("position", [0.0, 0.0, 0.0]), dtype=np.float64),
                    "quaternion": np.asarray(transform.get("quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64),
                    "linear_velocity": np.asarray(link.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
                    "angular_velocity": np.asarray(link.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
                }
        raise RuntimeError(f"Gobot runtime state has no link '{BASE_LINK}'")

    def _joint_values(self, state: dict) -> tuple[list[float], list[float]]:
        joints = {joint.get("name"): joint for joint in state.get("joints", [])}
        joint_pos = []
        joint_vel = []
        for default_pos, joint_name in zip(DEFAULT_POS, JOINT_NAMES, strict=True):
            joint = joints.get(joint_name, {})
            joint_pos.append(float(joint.get("position", default_pos)))
            joint_vel.append(float(joint.get("velocity", 0.0)))
        return joint_pos, joint_vel


def _sample_cmd(rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(-1.0, 1.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(-1.0, 1.0),
        ],
        dtype=np.float32,
    )


def _quat_to_rp(q: np.ndarray | list[float]) -> tuple[float, float]:
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


def _rotate_vec_by_quat_inv(v: np.ndarray, q: np.ndarray | list[float]) -> np.ndarray:
    w, x, y, z = q
    return _quat_rotate(v, np.array([w, -x, -y, -z], dtype=np.float64))


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z], dtype=np.float64), v)
    return v + w * t + np.cross(np.array([x, y, z], dtype=np.float64), t)
