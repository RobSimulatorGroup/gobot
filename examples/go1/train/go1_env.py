"""Vectorized Go1 locomotion environment for velocity tracking.

Observation (48-dim):
    base_lin_vel (3) + base_ang_vel (3) + projected_gravity (3) +
    cmd_vel (3: vx, vy, yaw) +
    joint_pos - default_pos (12) + joint_vel (12) + last_action (12)

Action (12-dim):
    target joint position offsets from the default standing pose.

Control:
    MuJoCo position actuators with PD-like behavior. kp is set on actuators,
    kd is applied as joint damping. The policy runs at 50 Hz by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

try:
    import mujoco
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "examples/go1 requires the Python mujoco package for training."
    ) from error

from rsl_rl.env import VecEnv


_EXAMPLE_DIR = Path(__file__).resolve().parent
GO1_XML = Path(
    os.environ.get(
        "GOBOT_GO1_XML",
        _EXAMPLE_DIR / "go1.xml",
    )
)

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


def _build_model(xml_path: str | Path = GO1_XML) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.option.timestep = 0.002
    spec.option.iterations = 10

    ground = spec.worldbody.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [50.0, 50.0, 0.1]
    ground.rgba = [0.8, 0.8, 0.8, 1.0]
    ground.friction = [1.0, 0.005, 0.0001]

    joint_name_set = set(JOINT_NAMES)
    for joint in spec.joints:
        if joint.name in joint_name_set:
            joint.damping[0] = KD

    for joint_name in JOINT_NAMES:
        actuator = spec.add_actuator()
        actuator.name = joint_name.replace("_joint", "")
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = joint_name
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.gainprm[:] = [KP] + [0.0] * 9
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[:] = [0.0, -KP, 0.0] + [0.0] * 7

    return spec.compile()


class Go1VecEnv(VecEnv):
    """rsl_rl VecEnv implemented directly with Python MuJoCo."""

    num_obs: int = 48

    def __init__(
        self,
        num_envs: int = 64,
        max_episode_length: int = 1000,
        device: str = "cpu",
        xml_path: str | Path = GO1_XML,
        seed: int = 42,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.device = device
        self.xml_path = Path(xml_path)
        self._rng = np.random.default_rng(seed)

        self._model = _build_model(self.xml_path)
        self._datas = [mujoco.MjData(self._model) for _ in range(self.num_envs)]
        self.num_actions = 12
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._cmds = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)

        self.cfg = {
            "num_envs": self.num_envs,
            "max_episode_length": self.max_episode_length,
            "xml_path": str(self.xml_path),
            "action_scale": ACTION_SCALE,
            "kp": KP,
            "kd": KD,
            "decimation": DECIMATION,
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

        for env_id, data in enumerate(self._datas):
            data.ctrl[:] = DEFAULT_POS + ACTION_SCALE * actions_np[env_id]
            for _ in range(DECIMATION):
                mujoco.mj_step(self._model, data)

            rewards[env_id] = self._compute_reward(data, self._cmds[env_id], actions_np[env_id])
            terminated = self._terminated(data)

            self.episode_length_buf[env_id] += 1
            timed_out = bool(self.episode_length_buf[env_id] >= self.max_episode_length)
            done = terminated or timed_out
            dones[env_id] = done
            time_outs[env_id] = timed_out and not terminated

            if done:
                self._reset_env(env_id)
                self.episode_length_buf[env_id] = 0

            self._last_actions[env_id] = actions_np[env_id]
            obs_list.append(self._get_obs(self._datas[env_id], self._cmds[env_id], self._last_actions[env_id]))

        self._obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        obs_td = TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        extras = {"time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device)}
        return (
            obs_td,
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            extras,
        )

    def close(self) -> None:
        pass

    def _compute_reward(self, data: mujoco.MjData, cmd: np.ndarray, action: np.ndarray) -> float:
        quat = data.qpos[3:7]
        base_lin_vel = _rotate_vec_by_quat_inv(data.qvel[0:3], quat)
        base_ang_vel = data.qvel[3:6].copy()
        vx_cmd, vy_cmd, yaw_cmd = cmd

        lin_vel_err = (base_lin_vel[0] - vx_cmd) ** 2 + (base_lin_vel[1] - vy_cmd) ** 2
        ang_vel_err = (base_ang_vel[2] - yaw_cmd) ** 2
        r_lin = np.exp(-lin_vel_err / 0.25)
        r_ang = np.exp(-ang_vel_err / 0.25)

        r_z_vel = -1.5 * float(data.qvel[2] ** 2)
        r_ang_xy = -0.05 * float(np.sum(base_ang_vel[:2] ** 2))
        r_action = -0.002 * float(np.sum(action ** 2))
        r_joint_vel = -0.001 * float(np.sum(data.qvel[6:] ** 2))
        r_height = -2.0 * max(0.0, 0.27 - float(data.qpos[2]))
        return float(r_lin + 0.5 * r_ang + r_z_vel + r_ang_xy + r_action + r_joint_vel + r_height)

    def _get_obs(self, data: mujoco.MjData, cmd: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        quat = data.qpos[3:7]
        base_lin_vel = _rotate_vec_by_quat_inv(data.qvel[0:3], quat)
        base_ang_vel = data.qvel[3:6].copy()
        projected_gravity = _rotate_vec_by_quat_inv(np.array([0.0, 0.0, -1.0]), quat)
        joint_pos_rel = (data.qpos[7:] - DEFAULT_POS).astype(np.float32)
        joint_vel = data.qvel[6:].astype(np.float32)
        return np.concatenate(
            [
                base_lin_vel.astype(np.float32),
                base_ang_vel.astype(np.float32),
                projected_gravity.astype(np.float32),
                cmd.astype(np.float32),
                joint_pos_rel,
                joint_vel,
                last_action.astype(np.float32),
            ]
        )

    def _terminated(self, data: mujoco.MjData) -> bool:
        trunk_z = float(data.qpos[2])
        roll, pitch = _quat_to_rp(data.qpos[3:7])
        return bool(trunk_z < 0.15 or abs(roll) > 0.8 or abs(pitch) > 0.8)

    def _reset_env(self, env_id: int) -> None:
        data = self._datas[env_id]
        mujoco.mj_resetData(self._model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.27]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = DEFAULT_POS + self._rng.uniform(-0.05, 0.05, self.num_actions)
        data.qvel[:] = self._rng.uniform(-0.05, 0.05, self._model.nv)
        data.ctrl[:] = DEFAULT_POS
        self._cmds[env_id] = _sample_cmd(self._rng)
        self._last_actions[env_id] = 0.0
        mujoco.mj_forward(self._model, data)

    def _reset_all(self) -> torch.Tensor:
        obs_list = []
        for env_id in range(self.num_envs):
            self._reset_env(env_id)
            obs_list.append(self._get_obs(self._datas[env_id], self._cmds[env_id], self._last_actions[env_id]))
        return torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)


def _sample_cmd(rng: np.random.Generator) -> np.ndarray:
    vx = rng.uniform(-1.0, 1.5)
    vy = rng.uniform(-0.5, 0.5)
    yaw = rng.uniform(-1.0, 1.0)
    return np.array([vx, vy, yaw], dtype=np.float32)


def _quat_to_rp(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def _rotate_vec_by_quat_inv(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return _quat_rotate(v, np.array([w, -x, -y, -z]))


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xyz = np.array([x, y, z])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)
