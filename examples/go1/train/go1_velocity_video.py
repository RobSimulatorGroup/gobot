"""RGB snapshot video recording for Go1 velocity training."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import gobot
from gobot.rl.rsl_rl import RslRlVecEnvWrapper
from gobot.rl.locomotion.math import _quat, _quat_to_yaw

try:
    from .go1_velocity_env import Go1VelocityEnv, VelocityRuntimeState
except ImportError:
    from go1_velocity_env import Go1VelocityEnv, VelocityRuntimeState


@dataclass
class Go1TrainingVideoCfg:
    interval: int = 100
    env_id: int = 0
    num_envs: int = 1
    seed: int | None = None
    steps: int = 240
    fps: int = 30
    width: int = 640
    height: int = 480
    directory: Path | None = None
    debug_arrows: bool = True


class Go1TrainingVideoRecorder:
    def __init__(self, env: Go1VelocityEnv, cfg: Go1TrainingVideoCfg) -> None:
        self.env = env
        self.cfg = cfg
        self.enabled = cfg.interval > 0 and cfg.steps > 0
        self._warned = False
        self._last_recorded_iteration: int | None = None
        self._node_cache: dict[str, Any] = {}
        self._eval_context: gobot.AppContext | None = None
        self._eval_env: Go1VelocityEnv | None = None

    def record(self, iteration: int, policy: Any) -> Path | None:
        if not self.enabled:
            return None
        if self._last_recorded_iteration == iteration:
            return None
        if iteration < 0 or iteration % self.cfg.interval != 0:
            return None
        eval_env = self._ensure_eval_env()
        if eval_env is None:
            return None
        if self.cfg.env_id < 0 or self.cfg.env_id >= eval_env.num_envs:
            self._warn_once(f"render env id {self.cfg.env_id} is outside [0, {eval_env.num_envs})")
            return None

        try:
            import imageio.v3 as iio
        except Exception as error:
            self._warn_once(f"imageio is unavailable; skipping Go1 training video capture: {error}")
            return None

        video_dir = self.cfg.directory or Path("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"go1_velocity_iter_{iteration:06d}.mp4"
        replay_prefix = video_dir / f"go1_velocity_iter_{iteration:06d}"

        frames: list[np.ndarray] = []
        replay_frames: list[dict[str, np.ndarray]] = []
        was_training = getattr(policy, "training", False)
        policy.eval()
        wrapper = RslRlVecEnvWrapper(eval_env, device=self.env.device)
        torch = wrapper.torch
        obs = wrapper.reset(seed=self._video_seed())
        try:
            with torch.inference_mode():
                for step_index in range(self.cfg.steps):
                    actions = policy(obs)
                    action_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
                    state_batch = eval_env.step(action_np)
                    obs = wrapper._tensor_obs(state_batch.obs)
                    state = eval_env._runtime_state(self.cfg.env_id)
                    self._sync_scene_from_state(eval_env, state)
                    debug_start, command_world, actual_world = self._velocity_arrow_vectors(eval_env, state)
                    debug_arrows = self._debug_arrows(debug_start, command_world, actual_world) if self.cfg.debug_arrows else []
                    frames.append(self._capture_frame(eval_env, state, debug_arrows))
                    replay_frames.append(
                        self._replay_frame(
                            eval_env,
                            state,
                            action_np,
                            step_index,
                            debug_start,
                            command_world,
                            actual_world,
                        )
                    )
        except Exception as error:
            self._warn_once(f"Go1 training video capture failed; continuing training: {error}")
            return None
        finally:
            if was_training:
                policy.train()

        if not frames:
            return None

        try:
            iio.imwrite(video_path, np.stack(frames, axis=0), fps=self.cfg.fps, codec="libx264")
        except Exception as error:
            self._warn_once(f"failed to write Go1 training video {video_path}: {error}")
            return None

        self._write_replay_bundle(iteration, replay_prefix, video_path, replay_frames)

        self._last_recorded_iteration = iteration
        print(f"Saved Go1 training video: {video_path}")
        return video_path

    def close(self) -> None:
        self._node_cache.clear()
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
        if self._eval_context is not None:
            self._eval_context.clear_scene()
            self._eval_context = None
        shutdown_capture = getattr(gobot.render, "_shutdown_headless_render_context", None)
        if shutdown_capture is not None:
            shutdown_capture()

    def _ensure_eval_env(self) -> Go1VelocityEnv | None:
        if self._eval_env is not None:
            return self._eval_env
        try:
            self._eval_context = gobot.app.create_context()
            self._eval_env = Go1VelocityEnv(
                copy.deepcopy(self.env.cfg_obj),
                num_envs=max(1, int(self.cfg.num_envs)),
                device=self.env.device,
                seed=self._video_seed(),
                max_episode_length=self.env.max_episode_length,
                sim_workers=1,
                profile_step=False,
                context=self._eval_context,
            )
        except Exception as error:
            self._warn_once(f"failed to initialize Go1 video eval environment: {error}")
            self.close()
            return None
        return self._eval_env

    def _video_seed(self) -> int:
        return int(self.cfg.seed if self.cfg.seed is not None else self.env.seed + 1_000_003)

    def _sync_scene_from_state(self, env: Go1VelocityEnv, state: VelocityRuntimeState) -> None:
        for link_name, link in state.links.items():
            transform = link.get("global_transform", {})
            position = transform.get("position")
            orientation = transform.get("quaternion")
            if position is None or orientation is None:
                continue
            node = self._find_node(env, link_name)
            if node is None or not hasattr(node, "set_global_transform"):
                continue
            node.set_global_transform(position, orientation)

    def _capture_frame(
        self,
        env: Go1VelocityEnv,
        state: VelocityRuntimeState,
        debug_arrows: list[gobot.render.DebugArrow],
    ) -> np.ndarray:
        base_position = np.asarray(state.base.get("global_transform", {}).get("position", (0.0, 0.0, 0.4)), dtype=float)
        command = np.asarray(env.command_b[self.cfg.env_id], dtype=float)
        heading = np.array([command[0], command[1], 0.0], dtype=float)
        if float(np.linalg.norm(heading[:2])) < 0.05:
            heading = np.array([1.0, 0.0, 0.0], dtype=float)
        heading /= max(float(np.linalg.norm(heading)), 1.0e-6)
        side = np.array([-heading[1], heading[0], 0.0], dtype=float)
        target = base_position + np.array([0.0, 0.0, 0.25], dtype=float)
        eye = target - heading * 2.2 + side * 0.65 + np.array([0.0, 0.0, 1.0], dtype=float)
        return gobot.render.capture_rgb(
            root=env.context.root,
            width=self.cfg.width,
            height=self.cfg.height,
            eye=eye.tolist(),
            target=target.tolist(),
            up=(0.0, 0.0, 1.0),
            fov_y=60.0,
            z_near=0.03,
            z_far=100.0,
            debug_arrows=debug_arrows,
        )

    def _replay_frame(
        self,
        env: Go1VelocityEnv,
        state: VelocityRuntimeState,
        actions: Any,
        step_index: int,
        debug_start: np.ndarray,
        command_world: np.ndarray,
        actual_world: np.ndarray,
    ) -> dict[str, np.ndarray]:
        action_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_np = np.asarray(action_np, dtype=np.float32)
        if action_np.ndim >= 2:
            action_np = action_np[self.cfg.env_id]
        command = np.asarray(env.command_b[self.cfg.env_id], dtype=np.float32)
        base_transform = state.base.get("global_transform", {})
        base_position = np.asarray(base_transform.get("position", (0.0, 0.0, 0.0)), dtype=np.float32)
        base_quaternion = np.asarray(base_transform.get("quaternion", (1.0, 0.0, 0.0, 0.0)), dtype=np.float32)
        joint_position = np.asarray(
            [float(state.joints.get(name, {}).get("position", 0.0)) for name in env.joint_names],
            dtype=np.float32,
        )
        joint_velocity = np.asarray(
            [float(state.joints.get(name, {}).get("velocity", 0.0)) for name in env.joint_names],
            dtype=np.float32,
        )
        debug_arrow_start = np.stack(
            [debug_start, debug_start + np.array([0.0, 0.0, 0.08], dtype=np.float32)],
            axis=0,
        ).astype(np.float32, copy=False)
        debug_arrow_vector = np.stack([command_world, actual_world], axis=0).astype(np.float32, copy=False)
        debug_arrow_color = np.asarray(
            [[0.15, 0.85, 0.20, 1.0], [0.10, 0.42, 1.0, 1.0]],
            dtype=np.float32,
        )
        debug_arrow_visible = np.asarray(
            [
                self.cfg.debug_arrows and float(np.linalg.norm(command_world[:2])) > 1.0e-4,
                self.cfg.debug_arrows and float(np.linalg.norm(actual_world[:2])) > 1.0e-4,
            ],
            dtype=np.bool_,
        )
        return {
            "step": np.asarray(step_index, dtype=np.int32),
            "time": np.asarray(step_index * env.step_dt, dtype=np.float32),
            "action": action_np.astype(np.float32, copy=False),
            "command": command,
            "base_position": base_position,
            "base_quaternion": base_quaternion,
            "base_linear_velocity": np.asarray(state.base.get("linear_velocity", (0.0, 0.0, 0.0)), dtype=np.float32),
            "base_angular_velocity": np.asarray(state.base.get("angular_velocity", (0.0, 0.0, 0.0)), dtype=np.float32),
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "debug_arrow_start": debug_arrow_start,
            "debug_arrow_vector": debug_arrow_vector,
            "debug_arrow_color": debug_arrow_color,
            "debug_arrow_visible": debug_arrow_visible,
        }

    def _debug_arrows(
        self,
        start: np.ndarray,
        command_world: np.ndarray,
        actual_world: np.ndarray,
    ) -> list[gobot.render.DebugArrow]:
        arrows: list[gobot.render.DebugArrow] = []
        if float(np.linalg.norm(command_world[:2])) > 1.0e-4:
            arrows.append(
                gobot.render.DebugArrow(
                    start=start.tolist(),
                    vector=command_world.tolist(),
                    color=(0.15, 0.85, 0.20, 1.0),
                    scale=0.55,
                    label="command_velocity",
                )
            )
        if float(np.linalg.norm(actual_world[:2])) > 1.0e-4:
            arrows.append(
                gobot.render.DebugArrow(
                    start=(start + np.array([0.0, 0.0, 0.08], dtype=np.float32)).tolist(),
                    vector=actual_world.tolist(),
                    color=(0.10, 0.42, 1.0, 1.0),
                    scale=0.55,
                    label="actual_velocity",
                )
            )
        return arrows

    def _velocity_arrow_vectors(
        self,
        env: Go1VelocityEnv,
        state: VelocityRuntimeState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_transform = state.base.get("global_transform", {})
        base_position = np.asarray(base_transform.get("position", (0.0, 0.0, 0.0)), dtype=np.float32)
        start = base_position + np.array([0.0, 0.0, 0.30], dtype=np.float32)
        command_b = np.asarray(env.command_b[self.cfg.env_id], dtype=np.float32)
        yaw = _quat_to_yaw(_quat(state.base))
        command_world = np.array(
            [
                math.cos(yaw) * command_b[0] - math.sin(yaw) * command_b[1],
                math.sin(yaw) * command_b[0] + math.cos(yaw) * command_b[1],
                0.0,
            ],
            dtype=np.float32,
        )
        actual_world = np.asarray(state.base.get("linear_velocity", (0.0, 0.0, 0.0)), dtype=np.float32)
        actual_world = np.array([actual_world[0], actual_world[1], 0.0], dtype=np.float32)
        return start, command_world, actual_world

    def _write_replay_bundle(
        self,
        iteration: int,
        replay_prefix: Path,
        video_path: Path,
        frames: list[dict[str, np.ndarray]],
    ) -> None:
        if not frames or self._eval_env is None:
            return
        metadata_path = replay_prefix.with_suffix(".replay.json")
        data_path = replay_prefix.with_suffix(".replay.npz")
        scene_path = replay_prefix.with_suffix(".replay.jscn")
        metadata: dict[str, Any] = {
            "version": "gobot.go1.replay.v1",
            "iteration": int(iteration),
            "video": video_path.name,
            "data": data_path.name,
            "scene": scene_path.name,
            "env_id": int(self.cfg.env_id),
            "seed": int(self._video_seed()),
            "steps": int(len(frames)),
            "dt": float(self._eval_env.step_dt),
            "fps": int(self.cfg.fps),
            "robot": self._eval_env.cfg_obj.robot_name,
            "base_link": self._eval_env.cfg_obj.base_link,
            "joint_names": list(self._eval_env.joint_names),
            "command_names": ["vx", "vy", "yaw"],
            "debug_overlays": [
                {"label": "command_velocity", "source": "command_velocity_world", "color": [0.15, 0.85, 0.20, 1.0], "scale": 0.55},
                {"label": "actual_velocity", "source": "actual_velocity_world", "color": [0.10, 0.42, 1.0, 1.0], "scale": 0.55},
            ] if self.cfg.debug_arrows else [],
        }
        try:
            if self._eval_env.context.root is not None:
                gobot.save_scene(self._eval_env.context.root, str(scene_path))
        except Exception as error:
            metadata["scene_error"] = str(error)
        try:
            np.savez_compressed(
                data_path,
                step=np.asarray([frame["step"] for frame in frames], dtype=np.int32),
                time=np.asarray([frame["time"] for frame in frames], dtype=np.float32),
                action=np.stack([frame["action"] for frame in frames], axis=0),
                command=np.stack([frame["command"] for frame in frames], axis=0),
                base_position=np.stack([frame["base_position"] for frame in frames], axis=0),
                base_quaternion=np.stack([frame["base_quaternion"] for frame in frames], axis=0),
                base_linear_velocity=np.stack([frame["base_linear_velocity"] for frame in frames], axis=0),
                base_angular_velocity=np.stack([frame["base_angular_velocity"] for frame in frames], axis=0),
                joint_position=np.stack([frame["joint_position"] for frame in frames], axis=0),
                joint_velocity=np.stack([frame["joint_velocity"] for frame in frames], axis=0),
                debug_arrow_start=np.stack([frame["debug_arrow_start"] for frame in frames], axis=0),
                debug_arrow_vector=np.stack([frame["debug_arrow_vector"] for frame in frames], axis=0),
                debug_arrow_color=np.stack([frame["debug_arrow_color"] for frame in frames], axis=0),
                debug_arrow_visible=np.stack([frame["debug_arrow_visible"] for frame in frames], axis=0),
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as error:
            self._warn_once(f"failed to write Go1 replay bundle {metadata_path}: {error}")

    def _find_node(self, env: Go1VelocityEnv, name: str) -> Any | None:
        if name in self._node_cache:
            return self._node_cache[name]
        root = env.context.root
        node = None
        if root is not None:
            node = root.find(name)
            if node is None:
                node = self._find_node_recursive(root, name)
        self._node_cache[name] = node
        return node

    def _find_node_recursive(self, node: Any, name: str) -> Any | None:
        if getattr(node, "name", None) == name:
            return node
        for child in getattr(node, "children", ()):
            found = self._find_node_recursive(child, name)
            if found is not None:
                return found
        return None

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        print(f"Warning: {message}")
        self._warned = True


class VideoCheckpointRunnerMixin:
    video_recorder: Go1TrainingVideoRecorder | None

    def save(self, path: str, infos: Mapping[str, Any] | None = None) -> None:
        super().save(path, infos=infos)  # type: ignore[misc]
        if self.video_recorder is None:
            return
        if not Path(path).stem.startswith("model_"):
            return
        policy = self.get_inference_policy(device=self.device)
        self.video_recorder.record(int(self.current_learning_iteration), policy)


__all__ = [
    "Go1TrainingVideoCfg",
    "Go1TrainingVideoRecorder",
    "VideoCheckpointRunnerMixin",
]
