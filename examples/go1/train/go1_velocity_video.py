"""RGB snapshot video recording for Go1 velocity training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import gobot

try:
    from .go1_velocity_env import Go1VelocityEnv, VelocityRuntimeState
except ImportError:
    from go1_velocity_env import Go1VelocityEnv, VelocityRuntimeState


@dataclass
class Go1TrainingVideoCfg:
    interval: int = 100
    env_id: int = 0
    steps: int = 240
    fps: int = 30
    width: int = 640
    height: int = 480
    directory: Path | None = None


class Go1TrainingVideoRecorder:
    def __init__(self, env: Go1VelocityEnv, cfg: Go1TrainingVideoCfg) -> None:
        self.env = env
        self.cfg = cfg
        self.enabled = cfg.interval > 0 and cfg.steps > 0
        self._warned = False
        self._last_recorded_iteration: int | None = None
        self._node_cache: dict[str, Any] = {}

    def record(self, iteration: int, policy: Any) -> Path | None:
        if not self.enabled:
            return None
        if self._last_recorded_iteration == iteration:
            return None
        if iteration < 0 or iteration % self.cfg.interval != 0:
            return None
        if self.cfg.env_id < 0 or self.cfg.env_id >= self.env.num_envs:
            self._warn_once(f"render env id {self.cfg.env_id} is outside [0, {self.env.num_envs})")
            return None

        try:
            import imageio.v3 as iio
        except Exception as error:
            self._warn_once(f"imageio is unavailable; skipping Go1 training video capture: {error}")
            return None

        video_dir = self.cfg.directory or Path("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"go1_velocity_iter_{iteration:06d}.mp4"

        frames: list[np.ndarray] = []
        was_training = getattr(policy, "training", False)
        policy.eval()
        torch = self.env.torch
        obs = self.env.get_observations().to(self.env.device)
        try:
            with torch.inference_mode():
                for _ in range(self.cfg.steps):
                    actions = policy(obs)
                    obs, _, _, _ = self.env.step(actions.to(self.env.device))
                    state = self.env._runtime_state(self.cfg.env_id)
                    self._sync_scene_from_state(state)
                    frames.append(self._capture_frame(state))
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

        self._last_recorded_iteration = iteration
        print(f"Saved Go1 training video: {video_path}")
        return video_path

    def _sync_scene_from_state(self, state: VelocityRuntimeState) -> None:
        for link_name, link in state.links.items():
            transform = link.get("global_transform", {})
            position = transform.get("position")
            orientation = transform.get("quaternion")
            if position is None or orientation is None:
                continue
            node = self._find_node(link_name)
            if node is None or not hasattr(node, "set_global_transform"):
                continue
            node.set_global_transform(position, orientation)

    def _capture_frame(self, state: VelocityRuntimeState) -> np.ndarray:
        base_position = np.asarray(state.base.get("global_transform", {}).get("position", (0.0, 0.0, 0.4)), dtype=float)
        command = np.asarray(self.env.command_manager.command_b[self.cfg.env_id], dtype=float)
        heading = np.array([command[0], command[1], 0.0], dtype=float)
        if float(np.linalg.norm(heading[:2])) < 0.05:
            heading = np.array([1.0, 0.0, 0.0], dtype=float)
        heading /= max(float(np.linalg.norm(heading)), 1.0e-6)
        side = np.array([-heading[1], heading[0], 0.0], dtype=float)
        target = base_position + np.array([0.0, 0.0, 0.25], dtype=float)
        eye = target - heading * 2.2 + side * 0.65 + np.array([0.0, 0.0, 1.0], dtype=float)
        return gobot.render.capture_rgb(
            root=self.env.context.root,
            width=self.cfg.width,
            height=self.cfg.height,
            eye=eye.tolist(),
            target=target.tolist(),
            up=(0.0, 0.0, 1.0),
            fov_y=60.0,
            z_near=0.03,
            z_far=100.0,
        )

    def _find_node(self, name: str) -> Any | None:
        if name in self._node_cache:
            return self._node_cache[name]
        root = self.env.context.root
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
