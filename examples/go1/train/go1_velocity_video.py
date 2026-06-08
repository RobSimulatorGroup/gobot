"""RGB snapshot video recording for Go1 velocity training."""

from __future__ import annotations

from dataclasses import dataclass
import copy
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
    num_envs: int = 1
    seed: int | None = None
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

        frames: list[np.ndarray] = []
        was_training = getattr(policy, "training", False)
        policy.eval()
        torch = eval_env.torch
        obs = eval_env.reset(seed=self._video_seed()).to(eval_env.device)
        try:
            with torch.inference_mode():
                for _ in range(self.cfg.steps):
                    actions = policy(obs)
                    obs, _, _, _ = eval_env.step(actions.to(eval_env.device))
                    state = eval_env._runtime_state(self.cfg.env_id)
                    self._sync_scene_from_state(eval_env, state)
                    frames.append(self._capture_frame(eval_env, state))
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

    def close(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
        if self._eval_context is not None:
            self._eval_context.clear_scene()
            self._eval_context = None
        shutdown_capture = getattr(gobot.render, "_shutdown_headless_render_context", None)
        if shutdown_capture is not None:
            shutdown_capture()
        self._node_cache.clear()

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

    def _capture_frame(self, env: Go1VelocityEnv, state: VelocityRuntimeState) -> np.ndarray:
        base_position = np.asarray(state.base.get("global_transform", {}).get("position", (0.0, 0.0, 0.4)), dtype=float)
        command = np.asarray(env.command_manager.command_b[self.cfg.env_id], dtype=float)
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
        )

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
