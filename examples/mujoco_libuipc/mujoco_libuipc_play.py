"""Editor Play entry point for the MuJoCo Warp + libuipc batch demo."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

# Load Torch's CUDA runtime before the native libuipc solver module.
import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
    MuJoCoWarpProvider,
)


SCENE_ROOT_NAME = "mujoco_libuipc_soft_press"
FIXED_DT = 0.002
NUM_ENVS = 4
ENVIRONMENTS_PER_SHARD = 4
DISPLAY_ENV = NUM_ENVS - 1
PRESS_DEPTH = 0.17
SETTLE_TICKS = 16
PRESS_TICKS = 128
HOLD_TICKS = 64
RELEASE_TICKS = 96
CYCLE_TICKS = SETTLE_TICKS + PRESS_TICKS + HOLD_TICKS + RELEASE_TICKS
SOLVER_MODULE_NAME = "libgobot_libuipc_solver.so"


def _nodes_by_name(root: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in result:
            raise RuntimeError(
                f"MuJoCo+libuipc demo has duplicate node name {node.name!r}"
            )
        result[node.name] = node
        pending.extend(node.children)
    return result


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _press_progress(tick: int) -> float:
    if tick < SETTLE_TICKS:
        return 0.0
    tick -= SETTLE_TICKS
    if tick < PRESS_TICKS:
        return _smoothstep(float(tick + 1) / float(PRESS_TICKS))
    tick -= PRESS_TICKS
    if tick < HOLD_TICKS:
        return 1.0
    tick -= HOLD_TICKS
    return 1.0 - _smoothstep(float(tick + 1) / float(RELEASE_TICKS))


def _repository_root(project_path: str) -> Path | None:
    current = Path(project_path).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "python" / "gobot"
        ).is_dir():
            return candidate
    return None


def _solver_module_path(project_path: str) -> str:
    configured = os.environ.get("GOBOT_LIBUIPC_SOLVER_MODULE", "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())

    repository = _repository_root(project_path)
    if repository is None:
        return ""
    candidates = [
        repository
        / "build"
        / "libuipc-novcpkg"
        / "python"
        / "gobot"
        / SOLVER_MODULE_NAME
    ]
    candidates.extend(
        sorted(
            (repository / "build").glob(
                "*/python/gobot/" + SOLVER_MODULE_NAME
            )
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def _batch_config(context: Any) -> LibuipcBatchConfig:
    return LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=FIXED_DT,
            friction_coefficient=0.8,
            module_path=_solver_module_path(context.project_path),
            workspace=str(
                Path(tempfile.gettempdir()) / "gobot-mujoco-libuipc-editor"
            ),
        ),
        environments_per_shard=ENVIRONMENTS_PER_SHARD,
    )


class Script(gobot.NodeScript):
    """Run the GPU batch and render its deepest-press environment in Play Mode."""

    def _ready(self) -> None:
        self.provider = None
        self.play_session = None
        self.press_view = None
        self.press_joint = None
        self.deformable_bodies = ()
        self.deformable_counts = ()
        self.deformable_buffer = None
        self.command = None
        self.depth_scale = None
        self.reset_mask = None
        self.tick = 0
        try:
            root = self.get_root()
            if root is None or root.name != SCENE_ROOT_NAME:
                raise RuntimeError("unexpected MuJoCo+libuipc demo scene root")
            nodes = _nodes_by_name(root)

            solver_config = _batch_config(self.context)
            rigid_availability = MuJoCoWarpProvider.availability()
            if not rigid_availability.available:
                raise RuntimeError(rigid_availability.reason)
            ipc_availability = LibuipcBatchSolver.availability(solver_config)
            if not ipc_availability.available:
                raise RuntimeError(
                    ipc_availability.reason
                    + "; set GOBOT_LIBUIPC_SOLVER_MODULE to the built solver module"
                )

            artifact = CompiledMuJoCoIpcArtifact.from_context(self.context)
            self.provider = MuJoCoIpcProvider(
                artifact,
                config=MuJoCoIpcConfig(
                    num_envs=NUM_ENVS,
                    device="cuda:0",
                    environments_per_shard=ENVIRONMENTS_PER_SHARD,
                    capture_mujoco_graphs=True,
                ),
                libuipc_config=solver_config,
                mujoco_options={
                    "nconmax": 32,
                    "njmax": 64,
                    "overflow_check_interval": 0,
                },
            )
            self.press_view = self.provider.create_robot_view(
                robot_name="press",
                base_link="press_head",
                joint_names=("press_slide",),
                link_names=("press_head",),
                scene_context=self.context,
                scene_links=(nodes["press_head"],),
            )
            self.press_joint = nodes["press_slide"]

            bodies = []
            counts = []
            for entry in self.provider.ipc_solver.deformable_bodies:
                name = str(entry["path"]).rsplit("/", 1)[-1]
                body = nodes.get(name)
                if body is None or body.type_name != "DeformableBody3D":
                    raise RuntimeError(
                        f"MuJoCo+libuipc demo is missing deformable body {name!r}"
                    )
                bodies.append(body)
                counts.append(int(entry["element_count"]))
            self.deformable_bodies = tuple(bodies)
            self.deformable_counts = tuple(counts)

            import numpy as np

            self.deformable_buffer = np.zeros(
                (len(counts), max(counts), 3), dtype=np.float32
            )
            control = self.provider.arrays["ctrl"]
            self.depth_scale = torch.linspace(
                0.75,
                1.0,
                NUM_ENVS,
                dtype=control.dtype,
                device=control.device,
            ).unsqueeze(1)
            self.command = torch.zeros_like(self.depth_scale)
            self.reset_mask = torch.ones(
                NUM_ENVS, dtype=torch.bool, device=control.device
            )

            self._reset_provider()
            self._sync_scene()
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.provider,
                fixed_dt=FIXED_DT,
                max_sub_steps=1,
                before_step=self._before_step,
                reset=self._reset_provider,
                sync_scene=self._sync_scene,
            ).start()
            self.play_session.set_status(
                f"GPU soft press batch; displaying environment {DISPLAY_ENV}"
            )
            print(
                "MuJoCo Warp + libuipc editor demo started: "
                f"environments={NUM_ENVS} display_env={DISPLAY_ENV} device=cuda:0"
            )
        except Exception:
            self._close_play_session()
            raise

    def _before_step(self, fixed_dt: float) -> None:
        del fixed_dt
        progress = _press_progress(self.tick)
        self.command.copy_(self.depth_scale).mul_(-PRESS_DEPTH * progress)
        self.press_view.set_position_targets(self.command)
        self.tick += 1

    def _physics_process(self, delta: float) -> None:
        del delta
        if self.provider is None or self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self.play_session.reset()
            return
        if self.tick >= CYCLE_TICKS:
            self.play_session.reset()
            return
        if self.tick and self.tick % 32 == 0:
            state = self.press_view.read_state()
            position = float(
                state.joint_position[DISPLAY_ENV, 0].detach().cpu().item()
            )
            self.play_session.set_status(
                f"GPU soft press batch | env {DISPLAY_ENV} | "
                f"joint {position:.3f} m"
            )

    def _process(self, delta: float) -> None:
        del delta

    def _reset_provider(self) -> None:
        if self.provider is None:
            return
        self.provider.reset(self.reset_mask)
        self.command.zero_()
        self.press_view.set_position_targets(self.command)
        self.tick = 0

    def _sync_scene(self) -> None:
        if self.provider is None:
            return
        self.provider.synchronize()
        positions = (
            self.provider.arrays["ipc_positions"][DISPLAY_ENV]
            .detach()
            .cpu()
            .numpy()
        )
        for body_index, (entry, count) in enumerate(
            zip(
                self.provider.ipc_solver.deformable_bodies,
                self.deformable_counts,
                strict=True,
            )
        ):
            offset = int(entry["element_offset"])
            self.deformable_buffer[body_index, :count] = positions[
                offset : offset + count
            ]
        self.context.apply_deformable_vertices(
            self.deformable_bodies,
            self.deformable_buffer,
            self.deformable_counts,
        )
        state = self.press_view.read_state()
        self.press_joint.joint_position = float(
            state.joint_position[DISPLAY_ENV, 0].detach().cpu().item()
        )
        self.press_view.sync_scene(env_index=DISPLAY_ENV)

    def _exit_tree(self) -> None:
        self._close_play_session()

    def _close_play_session(self) -> None:
        play_session = self.play_session
        self.play_session = None
        if play_session is not None:
            play_session.close()
        provider = self.provider
        self.provider = None
        if provider is not None and play_session is None:
            provider.close()
