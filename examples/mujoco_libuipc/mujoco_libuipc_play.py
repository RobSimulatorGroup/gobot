"""Editor Play entry point for the MuJoCo Warp + libuipc batch demo."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
from typing import Any

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
GRID_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.50, 0.0, 0.0),
    (0.0, 0.45, 0.0),
    (0.50, 0.45, 0.0),
)
DEPTH_SCALES = (0.75, 5.0 / 6.0, 11.0 / 12.0, 1.0)
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


def _load_scene_builder(project_path: str) -> Any:
    if not str(project_path).strip():
        raise RuntimeError("MuJoCo+libuipc display setup requires a project path")
    path = Path(project_path).expanduser().resolve() / "build_scene.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"MuJoCo+libuipc scene builder does not exist: {path}"
        )
    spec = importlib.util.spec_from_file_location(
        "gobot_mujoco_libuipc_runtime_builder", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MuJoCo+libuipc scene builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_display_scenes(
    root: Any,
    project_path: str,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    builder = _load_scene_builder(project_path)
    display_roots = [root]
    display_nodes = [_nodes_by_name(root)]
    for environment in range(1, NUM_ENVS):
        display_root = builder.create_scene()
        display_root.name = f"display_env_{environment}"
        root.add_child(display_root)
        display_roots.append(display_root)
        display_nodes.append(_nodes_by_name(display_root))
    for display_root, offset in zip(display_roots, GRID_OFFSETS, strict=True):
        display_root.position = offset
    return tuple(display_roots), tuple(display_nodes)


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
    """Run the GPU batch and render all four environments in Play Mode."""

    def _ready(self) -> None:
        # Load Torch's CUDA runtime before the native libuipc solver module.
        import torch

        self.provider = None
        self.play_session = None
        self.press_view = None
        self.display_roots = ()
        self.display_nodes = ()
        self.display_press_heads = ()
        self.display_deformable_bodies = ()
        self.display_deformable_counts = ()
        self.deformable_buffer = None
        self.link_pose_buffer = None
        self.command = None
        self.depth_scale = None
        self.reset_mask = None
        self.tick = 0
        try:
            root = self.get_root()
            if root is None or root.name != SCENE_ROOT_NAME:
                raise RuntimeError("unexpected MuJoCo+libuipc demo scene root")

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
            )

            display_roots, display_nodes = _create_display_scenes(
                root, self.context.project_path
            )
            self.display_roots = display_roots
            self.display_nodes = display_nodes
            self.display_press_heads = tuple(
                nodes["press_head"] for nodes in display_nodes
            )

            deformable_entries = tuple(self.provider.ipc_solver.deformable_bodies)
            bodies = []
            counts = []
            for nodes in display_nodes:
                for entry in deformable_entries:
                    name = str(entry["path"]).rsplit("/", 1)[-1]
                    body = nodes.get(name)
                    if body is None or body.type_name != "DeformableBody3D":
                        raise RuntimeError(
                            f"MuJoCo+libuipc demo is missing deformable body {name!r}"
                        )
                    bodies.append(body)
                    counts.append(int(entry["element_count"]))
            self.display_deformable_bodies = tuple(bodies)
            self.display_deformable_counts = tuple(counts)

            import numpy as np

            self.deformable_buffer = np.zeros(
                (len(bodies), max(counts), 3), dtype=np.float32
            )
            self.link_pose_buffer = np.zeros((NUM_ENVS, 7), dtype=np.float32)
            control = self.provider.arrays["ctrl"]
            self.depth_scale = torch.tensor(
                DEPTH_SCALES,
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
                "GPU soft press batch | 4 environments | 2x2 display"
            )
            print(
                "MuJoCo Warp + libuipc editor demo started: "
                f"environments={NUM_ENVS} display_grid=2x2 device=cuda:0"
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
            minimum = float(self.link_pose_buffer[:, 2].min())
            maximum = float(self.link_pose_buffer[:, 2].max())
            self.play_session.set_status(
                "GPU soft press batch | 4 environments | "
                f"press height {minimum:.3f}..{maximum:.3f} m"
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
        positions = self.provider.arrays["ipc_positions"].detach().cpu().numpy()
        state = self.press_view.read_state()
        self.link_pose_buffer[:] = state.link_pose[:, 0].detach().cpu().numpy()

        deformable_entries = tuple(self.provider.ipc_solver.deformable_bodies)
        body_index = 0
        for environment, grid_offset in enumerate(GRID_OFFSETS):
            for entry in deformable_entries:
                count = int(entry["element_count"])
                offset = int(entry["element_offset"])
                self.deformable_buffer[body_index, :count] = positions[
                    environment, offset : offset + count
                ]
                self.deformable_buffer[body_index, :count] += grid_offset
                body_index += 1
        self.context.apply_deformable_vertices(
            self.display_deformable_bodies,
            self.deformable_buffer,
            self.display_deformable_counts,
        )
        for environment in range(NUM_ENVS):
            self.link_pose_buffer[environment, :3] += GRID_OFFSETS[environment]
        self.context.apply_link_poses(
            self.display_press_heads, self.link_pose_buffer
        )

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
