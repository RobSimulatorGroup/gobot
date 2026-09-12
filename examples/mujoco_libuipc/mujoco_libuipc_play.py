"""Editor Play entry point for the MuJoCo Warp + libuipc batch demo."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
from typing import Any

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcConfig
from gobot.sim.providers import CompiledMuJoCoIpcArtifact


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


def _solver_module_path(project_path: str) -> str:
    del project_path
    configured = os.environ.get("GOBOT_LIBUIPC_SOLVER_MODULE", "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())

    # The installed package resolves its matching solver module. An unrelated
    # old build directory may have a different ABI or already-loaded SDK SONAME.
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
        self.runtime = None
        self.play_session = None
        try:
            root = self.get_root()
            if root is None or root.name != SCENE_ROOT_NAME:
                raise RuntimeError("unexpected MuJoCo+libuipc demo scene root")

            # Compile the authored template before creating display copies.
            # Only this immutable artifact crosses into the runtime worker.
            artifact = CompiledMuJoCoIpcArtifact.from_context(self.context)
            display_roots, display_nodes = _create_display_scenes(
                root, self.context.project_path
            )
            self.display_roots = display_roots
            self.display_nodes = display_nodes
            deformable_entries = artifact.ipc.deformable_bodies
            bodies = []
            for nodes in display_nodes:
                row = []
                for entry in deformable_entries:
                    name = str(entry["path"]).rsplit("/", 1)[-1]
                    body = nodes.get(name)
                    if body is None or body.type_name != "DeformableBody3D":
                        raise RuntimeError(
                            f"MuJoCo+libuipc demo is missing deformable body {name!r}"
                        )
                    row.append(body)
                bodies.append(row)
            self.scene_sync = gobot.sim.SceneSnapshotSync(self.context,
                    links={"robot.press.link_pose": [(nodes["press_head"],) for nodes in display_nodes]},
                    deformables=bodies,
                    vertex_counts=[int(entry["vertex_count"]) for entry in deformable_entries],
                    display_offsets=GRID_OFFSETS)
            recipe = gobot.sim.SimulationRuntimeSpec("mujoco_libuipc_runtime:create", {
                "artifact": artifact.to_mapping(),
                "solver_config": asdict(_batch_config(self.context)),
                "num_envs": NUM_ENVS,
                "environments_per_shard": ENVIRONMENTS_PER_SHARD,
                "trajectory": {"depth_scales": DEPTH_SCALES, "press_depth": PRESS_DEPTH,
                    "settle_ticks": SETTLE_TICKS, "press_ticks": PRESS_TICKS,
                    "hold_ticks": HOLD_TICKS, "release_ticks": RELEASE_TICKS},
            })
            self.runtime = gobot.sim.AsyncSimulationSession(recipe,
                    fields=("robot.press.link_pose", "deformable.local_vertices"),
                    environments=range(NUM_ENVS))
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.runtime,
                fixed_dt=FIXED_DT,
                max_sub_steps=1,
                sync_scene=self.scene_sync,
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

    def _physics_process(self, delta: float) -> None:
        del delta
        if self.runtime is None or self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self.play_session.reset()
            return
        if self.context.frame_count >= CYCLE_TICKS:
            self.play_session.reset()
            return

    def _process(self, delta: float) -> None:
        del delta

    def _exit_tree(self) -> None:
        self._close_play_session()

    def _close_play_session(self) -> None:
        play_session = self.play_session
        self.play_session = None
        if play_session is not None:
            play_session.close()
        runtime, self.runtime = self.runtime, None
        if runtime is not None and play_session is None:
            runtime.close()
