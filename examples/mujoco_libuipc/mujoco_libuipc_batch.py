"""Headless MuJoCo Warp + libuipc GPU batch soft-press example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

# Torch must load its CUDA runtime before the native libuipc module. The two
# packages may have been built against different CUDA 12 minor toolkits.
import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
    MuJoCoWarpProvider,
)

from build_scene import HERE, SCENE_NAME, build_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a batched soft-body press with MuJoCo Warp and libuipc."
    )
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--environments-per-shard", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixed-dt", type=float, default=0.002)
    parser.add_argument("--press-depth", type=float, default=0.17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--module-path", default="")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(tempfile.gettempdir()) / "gobot-mujoco-libuipc-example",
    )
    parser.add_argument("--no-mujoco-graph", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.environments_per_shard <= 0:
        raise ValueError("--environments-per-shard must be positive")
    if args.num_envs % args.environments_per_shard:
        raise ValueError(
            "--num-envs must be divisible by --environments-per-shard"
        )
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.fixed_dt <= 0.0:
        raise ValueError("--fixed-dt must be positive")
    if not 0.0 < args.press_depth <= 0.17:
        raise ValueError("--press-depth must be in (0, 0.17]")


def _load_artifact(scene_path: Path) -> tuple[Any, CompiledMuJoCoIpcArtifact]:
    scene_path = scene_path.expanduser().resolve()
    context = gobot.app.create_context()
    context.set_project_path(str(scene_path.parent))
    context.load_scene("res://" + scene_path.name)
    return context, CompiledMuJoCoIpcArtifact.from_context(context)


def _range(values: Any) -> list[float]:
    return [float(values.min().item()), float(values.max().item())]


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    scene_path = args.scene.expanduser().resolve()
    if args.rebuild_scene:
        scene_path = build_scene(scene_path.parent)
    elif not scene_path.is_file():
        raise FileNotFoundError(
            f"scene does not exist: {scene_path}; run build_scene.py first"
        )

    solver_config = LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=args.fixed_dt,
            module_path=args.module_path,
            workspace=str(args.workspace.expanduser().resolve()),
        ),
        environments_per_shard=args.environments_per_shard,
    )
    rigid_availability = MuJoCoWarpProvider.availability()
    if not rigid_availability.available:
        raise RuntimeError(rigid_availability.reason)
    ipc_availability = LibuipcBatchSolver.availability(solver_config)
    if not ipc_availability.available:
        raise RuntimeError(ipc_availability.reason)

    context, artifact = _load_artifact(scene_path)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=args.num_envs,
            device=args.device,
            environments_per_shard=args.environments_per_shard,
            capture_mujoco_graphs=not args.no_mujoco_graph,
        ),
        libuipc_config=solver_config,
        mujoco_options={
            "nconmax": 32,
            "njmax": 64,
            "overflow_check_interval": 0,
        },
    )
    try:
        press_view = provider.create_robot_view(
            robot_name="press",
            base_link="press_head",
            joint_names=("press_slide",),
            link_names=("press_head",),
        )
        positions = provider.arrays["ipc_positions"]
        qpos = provider.arrays["qpos"]
        position_pointer = positions.data_ptr()
        qpos_pointer = qpos.data_ptr()
        initial_positions = positions.clone()
        initial_height = (
            initial_positions[..., 2].amax(dim=1)
            - initial_positions[..., 2].amin(dim=1)
        )

        depth_scale = torch.linspace(
            0.75,
            1.0,
            args.num_envs,
            dtype=provider.arrays["ctrl"].dtype,
            device=args.device,
        ).unsqueeze(1)
        command = torch.empty_like(depth_scale)
        all_envs = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)

        for warmup_step in range(args.warmup_steps):
            progress = float(warmup_step + 1) / float(max(args.warmup_steps, 1))
            command.copy_(depth_scale).mul_(-0.5 * args.press_depth * progress)
            press_view.set_position_targets(command)
            provider.step()
        if args.warmup_steps:
            provider.synchronize()

        elapsed_samples = []
        throughput_samples = []
        ipc_shard_latency_samples = []
        for _ in range(args.repeats):
            provider.reset(all_envs)
            provider.synchronize()
            started = time.perf_counter()
            for step in range(args.steps):
                progress = float(step + 1) / float(args.steps)
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                command.copy_(depth_scale).mul_(
                    -args.press_depth * smooth_progress
                )
                press_view.set_position_targets(command)
                provider.step()
            provider.synchronize()
            elapsed = time.perf_counter() - started
            elapsed_samples.append(elapsed)
            throughput_samples.append(args.num_envs * args.steps / elapsed)
            ipc_shard_latency_samples.append(
                float(
                    provider.ipc_solver.diagnostics.get(
                        "last_step_latency_ms", 0.0
                    )
                )
                / provider.ipc_solver.shard_count
            )

        elapsed = statistics.median(elapsed_samples)
        median_throughput = statistics.median(throughput_samples)

        state = press_view.read_state()
        current_height = (
            positions[..., 2].amax(dim=1) - positions[..., 2].amin(dim=1)
        )
        compression = initial_height - current_height
        press_mapping = next(
            mapping
            for mapping in artifact.coupled_bodies
            if mapping.robot_name == "press" and mapping.link_name == "press_head"
        )
        press_body_id = provider.rigid_solver.resolve_object_ids(
            "body", (press_mapping.mujoco_body_name,)
        )[0]
        reaction_force = torch.linalg.vector_norm(
            provider.arrays["xfrc_applied"][:, press_body_id, :3], dim=1
        )
        provider.rigid_solver.assert_no_overflow()
        for name in ("qpos", "xfrc_applied", "ipc_positions"):
            if not bool(torch.isfinite(provider.arrays[name]).all().item()):
                raise RuntimeError(f"non-finite values in {name}")

        provider.reset(all_envs)
        provider.synchronize()
        reset_error = float(
            (provider.arrays["ipc_positions"] - initial_positions)
            .abs()
            .max()
            .item()
        )

        return {
            "artifact": artifact.digest,
            "collision_ownership": dict(artifact.collision_ownership),
            "composite_graph_captured": provider.graph_captured,
            "composite_graph_capture_reason": provider.diagnostics[
                "graph_capture_reason"
            ],
            "device": args.device,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": median_throughput,
            "environment_steps_per_second_samples": throughput_samples,
            "environments": args.num_envs,
            "environments_per_shard": args.environments_per_shard,
            "ipc_position_storage_stable": positions.data_ptr() == position_pointer,
            "ipc_shard_latency_median_ms": statistics.median(
                ipc_shard_latency_samples
            ),
            "feedback_source": provider.diagnostics["feedback_source"],
            "joint_position_range": _range(state.joint_position[:, 0]),
            "mujoco_graph_capture_enabled": (
                provider.rigid_solver.capabilities.graph_capture
            ),
            "press_target_range": _range(command[:, 0]),
            "qpos_storage_stable": qpos.data_ptr() == qpos_pointer,
            "reaction_force_range_newtons": _range(reaction_force),
            "reset_max_position_error": reset_error,
            "reset_scope": provider.diagnostics["reset_scope"],
            "scene": str(scene_path),
            "shards": provider.ipc_solver.shard_count,
            "soft_compression_range_meters": _range(compression),
            "affine_target_staging": provider.diagnostics[
                "affine_target_staging"
            ],
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "repeats": args.repeats,
        }
    finally:
        provider.close()
        context.clear_scene()


def main() -> None:
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
