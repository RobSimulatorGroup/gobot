"""Run batched dual-FR3 rope twisting with MuJoCo Warp and libuipc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import ExitStack
import tempfile
import time
from typing import Any

# Torch initializes CUDA before the native libuipc module is loaded.
import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.sim.providers import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcConvergencePolicy,
    MuJoCoIpcProvider,
    MuJoCoWarpProvider,
)

from build_scene import (
    FIXTURE_BODY_NAMES,
    HERE,
    ROBOT_NAMES,
    SCENE_NAME,
    build_scene,
)
from controllers import (
    CYCLE_TICKS, FINITE_TORQUE_DRIVE_MODE, GRIP_IMPEDANCE_RATIO,
    SHOWCASE_DRIVE_MODE, TWIST_START_TICK, TwistTrialLayout,
    make_trial_layout, wrist_drive_torque_limit,
)
from rope_twist_config import _grip_sensor_specs
from rope_twist_control import RopeControl
from rope_twist_metrics import RopeMetrics



DEFAULT_COUPLING_ITERATIONS = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run dual-FR3 rope twisting as a finite-torque experiment or "
            "constant-speed showcase."
        )
    )
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--environments-per-shard", type=int, default=1)
    parser.add_argument("--steps", type=int, default=CYCLE_TICKS)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--maximum-step-latency-seconds",
        type=float,
        default=0.0,
        help="stop after a physics step exceeds this latency; 0 disables",
    )
    parser.add_argument(
        "--continue-after-cycle-complete",
        action="store_true",
        help="continue stepping in the controller safety hold until --steps",
    )
    parser.add_argument("--fixed-dt", type=float, default=0.002)
    parser.add_argument("--rigid-substeps", type=int, default=1)
    parser.add_argument("--ipc-substeps", type=int, default=1)
    parser.add_argument(
        "--coupling-iterations",
        type=int,
        default=DEFAULT_COUPLING_ITERATIONS,
    )
    parser.add_argument(
        "--relaxation-mode", choices=("fixed", "aitken"), default=None
    )
    parser.add_argument("--relaxation-factor", type=float, default=1.0)
    parser.add_argument("--relaxation-min", type=float, default=0.1)
    parser.add_argument("--relaxation-max", type=float, default=1.0)
    parser.add_argument("--newton-max-iterations", type=int, default=16)
    parser.add_argument("--line-search-max-iterations", type=int, default=8)
    parser.add_argument(
        "--linear-system-tolerance-rate", type=float, default=1.0e-3
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--drive-mode",
        choices=(FINITE_TORQUE_DRIVE_MODE, SHOWCASE_DRIVE_MODE),
        default=FINITE_TORQUE_DRIVE_MODE,
        help="finite physical stall experiment or strong constant-speed showcase",
    )
    parser.add_argument("--module-path", default="")
    parser.add_argument("--friction", type=float, default=1.25)
    parser.add_argument("--contact-resistance", type=float, default=1.0e7)
    parser.add_argument("--coupling-feedback-scale", type=float, default=1.0)
    parser.add_argument(
        "--grip-friction-scale",
        type=float,
        default=1.0,
        help="scale MuJoCo pad/fixture friction for grasp ablations",
    )
    parser.add_argument("--no-mujoco-graph", action="store_true")
    parser.add_argument("--no-coupler-graph", action="store_true")
    parser.add_argument(
        "--defer-deformable-contact-forces",
        action="store_true",
        help="leave per-vertex IPC contact forces stale until explicitly refreshed",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(tempfile.gettempdir()) / "gobot-dual-arm-rope-twist",
    )
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
    if args.maximum_step_latency_seconds < 0.0:
        raise ValueError(
            "--maximum-step-latency-seconds must be non-negative"
        )
    if args.fixed_dt <= 0.0:
        raise ValueError("--fixed-dt must be positive")
    if args.rigid_substeps <= 0 or args.ipc_substeps <= 0:
        raise ValueError("solver substep counts must be positive")
    if args.coupling_iterations <= 0:
        raise ValueError("--coupling-iterations must be positive")
    if min(
        args.relaxation_factor,
        args.relaxation_min,
        args.relaxation_max,
    ) <= 0.0:
        raise ValueError("relaxation parameters must be positive")
    if not (
        args.relaxation_min
        <= args.relaxation_factor
        <= args.relaxation_max
    ):
        raise ValueError(
            "--relaxation-factor must be within relaxation min/max"
        )
    if args.newton_max_iterations <= 0:
        raise ValueError("--newton-max-iterations must be positive")
    if args.line_search_max_iterations <= 0:
        raise ValueError("--line-search-max-iterations must be positive")
    if args.linear_system_tolerance_rate <= 0.0:
        raise ValueError("--linear-system-tolerance-rate must be positive")
    if args.rigid_substeps != args.ipc_substeps:
        raise ValueError(
            "SolverCoupledProxy requires equal rigid and IPC substep counts"
        )
    if args.friction < 0.0:
        raise ValueError("--friction must be non-negative")
    if args.contact_resistance <= 0.0:
        raise ValueError("--contact-resistance must be positive")
    if args.coupling_feedback_scale < 0.0:
        raise ValueError("--coupling-feedback-scale must be non-negative")
    if args.grip_friction_scale < 0.0:
        raise ValueError("--grip-friction-scale must be non-negative")


def _load_artifact(
    scene_path: Path,
) -> tuple[Any, CompiledMuJoCoIpcArtifact]:
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(scene_path.parent))
        context.load_scene("res://" + scene_path.name)
        settings = context.get_mujoco_solver_settings()
        settings["integrator"] = gobot.PhysicsIntegratorType.ImplicitFast
        settings["cone"] = gobot.PhysicsFrictionConeType.Elliptic
        settings["impedance_ratio"] = GRIP_IMPEDANCE_RATIO
        context.set_mujoco_solver_settings(settings)
        return context, CompiledMuJoCoIpcArtifact.from_context(context)
    except Exception:
        context.clear_scene()
        raise


def _solver_config(args):
    return LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=(
                args.fixed_dt * args.rigid_substeps / args.ipc_substeps
            ),
            gravity=(0.0, 0.0, -9.81),
            friction_coefficient=args.friction,
            contact_activation_distance=8.0e-4,
            contact_resistance=args.contact_resistance,
            affine_stiffness=1.0e8,
            module_path=args.module_path,
            workspace=str(args.workspace.expanduser().resolve()),
        ),
        environments_per_shard=args.environments_per_shard,
        newton_max_iterations=args.newton_max_iterations,
        line_search_max_iterations=args.line_search_max_iterations,
        linear_system_tolerance_rate=args.linear_system_tolerance_rate,
        strict_convergence=args.coupling_iterations > 1,
        export_deformable_contact_forces=(
            not args.defer_deformable_contact_forces
        ),
    )


def _create_provider(args, artifact, solver_config):
    return MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=args.num_envs,
            device=args.device,
            environments_per_shard=args.environments_per_shard,
            force_scale=args.coupling_feedback_scale,
            torque_scale=args.coupling_feedback_scale,
            rigid_substeps=args.rigid_substeps,
            ipc_substeps=args.ipc_substeps,
            coupling_iterations=args.coupling_iterations,
            relaxation_mode=args.relaxation_mode,
            relaxation_factor=args.relaxation_factor,
            relaxation_min=args.relaxation_min,
            relaxation_max=args.relaxation_max,
            capture_mujoco_graphs=not args.no_mujoco_graph,
            capture_coupler_graphs=not args.no_coupler_graph,
            convergence_policy=MuJoCoIpcConvergencePolicy(
                enabled=args.coupling_iterations > 1
            ),
        ),
        libuipc_config=solver_config,
        mujoco_options={
            "nconmax": 512,
            "njmax": 2048,
            "contact_sensor_maxmatch": 32,
            "contact_sensors": _grip_sensor_specs(),
            "overflow_check_interval": 0,
        },
    )


def _pad_geom_names(robot_name: str) -> tuple[str, str]:
    return tuple(
        f"{robot_name}_{robot_name}_{side}_rubber_pad_collision"
        for side in ("left", "right")
    )


def _fixture_geom_name(fixture_name: str) -> str:
    return f"{fixture_name}_fixture_body_collision"


def _configure_grip_friction(provider, scale):
    grip_geom_ids = provider.rigid_solver.resolve_object_ids(
        "geom",
        tuple(
            name
            for pair in (
                _pad_geom_names(ROBOT_NAMES[0]),
                (_fixture_geom_name(FIXTURE_BODY_NAMES[0]),),
                _pad_geom_names(ROBOT_NAMES[1]),
                (_fixture_geom_name(FIXTURE_BODY_NAMES[1]),),
            )
            for name in pair
        ),
    )
    if scale != 1.0:
        geom_friction = provider.rigid_solver.model_array("geom_friction")
        geom_indices = torch.as_tensor(
            grip_geom_ids,
            dtype=torch.long,
            device=geom_friction.device,
        )
        scaled_friction = geom_friction.index_select(1, geom_indices)
        scaled_friction.mul_(scale)
        geom_friction.index_copy_(1, geom_indices, scaled_friction)
        if scale == 0.0:
            geom_condim = provider.rigid_solver.model_array("geom_condim")
            condim_indices = geom_indices.to(device=geom_condim.device)
            frictionless_condim = torch.ones_like(
                geom_condim.index_select(0, condim_indices)
            )
            geom_condim.index_copy_(
                0, condim_indices, frictionless_condim
            )
        provider.rigid_solver.recompute_constants()


def _reset_trial(provider, control, initial_qpos):
    provider.reset(
        torch.ones(provider.num_envs, dtype=torch.bool, device=initial_qpos.device),
        qpos=initial_qpos, qvel=torch.zeros_like(provider.arrays["qvel"]),
        ctrl=torch.zeros_like(provider.arrays["ctrl"]),
    )
    control.reset()


@dataclass
class TrialTiming:
    executed_steps: int = 0
    step_latency_samples_seconds: list[float] = field(default_factory=list)
    maximum_step_latency_seconds: float = 0.0
    admission_aborted: bool = False
    admission_abort_reason: str = ""
    elapsed: float = 0.0


def _execute_trial(args, provider, control, metrics):
    timing = TrialTiming()
    started = time.perf_counter()
    for _ in range(args.steps):
        control.step()
        step_started = time.perf_counter()
        provider.step()
        provider.sense()
        latency = time.perf_counter() - step_started
        timing.step_latency_samples_seconds.append(latency)
        timing.maximum_step_latency_seconds = max(timing.maximum_step_latency_seconds, latency)
        timing.executed_steps += 1
        metrics.update()
        controller = control.controller
        if (not args.continue_after_cycle_complete and controller.tick >= TWIST_START_TICK
                and controller.tick % 50 == 0 and controller.cycle_complete):
            break
        if args.maximum_step_latency_seconds > 0 and latency > args.maximum_step_latency_seconds:
            timing.admission_aborted = True
            timing.admission_abort_reason = (
                f"physics step latency {latency:.6f}s exceeded {args.maximum_step_latency_seconds:.6f}s"
            )
            break
    provider.synchronize()
    timing.elapsed = time.perf_counter() - started
    return timing


def run(args: argparse.Namespace, *, layout: TwistTrialLayout | None = None) -> dict[str, Any]:
    """Compile, prepare, measure and report one fixed-capacity trial."""
    _validate_args(args)
    scene_path = args.scene.expanduser().resolve()
    if args.rebuild_scene:
        scene_path = build_scene(scene_path.parent)
    elif not scene_path.is_file():
        raise FileNotFoundError(
            f"scene does not exist: {scene_path}; run build_scene.py first"
        )
    if layout is None:
        layout = make_trial_layout(args.num_envs, seed=args.seed)
    elif len(layout.stall_detection) != args.num_envs:
        raise ValueError("the supplied trial layout does not match --num-envs")

    solver_config = _solver_config(args)
    rigid_availability = MuJoCoWarpProvider.availability()
    if not rigid_availability.available:
        raise RuntimeError(rigid_availability.reason)
    ipc_availability = LibuipcBatchSolver.availability(solver_config)
    if not ipc_availability.available:
        raise RuntimeError(ipc_availability.reason)

    with ExitStack() as cleanup:
        context, artifact = _load_artifact(scene_path)
        cleanup.callback(context.clear_scene)
        provider = _create_provider(args, artifact, solver_config)
        cleanup.callback(provider.close)
        control = RopeControl(
            provider, artifact, layout, fixed_dt=args.fixed_dt,
            drive_torque_limit=wrist_drive_torque_limit(args.drive_mode),
            feedback_enabled=(args.drive_mode == FINITE_TORQUE_DRIVE_MODE
                              and args.coupling_feedback_scale > 0.0),
        )
        _configure_grip_friction(provider, args.grip_friction_scale)
        initial_qpos = provider.arrays["qpos"].clone()
        metrics = RopeMetrics(provider, artifact, control, args, solver_config)
        _reset_trial(provider, control, initial_qpos)
        metrics.capture_attachment_reference()
        if args.warmup_steps:
            for _ in range(args.warmup_steps):
                provider.step()
                provider.sense()
            provider.synchronize()
            _reset_trial(provider, control, initial_qpos)
        timing = _execute_trial(args, provider, control, metrics)
        return metrics.report(scene_path, timing)


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
