"""Run the native SuperDex conveyor workcell without the editor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable

import gobot

from conveyor_config import HERE, SCENE_NAME
from conveyor_control import ConveyorControl
from conveyor_metrics import ConveyorMetrics, assert_finite_state, reset_error
from conveyor_profile import CYCLE_TICKS, FIXED_DT, cycle_phase, quality_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two floating LEAP Hands and rigid/thin-shell/volumetric "
            "packages in one native SuperDex physics world."
        )
    )
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument(
        "--quality", choices=("interactive", "accurate"), default="interactive"
    )
    parser.add_argument("--steps", type=int, default=CYCLE_TICKS)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--execution", choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument(
        "--linear-solver", choices=("auto", "cg", "gmres"), default="auto"
    )
    parser.add_argument("--trace-force-flow", action="store_true")
    parser.add_argument("--phase-diagnostics", action="store_true")
    parser.add_argument("--solver-timings", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.num_envs != 1:
        raise ValueError("native SuperDex currently supports exactly one environment")


def _linear_solver(name: str) -> Any:
    return {
        "auto": gobot.SuperDexLinearSolver.Auto,
        "cg": gobot.SuperDexLinearSolver.CG,
        "gmres": gobot.SuperDexLinearSolver.GMRES,
    }[name]


def _execution_mode(name: str) -> Any:
    return {
        "cpu": gobot.SuperDexExecutionMode.Cpu,
        "cuda": gobot.SuperDexExecutionMode.Cuda,
    }[name]


def _step_with_diagnostics(
    context: Any,
    tick: int,
    control_tick: int,
    failure_observer: Callable[[dict[str, Any]], None] | None,
) -> float:
    started = time.perf_counter()
    try:
        context.step_once()
    except Exception as exc:
        if failure_observer is not None:
            # Capture before run() clears the failed world; never read invalid vertices.
            failure = {
                "tick": tick,
                "control_tick": control_tick,
                "phase": cycle_phase(control_tick),
                "step_once_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                failure["solver"] = context.get_solver_diagnostics()
            except Exception as diagnostic_error:
                failure["diagnostic_error"] = str(diagnostic_error)
            failure_observer(failure)
        raise
    return time.perf_counter() - started


def run(
    args: argparse.Namespace,
    *,
    tick_observer: Callable[[dict[str, Any]], bool] | None = None,
    failure_observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _validate_args(args)
    scene_path = args.scene.expanduser().resolve()
    if args.rebuild_scene:
        from build_scene import build_scene

        scene_path = build_scene(scene_path.parent)
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    profile = quality_profile(args.quality)
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(scene_path.parent))
        root = context.load_scene("res://" + scene_path.name)
        context.fixed_time_step = FIXED_DT
        context.max_sub_steps = 1
        settings = context.get_superdex_solver_settings()
        settings.update(
            {
                "execution_mode": _execution_mode(args.execution),
                "linear_solver": _linear_solver(args.linear_solver),
                "newton_iterations": profile.newton_max_iterations,
                "line_search_iterations": profile.line_search_max_iterations,
                "linear_iterations": -1,
                "substeps": 1,
                "record_deformable_contact_forces": True,
                "record_solver_timings": (
                    tick_observer is not None or getattr(args, "solver_timings", False)
                ),
            }
        )
        context.set_superdex_solver_settings(settings)
        context.build_world(gobot.PhysicsBackendType.SuperDex)

        control = ConveyorControl(context, root)
        for _ in range(args.warmup_steps):
            control.warmup(context.get_physics_state_view())
            context.step_once()
        control.reset()

        initial_state = context.get_physics_state_view()
        assert_finite_state(initial_state)
        metrics = ConveyorMetrics(initial_state, trace_force_flow=args.trace_force_flow,
                                  phase_diagnostics=args.phase_diagnostics)
        latency_samples: list[float] = []
        final_state = initial_state
        for tick in range(args.steps):
            tick_started = time.perf_counter()
            control_tick = min(tick, CYCLE_TICKS - 1)
            state = context.get_physics_state_view()
            state_view_seconds = time.perf_counter() - tick_started
            speed = control.apply(control_tick, state)
            started = time.perf_counter()
            command_seconds = started - tick_started - state_view_seconds
            latency_samples.append(_step_with_diagnostics(
                context, tick + 1, control_tick, failure_observer
            ))
            metrics.belt_travel += speed * FIXED_DT
            read_started = time.perf_counter()
            final_state = context.get_physics_state_view()
            state_view_seconds += time.perf_counter() - read_started
            assert_finite_state(final_state)
            metrics.update(final_state, tick=tick + 1, control_tick=control_tick,
                           rigid_drive=control.rigid_forces.drive_force,
                           soft_drive=control.soft_forces.drive_force)

            # Observers consume completed physics ticks, never render frames or authored poses.
            if tick_observer is not None:
                sample = {
                    "tick": tick + 1,
                    "control_tick": control_tick,
                    "phase": cycle_phase(control_tick),
                    "simulation_time_seconds": float(final_state["simulation_time"]),
                    "step_once_seconds": latency_samples[-1],
                    "command_seconds": command_seconds,
                    "state_view_seconds": state_view_seconds,
                    "observed_tick_seconds": time.perf_counter() - tick_started,
                    "solver": context.get_solver_diagnostics(),
                    **metrics.observation(final_state),
                }
                if tick == 0:
                    sample["scene_complexity"] = {
                        "robots": len(final_state["robots"]),
                        "links": sum(len(robot["links"]) for robot in final_state["robots"]),
                        "joints": sum(len(robot["joints"]) for robot in final_state["robots"]),
                        "deformable_vertices": {
                            str(body["name"]): len(body["local_vertices"])
                            for body in final_state["deformables"]
                        },
                    }
                if not tick_observer(sample):
                    break

        completed_steps = len(latency_samples)
        diagnostics = context.get_solver_diagnostics()
        control.reset()
        restored_state = context.get_physics_state_view()
        reset_max_error = reset_error(initial_state, restored_state)
        latency_sorted = sorted(latency_samples)
        p95_index = max(
            0, math.ceil(0.95 * len(latency_sorted)) - 1
        )
        elapsed = sum(latency_samples)
        return {
            "backend": "SuperDex",
            "experimental": True,
            "quality": profile.name,
            "execution_requested": args.execution,
            "execution_device": diagnostics["execution_device"],
            "device_native": bool(diagnostics["device_native"]),
            "graph_capture": bool(diagnostics["graph_capture"]),
            "environment_batch": False,
            "masked_reset": False,
            "environments": 1,
            "steps": completed_steps,
            "warmup_steps": args.warmup_steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": completed_steps / elapsed if elapsed else 0.0,
            "median_step_latency_seconds": (
                median(latency_samples) if latency_samples else 0.0
            ),
            "p95_step_latency_seconds": (
                latency_sorted[p95_index] if latency_sorted else 0.0
            ),
            **metrics.report(final_state, completed_steps),
            "solver": diagnostics,
            "reset_max_error": reset_max_error,
        }
    finally:
        context.clear_world()
        context.clear_scene()


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
