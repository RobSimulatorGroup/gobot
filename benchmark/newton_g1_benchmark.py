#!/usr/bin/env python3
"""Measure Newton G1 physics throughput without imposing hardware thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "newton_g1"


def benchmark(environment_counts: tuple[int, ...], ticks: int) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Newton G1 benchmark requires a CUDA-capable Torch runtime")

    sys.path.insert(0, str(EXAMPLE))
    try:
        contract = runpy.run_path(str(EXAMPLE / "scripts" / "g1_policy_contract.py"))
        playback = runpy.run_path(str(EXAMPLE / "scripts" / "g1_policy.py"))
    finally:
        sys.path.remove(str(EXAMPLE))

    import gobot
    from gobot.rl.providers import NewtonModelConfig, NewtonProvider

    context = gobot.app.create_context()
    results = []
    try:
        context.set_project_path(str(EXAMPLE))
        root = context.load_scene("res://newton_g1.jscn")
        task = playback["_load_task_config"](str(EXAMPLE))
        native = contract["load_native_policy_contract"](
            EXAMPLE / task["resources"]["policy_contract"].removeprefix("res://")
        )
        stack = [root]
        robots = []
        while stack:
            node = stack.pop()
            if node.type_name == "Robot3D":
                robots.append(node)
            stack.extend(reversed(node.children))
        if len(robots) != 1:
            raise RuntimeError(f"Newton G1 benchmark expected one Robot3D, got {len(robots)}")
        joints = playback["_nodes_by_name"](
            robots[0], contract["JOINT_NAMES"], type_name="Joint3D"
        )
        playback["_validate_native_g1_scene"](robots[0], joints, native)
        artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        for num_envs in environment_counts:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            provider = NewtonProvider(
                artifact,
                num_envs=num_envs,
                device="cuda:0",
                fixed_time_step=contract["PHYSICS_DT"],
                nconmax=30,
                njmax=100,
                use_mujoco_contacts=True,
                capture_graphs=True,
                model_config=NewtonModelConfig(**task["newton_model"]),
            )
            initialization_seconds = time.perf_counter() - started
            try:
                view = provider.create_robot_view(
                    robot_name=artifact["robot_names"][0],
                    base_link=contract["BASE_LINK"],
                    joint_names=contract["JOINT_NAMES"],
                    link_names=contract["LINK_NAMES"],
                )
                device = provider.arrays["joint_q"].device
                base_pose = torch.tensor(
                    [contract["BASE_POSE_XYZW"]], dtype=torch.float32, device=device
                ).repeat(num_envs, 1)
                joint_position = torch.tensor(
                    [native["mjw_joint_pos"]], dtype=torch.float32, device=device
                ).repeat(num_envs, 1)
                view.reset(
                    torch.ones(num_envs, dtype=torch.bool, device=device),
                    base_pose=base_pose,
                    base_velocity=torch.zeros((num_envs, 6), device=device),
                    joint_position=joint_position,
                    joint_velocity=torch.zeros((num_envs, len(contract["JOINT_NAMES"])), device=device),
                    controls=joint_position,
                )
                view.set_position_targets(joint_position)
                first_started = time.perf_counter()
                provider.step()
                provider.synchronize()
                first_step_seconds = time.perf_counter() - first_started

                measured_started = time.perf_counter()
                provider.step(nsteps=ticks)
                provider.synchronize()
                measured_seconds = time.perf_counter() - measured_started
                provider.assert_no_overflow()
                provider.assert_finite()
                results.append(
                    {
                        "num_envs": num_envs,
                        "initialization_seconds": initialization_seconds,
                        "first_step_seconds": first_step_seconds,
                        "physics_ticks_per_second": ticks / measured_seconds,
                        "environment_steps_per_second": num_envs * ticks / measured_seconds,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                        "graph_captured": provider.graph_captured,
                        "capacities": dict(provider.capacities),
                    }
                )
            finally:
                provider.close()
    finally:
        context.clear_world()
        context.clear_scene()
    return {
        "schema_version": 3,
        "benchmark": "newton_g1_physics",
        "asset_revision": "261cd1f429619d8ef4f546bd788ab9dea906b5e1",
        "ticks": ticks,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", default="1,32,128")
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    environment_counts = tuple(int(value) for value in args.envs.split(","))
    if not environment_counts or min(environment_counts) <= 0 or args.ticks <= 0:
        parser.error("environment counts and ticks must be positive")
    report = benchmark(environment_counts, args.ticks)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
