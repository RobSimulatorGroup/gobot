"""Evaluate Go1 checkpoints over every authored rough-terrain cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import gobot
from gobot.rl.policy import policy_manifest_from_checkpoint

from examples.go1.tools.checkpoint_policy import CheckpointPolicy
from examples.go1.train.go1_scene_runtime import (
    prepare_go1_scene,
    terrain_spawn_origins,
)
from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
from examples.go1.train.go1_warp_velocity_env import Go1WarpVelocityEnv


REPO_ROOT = Path(__file__).resolve().parents[3]
GO1_PROJECT = REPO_ROOT / "examples/go1"


def _evaluation_cfg(*, command_x: float):
    cfg = go1_velocity_cfg(project_path=GO1_PROJECT)
    cfg.observations.actor_noise = False
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = False
    cfg.spawn_jitter = 0.0
    cfg.reset_z_range = (0.03, 0.03)
    cfg.randomize_reset_yaw = False
    cfg.terrain_curriculum = True
    cfg.command.rel_standing_envs = 0.0
    cfg.command.rel_heading_envs = 0.0
    cfg.command.rel_world_envs = 0.0
    cfg.command.rel_forward_envs = 0.0
    cfg.command.ranges.lin_vel_x = (command_x, command_x)
    cfg.command.ranges.lin_vel_y = (0.0, 0.0)
    cfg.command.ranges.ang_vel_z = (0.0, 0.0)
    return cfg


def _terrain_assignments(
    env: Go1WarpVelocityEnv,
    episodes_per_cell: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    levels: list[int] = []
    types: list[int] = []
    for level in range(env._spawn_rows):
        for terrain_type in range(env._spawn_cols):
            levels.extend([level] * episodes_per_cell)
            types.extend([terrain_type] * episodes_per_cell)
    return (
        torch.tensor(levels, dtype=torch.long, device=env.device),
        torch.tensor(types, dtype=torch.long, device=env.device),
    )


def _reset_to_assignments(
    env: Go1WarpVelocityEnv,
    levels: torch.Tensor,
    types: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    env._terrain_levels.copy_(levels)
    env._terrain_types.copy_(types)
    env._env_origins.copy_(env._terrain_origin_for(env._all_env_ids))
    observations, _ = env.reset(seed=seed)
    return observations


def _load_policy(path: Path, device: str) -> CheckpointPolicy:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manifest = policy_manifest_from_checkpoint(checkpoint)
    if manifest is None:
        raise RuntimeError(f"checkpoint {path} has no Gobot policy manifest")
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise RuntimeError(f"checkpoint {path} has no actor_state_dict")
    return CheckpointPolicy(actor_state, manifest).eval().to(device)


def _terrain_type_names(env: Go1WarpVelocityEnv) -> list[str]:
    entries = env._terrain_config.get("sub_terrains", [])
    result: list[str] = []
    for index in range(env._spawn_cols):
        entry = entries[index] if index < len(entries) else None
        if isinstance(entry, dict):
            result.append(str(entry.get("name", entry.get("type", f"type_{index}"))))
        else:
            result.append(f"type_{index}")
    return result


def _group_metrics(
    mask: np.ndarray,
    *,
    reward: np.ndarray,
    velocity_error: np.ndarray,
    body_velocity_x: np.ndarray,
    velocity_samples: np.ndarray,
    survival_steps: np.ndarray,
    reset_reason: np.ndarray,
    max_steps: int,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"episodes": 0}
    samples = np.maximum(velocity_samples[mask], 1)
    reasons = reset_reason[mask]
    survived = (reasons == 0) | (reasons == 2)
    return {
        "episodes": count,
        "survival_rate": float(np.mean(survived)),
        "illegal_contact_rate": float(np.mean(reasons == 1)),
        "terrain_out_rate": float(np.mean(reasons == 3)),
        "mean_survival_steps": float(np.mean(survival_steps[mask])),
        "mean_episode_reward": float(np.mean(reward[mask])),
        "mean_velocity_error": float(np.mean(velocity_error[mask] / samples)),
        "mean_body_velocity_x": float(np.mean(body_velocity_x[mask] / samples)),
        "max_steps": int(max_steps),
    }


@torch.inference_mode()
def evaluate_checkpoint(
    env: Go1WarpVelocityEnv,
    checkpoint: Path,
    *,
    levels: torch.Tensor,
    types: torch.Tensor,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    policy = _load_policy(checkpoint, env.device)
    observations = _reset_to_assignments(env, levels, types, seed=seed)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    velocity_error_sum = torch.zeros_like(reward_sum)
    body_velocity_x_sum = torch.zeros_like(reward_sum)
    velocity_samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    survival_steps = torch.full(
        (env.num_envs,), max_steps, dtype=torch.long, device=env.device
    )
    reset_reason = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for step in range(1, max_steps + 1):
        actions = policy(observations["actor"])
        state = env.step(actions)
        reasons = state.info["reset_reason"]
        done = active & (reasons > 0)
        linear_body, _ = env._root_velocity_body()
        current_error = torch.linalg.vector_norm(
            env.command_b[:, :2] - linear_body[:, :2], dim=1
        )
        reward_sum.add_(torch.where(active, state.reward, 0.0))
        velocity_error_sum.add_(torch.where(active & ~done, current_error, 0.0))
        body_velocity_x_sum.add_(torch.where(active & ~done, linear_body[:, 0], 0.0))
        velocity_samples.add_((active & ~done).long())

        survival_steps.copy_(torch.where(done, step, survival_steps))
        reset_reason.copy_(torch.where(done, reasons, reset_reason))
        active &= ~done
        observations = state.obs

    env.provider.synchronize()
    levels_np = levels.cpu().numpy()
    types_np = types.cpu().numpy()
    reward_np = reward_sum.cpu().numpy()
    velocity_error_np = velocity_error_sum.cpu().numpy()
    body_velocity_x_np = body_velocity_x_sum.cpu().numpy()
    velocity_samples_np = velocity_samples.cpu().numpy()
    survival_steps_np = survival_steps.cpu().numpy()
    reset_reason_np = reset_reason.cpu().numpy()
    all_mask = np.ones(env.num_envs, dtype=bool)
    type_names = _terrain_type_names(env)

    by_level = {
        str(level): _group_metrics(
            levels_np == level,
            reward=reward_np,
            velocity_error=velocity_error_np,
            body_velocity_x=body_velocity_x_np,
            velocity_samples=velocity_samples_np,
            survival_steps=survival_steps_np,
            reset_reason=reset_reason_np,
            max_steps=max_steps,
        )
        for level in range(env._spawn_rows)
    }
    by_type = {
        type_names[terrain_type]: _group_metrics(
            types_np == terrain_type,
            reward=reward_np,
            velocity_error=velocity_error_np,
            body_velocity_x=body_velocity_x_np,
            velocity_samples=velocity_samples_np,
            survival_steps=survival_steps_np,
            reset_reason=reset_reason_np,
            max_steps=max_steps,
        )
        for terrain_type in range(env._spawn_cols)
    }
    return {
        "checkpoint": str(checkpoint),
        "overall": _group_metrics(
            all_mask,
            reward=reward_np,
            velocity_error=velocity_error_np,
            body_velocity_x=body_velocity_x_np,
            velocity_samples=velocity_samples_np,
            survival_steps=survival_steps_np,
            reset_reason=reset_reason_np,
            max_steps=max_steps,
        ),
        "by_level": by_level,
        "by_type": by_type,
    }


def evaluate(
    checkpoints: Sequence[Path],
    *,
    device: str,
    seed: int,
    command_x: float,
    episodes_per_cell: int,
    max_steps: int,
) -> dict[str, Any]:
    cfg = _evaluation_cfg(command_x=command_x)
    authoring_context = gobot.app.create_context()
    try:
        _, _, terrain = prepare_go1_scene(cfg, context=authoring_context)
        num_cells = int(terrain_spawn_origins(terrain).shape[0])
    finally:
        authoring_context.clear_world()
        authoring_context.clear_scene()
    num_envs = num_cells * episodes_per_cell

    env = Go1WarpVelocityEnv(
        cfg,
        num_envs=num_envs,
        device=device,
        seed=seed,
        max_episode_length=max_steps,
        collect_step_extras=False,
        capture_graphs=True,
        context=gobot.app.create_context(),
    )
    try:
        levels, types = _terrain_assignments(env, episodes_per_cell)
        results = [
            evaluate_checkpoint(
                env,
                checkpoint.resolve(),
                levels=levels,
                types=types,
                seed=seed,
                max_steps=max_steps,
            )
            for checkpoint in checkpoints
        ]
        return {
            "backend": "mujoco-warp",
            "device": device,
            "seed": seed,
            "command": [command_x, 0.0, 0.0],
            "episodes_per_cell": episodes_per_cell,
            "terrain_rows": env._spawn_rows,
            "terrain_cols": env._spawn_cols,
            "results": results,
        }
    finally:
        env.close()


def _print_summary(report: dict[str, Any]) -> None:
    print("\nGo1 rough-terrain checkpoint evaluation")
    print("---------------------------------------")
    for result in report["results"]:
        metrics = result["overall"]
        print(
            f"{Path(result['checkpoint']).name}: "
            f"survival={metrics['survival_rate']:.3f}, "
            f"steps={metrics['mean_survival_steps']:.1f}, "
            f"vx={metrics['mean_body_velocity_x']:.3f}, "
            f"velocity_error={metrics['mean_velocity_error']:.3f}, "
            f"reward={metrics['mean_episode_reward']:.3f}, "
            f"illegal={metrics['illegal_contact_rate']:.3f}, "
            f"out={metrics['terrain_out_rate']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--episodes-per-cell", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes_per_cell <= 0:
        raise ValueError("--episodes-per-cell must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    report = evaluate(
        args.checkpoint,
        device=args.device,
        seed=args.seed,
        command_x=args.command_x,
        episodes_per_cell=args.episodes_per_cell,
        max_steps=args.max_steps,
    )
    _print_summary(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
