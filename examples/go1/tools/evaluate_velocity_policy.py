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
DEFAULT_MIN_PROGRESS_RATIO = 0.5


def _evaluation_cfg(*, command_x: float, command_y: float, command_yaw: float):
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
    cfg.command.ranges.lin_vel_y = (command_y, command_y)
    cfg.command.ranges.ang_vel_z = (command_yaw, command_yaw)
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
    command_progress: np.ndarray,
    target_planar_speed: np.ndarray,
    yaw_command_progress: np.ndarray,
    target_yaw_speed: np.ndarray,
    velocity_samples: np.ndarray,
    survival_steps: np.ndarray,
    reset_reason: np.ndarray,
    max_steps: int,
    min_progress_ratio: float,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"episodes": 0}
    samples = np.maximum(velocity_samples[mask], 1)
    reasons = reset_reason[mask]
    survived = (reasons == 0) | (reasons == 2)
    progress_speed = command_progress[mask] / samples
    target_speed = target_planar_speed[mask]
    has_planar_command = target_speed > 1.0e-6
    progress_ratio = np.divide(
        progress_speed,
        target_speed,
        out=np.ones_like(progress_speed, dtype=np.float32),
        where=has_planar_command,
    )
    made_planar_progress = ~has_planar_command | (progress_ratio >= min_progress_ratio)
    yaw_progress_speed = yaw_command_progress[mask] / samples
    yaw_target_speed = target_yaw_speed[mask]
    has_yaw_command = yaw_target_speed > 1.0e-6
    yaw_progress_ratio = np.divide(
        yaw_progress_speed,
        yaw_target_speed,
        out=np.ones_like(yaw_progress_speed, dtype=np.float32),
        where=has_yaw_command,
    )
    made_yaw_progress = ~has_yaw_command | (yaw_progress_ratio >= min_progress_ratio)
    made_progress = made_planar_progress & made_yaw_progress
    return {
        "episodes": count,
        "survival_rate": float(np.mean(survived)),
        "planar_progress_success_rate": float(np.mean(made_planar_progress)),
        "yaw_progress_success_rate": float(np.mean(made_yaw_progress)),
        "progress_success_rate": float(np.mean(made_progress)),
        "admission_rate": float(np.mean(survived & made_progress)),
        "illegal_contact_rate": float(np.mean(reasons == 1)),
        "terrain_out_rate": float(np.mean(reasons == 3)),
        "mean_survival_steps": float(np.mean(survival_steps[mask])),
        "mean_episode_reward": float(np.mean(reward[mask])),
        "mean_velocity_error": float(np.mean(velocity_error[mask] / samples)),
        "mean_body_velocity_x": float(np.mean(body_velocity_x[mask] / samples)),
        "mean_command_progress": float(np.mean(progress_speed)),
        "mean_progress_ratio": float(np.mean(progress_ratio)),
        "mean_yaw_command_progress": float(np.mean(yaw_progress_speed)),
        "mean_yaw_progress_ratio": float(np.mean(yaw_progress_ratio)),
        "min_progress_ratio": float(min_progress_ratio),
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
    min_progress_ratio: float,
) -> dict[str, Any]:
    policy = _load_policy(checkpoint, env.device)
    observations = _reset_to_assignments(env, levels, types, seed=seed)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    velocity_error_sum = torch.zeros_like(reward_sum)
    body_velocity_x_sum = torch.zeros_like(reward_sum)
    command_progress_sum = torch.zeros_like(reward_sum)
    target_planar_speed = torch.linalg.vector_norm(env.command_b[:, :2], dim=1)
    command_direction = env.command_b[:, :2] / target_planar_speed.clamp(min=1.0e-6).unsqueeze(1)
    yaw_command_progress_sum = torch.zeros_like(reward_sum)
    target_yaw_speed = env.command_b[:, 2].abs()
    yaw_command_direction = env.command_b[:, 2].sign()
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
        linear_body, angular_body = env._root_velocity_body()
        current_error = torch.linalg.vector_norm(
            env.command_b[:, :2] - linear_body[:, :2], dim=1
        )
        reward_sum.add_(torch.where(active, state.reward, 0.0))
        velocity_error_sum.add_(torch.where(active & ~done, current_error, 0.0))
        body_velocity_x_sum.add_(torch.where(active & ~done, linear_body[:, 0], 0.0))
        command_progress_sum.add_(
            torch.where(
                active & ~done,
                (linear_body[:, :2] * command_direction).sum(dim=1),
                0.0,
            )
        )
        yaw_command_progress_sum.add_(
            torch.where(
                active & ~done,
                angular_body[:, 2] * yaw_command_direction,
                0.0,
            )
        )
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
    command_progress_np = command_progress_sum.cpu().numpy()
    target_planar_speed_np = target_planar_speed.cpu().numpy()
    yaw_command_progress_np = yaw_command_progress_sum.cpu().numpy()
    target_yaw_speed_np = target_yaw_speed.cpu().numpy()
    velocity_samples_np = velocity_samples.cpu().numpy()
    survival_steps_np = survival_steps.cpu().numpy()
    reset_reason_np = reset_reason.cpu().numpy()
    all_mask = np.ones(env.num_envs, dtype=bool)
    type_names = _terrain_type_names(env)

    def metrics(mask: np.ndarray) -> dict[str, float | int]:
        return _group_metrics(
            mask,
            reward=reward_np,
            velocity_error=velocity_error_np,
            body_velocity_x=body_velocity_x_np,
            command_progress=command_progress_np,
            target_planar_speed=target_planar_speed_np,
            yaw_command_progress=yaw_command_progress_np,
            target_yaw_speed=target_yaw_speed_np,
            velocity_samples=velocity_samples_np,
            survival_steps=survival_steps_np,
            reset_reason=reset_reason_np,
            max_steps=max_steps,
            min_progress_ratio=min_progress_ratio,
        )

    by_level = {
        str(level): metrics(levels_np == level)
        for level in range(env._spawn_rows)
    }
    by_type = {
        type_names[terrain_type]: metrics(types_np == terrain_type)
        for terrain_type in range(env._spawn_cols)
    }
    by_cell = {
        str(level): {
            type_names[terrain_type]: metrics(
                (levels_np == level) & (types_np == terrain_type)
            )
            for terrain_type in range(env._spawn_cols)
        }
        for level in range(env._spawn_rows)
    }
    return {
        "checkpoint": str(checkpoint),
        "overall": metrics(all_mask),
        "by_level": by_level,
        "by_type": by_type,
        "by_cell": by_cell,
    }


def evaluate(
    checkpoints: Sequence[Path],
    *,
    device: str,
    seed: int,
    command_x: float,
    command_y: float,
    command_yaw: float,
    episodes_per_cell: int,
    max_steps: int,
    min_progress_ratio: float,
) -> dict[str, Any]:
    cfg = _evaluation_cfg(
        command_x=command_x,
        command_y=command_y,
        command_yaw=command_yaw,
    )
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
                min_progress_ratio=min_progress_ratio,
            )
            for checkpoint in checkpoints
        ]
        return {
            "backend": "mujoco-warp",
            "device": device,
            "seed": seed,
            "command": [command_x, command_y, command_yaw],
            "episodes_per_cell": episodes_per_cell,
            "terrain_rows": env._spawn_rows,
            "terrain_cols": env._spawn_cols,
            "min_progress_ratio": min_progress_ratio,
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
            f"admission={metrics['admission_rate']:.3f}, "
            f"survival={metrics['survival_rate']:.3f}, "
            f"progress_ok={metrics['progress_success_rate']:.3f}, "
            f"progress={metrics['mean_command_progress']:.3f}, "
            f"ratio={metrics['mean_progress_ratio']:.3f}, "
            f"yaw_ratio={metrics['mean_yaw_progress_ratio']:.3f}, "
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
    parser.add_argument("--command-x", type=float, default=1.0)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--episodes-per-cell", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--min-progress-ratio",
        type=float,
        default=DEFAULT_MIN_PROGRESS_RATIO,
        help="Require this fraction of planar command speed for policy admission.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes_per_cell <= 0:
        raise ValueError("--episodes-per-cell must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.min_progress_ratio < 0.0:
        raise ValueError("--min-progress-ratio must be non-negative")
    report = evaluate(
        args.checkpoint,
        device=args.device,
        seed=args.seed,
        command_x=args.command_x,
        command_y=args.command_y,
        command_yaw=args.command_yaw,
        episodes_per_cell=args.episodes_per_cell,
        max_steps=args.max_steps,
        min_progress_ratio=args.min_progress_ratio,
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
