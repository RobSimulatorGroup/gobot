"""Export a pinned reference trace for the Go1 rough-terrain task.

Run this module with the reference project's Python environment. It deliberately
does not import Gobot, so the reference trace remains independent.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

import torch

import mjlab
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from .go1_parity import (
    PARITY_ACTIONS,
    PARITY_COMMAND,
    PARITY_NUM_ENVS,
    PARITY_SEED,
    PARITY_TERRAIN_ROWS,
    PARITY_TERRAIN_SEED,
    PARITY_TERRAIN_TYPES,
    REFERENCE_REVISION,
    REFERENCE_COMPAT_PATCH_ID,
    REFERENCE_TASK_ID,
    TRACE_SCHEMA_VERSION,
    mujoco_model_diagnostics,
    to_json_value,
    write_trace,
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _repository_revision() -> str:
    root = Path(mjlab.__file__).resolve().parents[2]
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_compat_patch_id() -> str:
    root = Path(mjlab.__file__).resolve().parents[2]
    diff = subprocess.run(
        ("git", "diff", "--", "src/mjlab/sim/sim.py"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    result = subprocess.run(
        ("git", "patch-id", "--stable"),
        cwd=root,
        input=diff,
        check=True,
        capture_output=True,
        text=False,
    )
    fields = result.stdout.decode("ascii").split()
    if not fields or fields[0] != REFERENCE_COMPAT_PATCH_ID:
        raise RuntimeError(
            "reference source does not have the expected MuJoCo Warp 3.10 "
            f"compatibility patch {REFERENCE_COMPAT_PATCH_ID}"
        )
    return fields[0]


def _deterministic_cfg():
    cfg = load_env_cfg(REFERENCE_TASK_ID)
    cfg.scene.num_envs = PARITY_NUM_ENVS
    cfg.seed = PARITY_SEED
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator.seed = PARITY_TERRAIN_SEED
    cfg.curriculum = {}
    for name in (
        "push_robot",
        "encoder_bias",
        "base_com",
        "foot_friction_slide",
        "foot_friction_spin",
        "foot_friction_roll",
    ):
        cfg.events.pop(name, None)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.03, 0.03),
        "yaw": (0.0, 0.0),
    }
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_world_envs = 0.0
    command.rel_forward_envs = 0.0
    command.ranges.lin_vel_x = (PARITY_COMMAND[0], PARITY_COMMAND[0])
    command.ranges.lin_vel_y = (PARITY_COMMAND[1], PARITY_COMMAND[1])
    command.ranges.ang_vel_z = (PARITY_COMMAND[2], PARITY_COMMAND[2])
    return cfg


def _assign_terrain_matrix(env: ManagerBasedRlEnv) -> None:
    terrain = env.scene.terrain
    levels = torch.arange(PARITY_TERRAIN_ROWS, device=env.device).repeat_interleave(
        len(PARITY_TERRAIN_TYPES)
    )
    types = torch.arange(len(PARITY_TERRAIN_TYPES), device=env.device).repeat(
        PARITY_TERRAIN_ROWS
    )
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(types)
    env.scene.env_origins.copy_(terrain.terrain_origins[levels, types])


def _terrain_matrix_snapshot(env: ManagerBasedRlEnv, obs, *, reward=None) -> dict:
    feet = env.scene["feet_ground_contact"].data
    foot_scan = env.scene["foot_height_scan"].data
    result = {
        "qpos": env.sim.data.qpos,
        "qvel": env.sim.data.qvel,
        "height_scan": obs["actor"][:, -187:],
        "foot_height": foot_scan.heights,
        "contact_found": feet.found > 0,
        "reward": None if reward is None else reward,
        "reward_terms": None if reward is None else env.reward_manager._step_reward,
        "terminated": None if reward is None else env.reset_terminated,
        "truncated": None if reward is None else env.reset_time_outs,
    }
    return to_json_value(result)


def _snapshot(env: ManagerBasedRlEnv, obs, *, step: int, action, reward=None) -> dict:
    feet = env.scene["feet_ground_contact"].data
    terrain = env.scene["terrain_scan"].data
    foot_scan = env.scene["foot_height_scan"].data
    snapshot = {
        "step": step,
        "action": action,
        "qpos": env.sim.data.qpos[0],
        "qvel": env.sim.data.qvel[0],
        "actor": obs["actor"][0],
        "critic": obs["critic"][0],
        "command": env.command_manager.get_command("twist")[0],
        "terrain": {
            "level": env.scene.terrain.terrain_levels[0],
            "type": env.scene.terrain.terrain_types[0],
            "origin": env.scene.env_origins[0],
        },
        "sensors": {
            "terrain_distances": terrain.distances[0],
            "terrain_hit_z": terrain.hit_pos_w[0, :, 2],
            "foot_height": foot_scan.heights[0],
            "current_air_time": feet.current_air_time[0],
            "contact_found": feet.found[0] > 0,
            "contact_force": feet.force[0],
        },
        "reward": None if reward is None else reward[0],
        "reward_terms": (
            None if reward is None else env.reward_manager._step_reward[0]
        ),
        "terminated": None if reward is None else env.reset_terminated[0],
        "truncated": None if reward is None else env.reset_time_outs[0],
    }
    return to_json_value(snapshot)


def export_trace(*, output: str | Path, device: str, allow_revision_mismatch: bool) -> Path:
    revision = _repository_revision()
    if revision != REFERENCE_REVISION and not allow_revision_mismatch:
        raise RuntimeError(
            f"reference revision is {revision}, expected {REFERENCE_REVISION}; "
            "pass --allow-revision-mismatch only for exploratory comparisons"
        )

    cfg = _deterministic_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        _assign_terrain_matrix(env)
        obs, _ = env.reset(seed=PARITY_SEED)
        terrain_origins = to_json_value(env.scene.env_origins)
        reset_matrix = _terrain_matrix_snapshot(env, obs)
        snapshots = [_snapshot(env, obs, step=0, action=None)]
        first_step_matrix = None
        for step, values in enumerate(PARITY_ACTIONS, start=1):
            action = torch.tensor((values,), dtype=torch.float32, device=device).expand(
                env.num_envs, -1
            )
            obs, reward, _, _, _ = env.step(action)
            if step == 1:
                first_step_matrix = _terrain_matrix_snapshot(env, obs, reward=reward)
            snapshots.append(
                _snapshot(env, obs, step=step, action=list(values), reward=reward)
            )
        robot = env.scene["robot"]
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metadata": {
                "source": "reference",
                "task_id": REFERENCE_TASK_ID,
                "source_revision": revision,
                "source_compat_patch_id": _repository_compat_patch_id(),
                "seed": PARITY_SEED,
                "terrain_seed": PARITY_TERRAIN_SEED,
                "device": device,
                "packages": {
                    "mjlab": _package_version("mjlab"),
                    "mujoco": _package_version("mujoco"),
                    "mujoco-warp": _package_version("mujoco-warp"),
                    "torch": _package_version("torch"),
                    "warp-lang": _package_version("warp-lang"),
                },
                "physics_dt": float(env.physics_dt),
                "decimation": int(env.cfg.decimation),
                "step_dt": float(env.step_dt),
                "joint_names": list(robot.joint_names),
                "reward_names": list(env.reward_manager.active_terms),
                "actor_dim": int(obs["actor"].shape[1]),
                "critic_dim": int(obs["critic"].shape[1]),
            },
            "terrain_matrix": {
                "rows": PARITY_TERRAIN_ROWS,
                "cols": len(PARITY_TERRAIN_TYPES),
                "type_names": list(PARITY_TERRAIN_TYPES),
                "levels": [
                    level
                    for level in range(PARITY_TERRAIN_ROWS)
                    for _ in PARITY_TERRAIN_TYPES
                ],
                "types": list(range(len(PARITY_TERRAIN_TYPES))) * PARITY_TERRAIN_ROWS,
                "origins": terrain_origins,
                "reset": reset_matrix,
                "first_step": first_step_matrix,
            },
            "mujoco_model": mujoco_model_diagnostics(env.sim._mj_model),
            "snapshots": snapshots,
        }
        return write_trace(output, trace)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    args = parser.parse_args()
    path = export_trace(
        output=args.output,
        device=args.device,
        allow_revision_mismatch=args.allow_revision_mismatch,
    )
    print(f"Wrote reference trace: {path}")


if __name__ == "__main__":
    main()
