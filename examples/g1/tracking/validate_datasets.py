"""Bounded SEED/LAFAN integration validation: prepare, train, evaluate each clip.

Each clip trains a separate PPO policy, as in mjlab tracking. This is not a
multi-motion policy benchmark or evidence of converged motion quality.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch
from gobot.rl.tracking.prepare import prepare
from gobot.rl.tracking import TrackingEnv
from gobot.rl.tracking.training import Config, run
from gobot.rl.tracking.evaluate import evaluate
from gobot.rl.training.status import write_json


def validate(
    manifest,
    output,
    *,
    backend="mujoco-warp",
    device=None,
    num_envs=64,
    updates=10,
    seed=42,
    nconmax=128,
    njmax=500,
):
    entries = json.loads(Path(manifest).read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a nonempty clip list")
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("duplicate clip name")
    # Verify every source before creating outputs or launching any training.
    for entry in entries:
        if entry["dataset"] not in ("seed", "lafan"):
            raise ValueError("dataset must be seed or lafan")
        if Path(entry["name"]).name != entry["name"] or entry["name"] in (".", ".."):
            raise ValueError("invalid clip name")
        digest = hashlib.sha256(Path(entry["csv"]).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"source digest mismatch: {entry['name']}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "inputs.json", entries)
    torch.set_num_threads(4)
    results = []
    for entry in entries:
        directory = output / entry["dataset"] / entry["name"]
        directory.mkdir(parents=True)
        project = prepare(
            directory / "assets",
            csv=entry["csv"],
            csv_format=entry["dataset"],
            input_fps=entry["fps"],
        )
        cfg = Config(
            project_path=project,
            backend=backend,
            device=device,
            num_envs=num_envs,
            seed=seed,
            nconmax=nconmax,
            njmax=njmax,
            updates=updates,
            output=directory / "training",
        )
        # Test reference resets at distributed phases before learning, not only frame 0.
        probe = TrackingEnv(
            replace(
                cfg,
                num_envs=4,
                domain_randomization=False,
                reset_noise=False,
                observation_noise=False,
                pushes=False,
            )
        )
        try:
            frames = torch.linspace(
                0, probe.motion.frames - 2, 4, device=probe.device
            ).long()
            probe.reset(frames=frames)
            physical = probe._bodies(probe.batch.read_state())
            reference = probe.motion.bodies(frames)
            physical_poses = probe.batch.read_state().link_pose
            fk_error = float((physical[0] - physical_poses[..., :3]).norm(dim=-1).max())
            from gobot.rl.tracking.math import angle_error

            fk_angle = float(
                angle_error(physical[1], physical_poses[..., (6, 3, 4, 5)]).max()
            )
            if fk_error > 1e-4 or fk_angle > 1e-3:
                raise AssertionError(
                    f"scene FK differs from physics: {fk_error} m, {fk_angle} rad"
                )
            # Limit clipping can intentionally change the reset pose. Record it.
            joint = probe.batch.read_state().joint_position
            reset_error = float((joint - probe.motion.joint_pos[frames]).abs().max())
            body_error = float((physical[0] - reference[0]).norm(dim=-1).max())
            for _ in range(8):
                transition = probe.step(
                    torch.zeros(4, probe.num_actions, device=probe.device)
                )
                if (
                    not torch.isfinite(transition.reward).all()
                    or not torch.isfinite(transition.obs["actor"]).all()
                ):
                    raise FloatingPointError("nonfinite dataset smoke transition")
            probe_result = dict(
                frames=probe.motion.frames,
                reference_seconds=(probe.motion.frames - 1) * probe.control_dt,
                actor_dim=probe.num_obs,
                critic_dim=probe.num_privileged_obs,
                reset_joint_clipping_rad=reset_error,
                fk_to_physics_position_error_m=fk_error,
                fk_to_physics_orientation_error_rad=fk_angle,
                reset_body_error_m=body_error,
                reset_frames=frames.cpu().tolist(),
                steps=8,
            )
        finally:
            probe.close()
        training = run(cfg)
        if training["actor_max_weight_change"] <= 0:
            raise RuntimeError("PPO did not update actor weights")
        evaluation = evaluate(
            training["checkpoint"], steps=500, backend=backend, device=device
        )
        result = dict(
            dataset=entry["dataset"],
            clip=entry["name"],
            source_sha256=entry["sha256"],
            seed=seed,
            probe=probe_result,
            training=training,
            evaluation=evaluation,
        )
        results.append(result)
        write_json(directory / "result.json", result)
        write_json(
            output / "report.json",
            dict(
                scope="bounded integration validation; not converged or full dataset evaluation",
                complete=len(results) == len(entries),
                expected_clips=len(entries),
                settings=dict(
                    backend=backend,
                    device=device,
                    num_envs=num_envs,
                    updates=updates,
                    seed=seed,
                    nconmax=nconmax,
                    njmax=njmax,
                ),
                results=results,
            ),
        )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--backend", choices=("mujoco-cpu", "mujoco-warp"), default="mujoco-warp"
    )
    p.add_argument("--device")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--updates", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nconmax", type=int, default=128)
    p.add_argument("--njmax", type=int, default=500)
    args = p.parse_args()
    validate(**vars(args))


if __name__ == "__main__":
    main()
