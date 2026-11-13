"""Evaluate frozen SuperMimic policies with one FP32 full-motion protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from .config import Config
from .env import TrackingEnv
from .models import MultiEncoderAE
from .motion import digest, FIELDS, inspect_library
from .state import GenCoordRepresentation
from .tasks import EncoderSpec
from .logging import write_json
from .physics_rewards import (
    TrackingRewardCfg,
    reanchor_reference,
    tracking_reward_terms,
)
from .training import FORMAT

PROTOCOL = "gobot_supermimic_fp32_evaluation_v1"
ANCHORS = ("pelvis", "torso_link")
CONTROL_FIELDS = (
    "robot_name",
    "base_link",
    "joint_names",
    "tracked_links",
    "default_joint_position",
    "action_scale",
    "soft_joint_limit_factor",
    "physics_dt",
    "decimation",
    "solver_settings",
    "nominal_evaluation_foot_friction",
    "foot_shapes",
)


def configure_precision():
    # An explicit inference setting; no inference compile/capture for the actor.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hash_value(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def control_profile(path):
    profile = json.loads(Path(path).read_text())
    return {key: profile[key] for key in CONTROL_FIELDS}


def shared_sources():
    from . import env, motion, models, state, normalization, tasks
    from . import rewards, physics_rewards, rotations, rotation_primitives, geometry
    from gobot.rl import articulation, kinematics, runtime
    from gobot.sim.providers import base, mujoco_warp

    modules = (
        env,
        motion,
        models,
        state,
        normalization,
        tasks,
        rewards,
        physics_rewards,
        rotations,
        rotation_primitives,
        geometry,
        articulation,
        kinematics,
        runtime,
        base,
        mujoco_warp,
    )
    result = {module.__name__: digest(module.__file__) for module in modules}
    result["gobot.rl.supermimic.evaluate"] = digest(__file__)
    return result


class EvaluationEnv(TrackingEnv):
    """Training physics with deterministic resets and two reward anchors."""

    def refresh_reference(self):
        self.reference = self.motion.body_state(self.frame)
        current = self.full_state()
        bodies = self.body_state(current)
        self.relative_by_anchor = {}
        for name in ANCHORS:
            index = self.body_names.index(name)
            self.relative_by_anchor[name] = reanchor_reference(
                self.reference[0],
                self.reference[1],
                self.reference[0][:, index],
                self.reference[1][:, index],
                bodies[0][:, index],
                bodies[1][:, index],
            )
        self.relative_pose = self.relative_by_anchor["pelvis"]

    def _advance(self):
        super()._advance()
        # The torso reference is anchored at the same pre-control state as pelvis.
        index = self.body_names.index("torso_link")
        bodies = self.body_state(self.full_state())
        torso = reanchor_reference(
            self.reference[0],
            self.reference[1],
            self.reference[0][:, index],
            self.reference[1][:, index],
            bodies[0][:, index],
            bodies[1][:, index],
        )
        self.relative_by_anchor = {"pelvis": self.relative_pose, "torso_link": torso}

    @torch.no_grad()
    def step(self, actions):
        previous = self.action.clone()
        reference = self.reference
        relative = self.relative_by_anchor["torso_link"]
        state, reward, term, trunc, info = super().step(actions)
        # super.step may reset completed lanes. Use the retained pre-reset lagged
        # state, target, last action and terminal joints for both scores.
        body = self.body_state(self.lag_state)
        torso = tracking_reward_terms(
            robot_body_pos_w=body[0],
            robot_body_quat_w=body[1],
            robot_body_lin_vel_w=body[2],
            robot_body_ang_vel_w=body[3],
            ref_body_pos_w=reference[0],
            ref_body_quat_w=reference[1],
            ref_body_lin_vel_w=reference[2],
            ref_body_ang_vel_w=reference[3],
            anchor_index=self.body_names.index("torso_link"),
            action=actions,
            prev_action=previous,
            joint_pos=info["terminal_state"][:, 7:36],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
            relative_body_pose=relative,
        )
        info["anchor_rewards"] = {"pelvis": reward, "torso_link": torso["total"]}
        if not torch.isfinite(torso["total"]).all():
            raise FloatingPointError("nonfinite torso reward")
        return state, reward, term, trunc, info


class FrozenPolicy:
    """Restore the trained actor's native observation layout without noise."""

    def __init__(self, checkpoint, env):
        if checkpoint.get("format") != FORMAT:
            raise ValueError("unsupported SuperMimic checkpoint format")
        cfg = checkpoint["config"]
        self.actor_obs = cfg["actor_obs"]
        self.history_length = cfg["history_length"]
        if self.actor_obs not in ("rep_state", "proprio") or self.history_length < 1:
            raise ValueError("invalid actor observation configuration")
        if self.actor_obs == "rep_state" and self.history_length != 1:
            raise ValueError("rep_state actor requires history_length=1")
        self.representation = GenCoordRepresentation(
            29,
            env.default_q,
            dt=env.step_dt,
            decimation=env.batch.decimation,
            actor_obs_mode=self.actor_obs,
        ).to(env.device)
        state_dim = 68 if self.actor_obs == "rep_state" else 93
        self.actor = (
            MultiEncoderAE(
                state_dim,
                29,
                [EncoderSpec("motion", 640), EncoderSpec("velocity", 3)],
                {"motion": 1.0, "velocity": 0.0},
                cfg["latent_dim"],
                tuple(cfg["actor_hidden_dims"]),
                cfg["latent_bc_weight"],
                self.history_length,
            )
            .to(env.device)
            .eval()
            .requires_grad_(False)
        )
        self.actor.load_state_dict(checkpoint["actor"], strict=True)
        self.history = None
        self.description = {
            "kind": "gencoord",
            "actor_obs": self.actor_obs,
            "history": self.history_length,
            "state_dim": state_dim,
            "goal_frames": 10,
            "goal_stride": 5,
            "goal_dim": 640,
        }

    def observation(self, env, state, reset_ids):
        sensor = (
            self.representation.represent(state)
            if self.actor_obs == "rep_state"
            else env.sensor(state)
        )
        if self.history is None:
            self.history = sensor[:, None].repeat(1, self.history_length, 1)
        else:
            self.history = torch.cat((self.history[:, 1:], sensor[:, None]), dim=1)
        self.history[reset_ids] = sensor[reset_ids, None]
        _, goal = self.representation.compute_wm_goal(
            env.motion, env.frame, state[:, :7]
        )
        return TensorDict(
            {"history": self.history, "goal_motion": goal},
            batch_size=[len(state)],
            device=state.device,
        )

    def __call__(self, env, state, reset_ids):
        return self.actor(self.observation(env, state, reset_ids))


def validate_row(row):
    frames, steps, expected = (
        row["frames"],
        row["executed_steps"],
        row["expected_steps"],
    )
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (frames, steps, expected)
        )
        or frames < 3
        or expected != frames - 1
        or not 1 <= steps <= expected
    ):
        raise ValueError(f"invalid evaluation frame counts: {row['name']}")
    if not all(
        isinstance(row[key], bool)
        for key in ("failed", "finished", "completed", "capped")
    ):
        raise ValueError("evaluation masks must be boolean")
    if not math.isfinite(row["return"]) or not math.isfinite(row["coverage"]):
        raise ValueError("nonfinite evaluation result")
    if not math.isclose(row["coverage"], steps / expected, abs_tol=1e-12):
        raise ValueError("coverage differs from executed clip length")
    if row["finished"] != (steps == expected) or row["completed"] != (
        row["finished"] and not row["failed"]
    ):
        raise ValueError("inconsistent finished/completed flags")
    if row["capped"] and (row["failed"] or row["finished"]):
        raise ValueError("a capped clip cannot also be resolved")
    if not (row["failed"] or row["finished"] or row["capped"]):
        raise ValueError("unresolved clip reported as evaluated")


def summarize(rows):
    if not rows:
        raise ValueError("empty common evaluation")
    for row in rows:
        validate_row(row)
        if set(row["returns"]) != set(ANCHORS) or not all(
            math.isfinite(v) for v in row["returns"].values()
        ):
            raise ValueError("invalid dual-anchor rewards")
        if row["return"] != row["returns"]["pelvis"]:
            raise ValueError("primary score must be pelvis")
    count = len(rows)
    return {
        "clips": count,
        "failures": sum(r["failed"] for r in rows),
        "failure_rate": sum(r["failed"] for r in rows) / count,
        "completed": sum(r["completed"] for r in rows),
        "frame_coverage": sum(r["executed_steps"] for r in rows)
        / sum(r["expected_steps"] for r in rows),
        "mean_coverage": sum(r["coverage"] for r in rows) / count,
        "anchors": {
            anchor: {
                "mean_return": sum(r["returns"][anchor] for r in rows) / count,
                "mean_expected_length_return": sum(
                    r["returns"][anchor] * r["coverage"] for r in rows
                )
                / count,
            }
            for anchor in ANCHORS
        },
    }


def merge_reports(paths, names, output):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if not reports:
        raise ValueError("no common evaluation reports")
    first = reports[0]
    count = first["shard_count"]
    if len(reports) != count or {r["shard_index"] for r in reports} != set(
        range(count)
    ):
        raise ValueError("missing/duplicate shards")
    if first["selection_sha256"] != hash_value(names) or first[
        "selection_count"
    ] != len(names):
        raise ValueError("selection differs")
    identity = (
        "protocol",
        "checkpoint_sha256",
        "policy_adapter",
        "common",
        "common_sha256",
        "selection_sha256",
        "selection_count",
        "shard_count",
    )
    rows = []
    for report in reports:
        if (
            report["protocol"] != PROTOCOL
            or not report["complete"]
            or any(report[k] != first[k] for k in identity)
        ):
            raise ValueError("incomplete or mismatched common evaluation")
        if report["common_sha256"] != hash_value(report["common"]):
            raise ValueError("common fingerprint differs")
        selected = names[report["shard_index"] :: count]
        actual = [r["name"] for r in report["clips"]]
        if len(actual) != len(selected) or set(actual) != set(selected):
            raise ValueError("missing/duplicate clip")
        if any(r["capped"] for r in report["clips"]):
            raise ValueError("capped evaluation")
        if report["summary"] != summarize(report["clips"]):
            raise ValueError("summary differs")
        rows.extend(report["clips"])
    order = {n: i for i, n in enumerate(names)}
    if len(order) != len(names):
        raise ValueError("duplicate names")
    rows.sort(key=lambda r: order[r["name"]])
    result = {key: first[key] for key in identity}
    result.update(
        complete=True,
        clips=rows,
        summary=summarize(rows),
        shards=[{"path": str(p), "sha256": digest(p)} for p in paths],
    )
    write_json(output, result)
    return result


@torch.no_grad()
def evaluate(
    checkpoint,
    project,
    library,
    output,
    *,
    device="cuda:0",
    num_envs=512,
    shard_index=0,
    shard_count=1,
    clips=None,
    max_steps=None,
):
    if (
        num_envs < 1
        or shard_count < 1
        or not 0 <= shard_index < shard_count
        or (max_steps is not None and max_steps < 1)
    ):
        raise ValueError("invalid evaluation sizes")
    output = Path(output)
    if output.exists() or output.with_suffix(".partial.jsonl").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress = output.with_suffix(".progress.json")
    partial = output.with_suffix(".partial.jsonl")
    write_json(progress, {"state": "loading", "shard_index": shard_index})
    configure_precision()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    manifest = inspect_library(library, verify_hashes=True)
    if manifest.get("source", {}).get("dtype") != "float32" or not manifest[
        "source"
    ].get("every_npz_hash_verified"):
        raise ValueError(
            "common benchmark requires audited original dense FP32 extraction"
        )
    for key in FIELDS:
        if np.load(Path(library) / f"{key}.npy", mmap_mode="r").dtype != np.float32:
            raise ValueError("common reference is not FP32")
    profile_path = Path(project) / "assets/supermimic/robot_profile.json"
    controls = control_profile(profile_path)
    checkpoint_hash = digest(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config(
        project=Path(project),
        library=Path(library),
        backend="mujoco-warp" if device.startswith("cuda") else "mujoco-cpu",
        device=device,
        num_envs=num_envs,
        domain_randomization=False,
        reset_noise=False,
        sampling_mode="start",
        episode_length_s=1.0e9,
        compile=False,
        capture_updates=False,
        verify_data=False,
        nconmax=128,
        njmax=500,
    )
    env = EvaluationEnv(cfg)
    try:
        adapter = FrozenPolicy(ck, env)
        if (
            ck["identity"]["profile"] != env.profile_digest
            or ck["identity"]["scene"] != env.batch.artifact.content_digest
        ):
            raise ValueError("evaluation scene differs from checkpoint")
        del ck
        common = {
            "data_manifest_sha256": env.motion.identity,
            "scene_content_digest": env.batch.artifact.content_digest,
            "profile_sha256": env.profile_digest,
            "control_profile": controls,
            "sources": shared_sources(),
            "software": {
                p: importlib.metadata.version(p)
                for p in ("gobot", "torch", "mujoco", "mujoco-warp", "warp-lang")
            },
            "backend": cfg.backend,
            "nconmax": 128,
            "njmax": 500,
            "num_envs": num_envs,
            "physics_dt": env.batch.physics_dt,
            "decimation": env.batch.decimation,
            "capture_physics_graphs": cfg.capture_graphs,
            "warp_deterministic_reductions": False,
            "reference": "original FP32 root/joints; shared scene FK",
            "tf32": False,
            "inference_dtype": "float32",
            "actor_noise": False,
            "domain_randomization": False,
            "reset_noise": False,
            "friction": 1.0,
            "reset": "reference frame0 pose/velocity; zero action and repeated initial history; first target frame1",
            "end": "N frames -> N-1 control steps; stop on first failure or final frame",
            "failure": {
                "body": "pelvis",
                "height": env.profile["height_threshold"],
                "gravity_z": env.profile["tilt_threshold"],
            },
            "reward": asdict(TrackingRewardCfg()),
            "anchors": list(ANCHORS),
            "body_measurement": "one physics substep before final integration; terminal joints for limit penalty",
            "score": "per clip sum(reward)/(0.1*executed_steps), then mean over clips",
            "max_steps": max_steps,
        }
        names = env.motion.names if clips is None else list(clips)
        if (
            not names
            or len(names) != len(set(names))
            or not set(names) <= set(env.motion.names)
        ):
            raise ValueError("invalid clips")
        selected = names[shard_index::shard_count]
        if not selected:
            raise ValueError("empty shard")
        index = {name: i for i, name in enumerate(env.motion.names)}
        ends = env.motion.motion_ends.cpu().tolist()
        starts = [0, *ends[:-1]]
        state = env.reset()
        active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        sums = {name: torch.zeros(num_envs, device=device) for name in ANCHORS}
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        lane = [None] * num_envs
        queue = 0
        rows = []
        next_log = 0.0
        partial.write_text("")
        while queue < len(selected) or bool(active.any()):
            free = (~active).nonzero().flatten().tolist()
            assigned = free[: min(len(free), len(selected) - queue)]
            ids = torch.tensor(assigned, dtype=torch.long, device=device)
            if assigned:
                frames = []
                for e in assigned:
                    name = selected[queue]
                    queue += 1
                    i = index[name]
                    lane[e] = (name, i)
                    frames.append(starts[i])
                env._reset(ids, torch.tensor(frames, device=device))
                env.frame[ids] += 1
                env.refresh_reference()
                state = env.full_state()
                active[ids] = True
                steps[ids] = 0
                for total in sums.values():
                    total[ids] = 0
            actions = adapter(env, state, ids)
            state, _, term, trunc, info = env.step(actions)
            for anchor in ANCHORS:
                sums[anchor][active] += info["anchor_rewards"][anchor][active]
            steps[active] += 1
            capped = (
                (steps >= max_steps) & active
                if max_steps is not None
                else torch.zeros_like(active)
            )
            done = active & (term | trunc | capped)
            new = []
            for e in done.nonzero().flatten().tolist():
                name, i = lane[e]
                n = int(steps[e])
                expected = ends[i] - starts[i] - 1
                failed = bool(term[e])
                finished = bool(env.motion_end[e])
                returns = {a: float(sums[a][e] / (0.1 * n)) for a in ANCHORS}
                new.append(
                    {
                        "name": name,
                        "frames": expected + 1,
                        "executed_steps": n,
                        "expected_steps": expected,
                        "return": returns["pelvis"],
                        "returns": returns,
                        "coverage": n / expected,
                        "failed": failed,
                        "finished": finished,
                        "completed": finished and not failed,
                        "capped": bool(capped[e]) and not (failed or finished),
                    }
                )
                active[e] = False
            if new:
                rows.extend(new)
                with partial.open("a") as stream:
                    for row in new:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
            now = time.monotonic()
            if now >= next_log or len(rows) == len(selected):
                value = {
                    "state": "evaluating",
                    "shard_index": shard_index,
                    "policy": adapter.description["kind"],
                    "completed_clips": len(rows),
                    "total_clips": len(selected),
                    "failed_clips": sum(r["failed"] for r in rows),
                    "elapsed_seconds": now - started,
                }
                write_json(progress, value)
                print(json.dumps(value), flush=True)
                next_log = now + 30
        if len(rows) != len(selected) or {r["name"] for r in rows} != set(selected):
            raise RuntimeError("queue lost clips")
        rows.sort(key=lambda r: index[r["name"]])
        if digest(checkpoint) != checkpoint_hash:
            raise ValueError("checkpoint changed during evaluation")
        result = {
            "protocol": PROTOCOL,
            "checkpoint_sha256": checkpoint_hash,
            "policy_adapter": adapter.description,
            "common": common,
            "common_sha256": hash_value(common),
            "selection_count": len(names),
            "selection_sha256": hash_value(names),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "complete": not any(r["capped"] for r in rows),
            "clips": rows,
            "summary": summarize(rows),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output, result)
        write_json(
            progress,
            {
                "state": "completed" if result["complete"] else "incomplete",
                "completed_clips": len(rows),
                "total_clips": len(selected),
                **result["summary"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
        )
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "project", "library", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--clips", nargs="+")
    p.add_argument("--max-steps", type=int)
    result = evaluate(**vars(p.parse_args()))
    print(json.dumps(result["summary"], indent=2))
    if not result["complete"]:
        raise SystemExit("capped evaluation is incomplete")


if __name__ == "__main__":
    main()
