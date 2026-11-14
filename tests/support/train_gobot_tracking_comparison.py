"""Run Gobot only after the matching native mjlab baseline is complete."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import torch
import gobot
from gobot.rl.tracking.training import Config, run
from gobot.rl.training.workflow import update_config
from gobot.sim.providers.base import CompiledSceneArtifact
from mjlab_tracking_reference import sha256, write_json, VERSIONS, REFERENCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--preflight-updates",
        type=int,
        help="Explicit short validation before the baseline completes; never a final comparator",
    )
    args = p.parse_args()
    base = json.loads((args.baseline / "comparison.json").read_text())
    status = json.loads((args.baseline / "status.json").read_text())
    if base["reference_revision"] != REFERENCE:
        raise ValueError("unexpected mjlab reference")
    if args.preflight_updates is None and status["state"] != "trained":
        raise ValueError("complete native mjlab baseline before Gobot long training")
    if args.preflight_updates is not None and args.preflight_updates < 1:
        raise ValueError("preflight updates must be positive")
    for name, version in VERSIONS.items():
        if metadata.version(name).split("+")[0] != version:
            raise ValueError(f"incompatible {name}")
    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != base["world_size"]:
        raise ValueError("world size differs from baseline")
    cfg = update_config(Config(), json.loads((args.project / "task.json").read_text()))
    cfg.project_path = args.project.resolve()
    cfg.num_envs = base["num_envs_per_rank"]
    cfg.seed = base["seed"]
    cfg.batch_scope = "per_rank"
    cfg.backend = "mujoco-warp"
    cfg.device = "cuda:0"
    cfg.output = args.output
    cfg.resume = args.resume
    cfg.sampler_sync = "rank_local"
    cfg.random_source = "torch_global"
    cfg.init_at_random_ep_len = True
    cfg.allow_tf32 = base["allow_tf32"]
    cfg.nconmax = base["nconmax"]
    cfg.njmax = base["njmax"]
    cfg.updates = base["iterations"]
    if args.preflight_updates is not None:
        cfg.updates = args.preflight_updates
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg.updates -= int(saved["iter"]) + 1
        if cfg.updates <= 0:
            raise ValueError("target updates already reached")
    if sha256(args.project / "motion.npz") != base["motion_sha256"]:
        raise ValueError("motion does not match native baseline")
    audit = json.loads((args.project / "physics-audit.json").read_text())
    if not audit["passed"] or audit["motion_sha256"] != base["motion_sha256"]:
        raise ValueError("missing compatible physics audit")
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(cfg.project_path))
        context.load_scene(cfg.scene_path)
        context.fixed_time_step = cfg.physics_dt
        context.set_mujoco_solver_settings(cfg.solver_settings)
        artifact = CompiledSceneArtifact.from_mapping(
            context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        )
        if artifact.content_digest != audit["compiled_artifact_digest"]:
            raise ValueError("Gobot scene changed since physics audit")
    finally:
        context.clear_scene()
    cfg.validate()
    if args.check:
        print(
            json.dumps(
                dict(
                    passed=True,
                    world_size=world,
                    motion_sha256=base["motion_sha256"],
                    scene_digest=artifact.content_digest,
                ),
                indent=2,
            )
        )
        return
    result = run(cfg)
    if int(os.environ.get("RANK", "0")) == 0:
        write_json(
            args.output / "comparison.json",
            dict(
                base,
                engine="gobot",
                baseline=str(args.baseline.resolve()),
                preflight=args.preflight_updates is not None,
                physics_audit=artifact.content_digest,
            ),
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
