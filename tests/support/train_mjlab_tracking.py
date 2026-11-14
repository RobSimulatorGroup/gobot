"""Native pinned mjlab tracking baseline; this process never imports Gobot.

Run this file with mjlab's prepared environment. The prepare subcommand uses
upstream interpolation and physics FK once; the identical NPZ is used by both
engines. Training uses the upstream MotionTrackingOnPolicyRunner unchanged.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from mjlab_tracking_reference import (
    reference,
    export_motion,
    sha256,
    write_json,
    TrackingLog,
)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2] / "mjlab"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--csv", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--reference", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--num-envs", type=positive, default=1024)
    train.add_argument("--iterations", type=positive, default=30000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save-interval", type=positive, default=500)
    train.add_argument("--nconmax", type=positive, default=128)
    train.add_argument("--njmax", type=positive, default=500)
    train.add_argument("--resume", type=Path)
    train.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        export_motion(args.source, args.csv, args.output)
        return
    identity = reference(args.source)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    motion = args.reference.resolve() / "motion.npz"
    source_info = json.loads((args.reference / "reference.json").read_text())
    if sha256(motion) != source_info["motion_sha256"]:
        raise ValueError("reference motion digest changed")
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.cuda.set_device(local)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local)
    configure_torch_backends(allow_tf32=False)
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed + rank
    cfg.commands["motion"].motion_file = str(motion)
    cfg.sim.nconmax = args.nconmax
    cfg.sim.njmax = args.njmax
    agent = unitree_g1_tracking_ppo_runner_cfg()
    agent.seed = cfg.seed
    agent.max_iterations = args.iterations
    agent.save_interval = args.save_interval
    agent.logger = "tensorboard"
    agent.upload_model = False
    settings = dict(
        identity,
        motion_sha256=source_info["motion_sha256"],
        source_csv_sha256=source_info["csv_sha256"],
        world_size=world,
        num_envs_per_rank=args.num_envs,
        iterations=args.iterations,
        rollout_steps=agent.num_steps_per_env,
        seed=args.seed,
        allow_tf32=False,
        nconmax=args.nconmax,
        njmax=args.njmax,
        sampler="native rank-local per-control-step EMA",
        init_at_random_ep_len=True,
        solver=asdict(cfg.sim.mujoco),
        statistics="completed episodes across all ranks",
        algorithm="unmodified mjlab MotionTrackingOnPolicyRunner and RSL-RL PPO",
    )
    if args.check:
        if rank == 0:
            print(json.dumps(settings, indent=2))
        return
    out = args.output.resolve()
    if args.resume:
        old = json.loads((args.resume.parent / "comparison.json").read_text())
        for key in (
            "reference_revision",
            "versions",
            "source_sha256",
            "motion_sha256",
            "world_size",
            "num_envs_per_rank",
            "rollout_steps",
            "seed",
            "allow_tf32",
            "nconmax",
            "njmax",
            "solver",
        ):
            if settings[key] != old[key]:
                raise ValueError(f"resume differs: {key}")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "comparison.json", settings)
        dump_yaml(out / "params/env.yaml", asdict(cfg))
        dump_yaml(out / "params/agent.yaml", asdict(agent))
    started = time.monotonic()
    env = None
    try:
        env = ManagerBasedRlEnv(cfg, device=f"cuda:{local}")
        runner = MotionTrackingOnPolicyRunner(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions),
            asdict(agent),
            str(out),
            f"cuda:{local}",
        )
        TrackingLog(env, runner, out, settings)
        if args.resume:
            runner.load(str(args.resume))
            runner.current_learning_iteration += 1
        remaining = args.iterations - runner.current_learning_iteration
        if remaining <= 0:
            raise ValueError("iterations must exceed completed updates")
        runner.add_git_repo_to_log(
            str(Path(args.source) / "src/mjlab/scripts/train.py")
        )
        runner.learn(remaining, init_at_random_ep_len=True)
        if rank == 0:
            runner.save(str(out / "model.pt"))
            write_json(
                out / "status.json",
                dict(
                    state="trained",
                    last_iteration=runner.current_learning_iteration,
                    elapsed_s=time.monotonic() - started,
                    world_size=world,
                    samples=args.iterations
                    * world
                    * args.num_envs
                    * agent.num_steps_per_env,
                ),
            )
    except BaseException as error:
        if rank == 0 and out.exists():
            write_json(out / "status.json", dict(state="failed", error=str(error)))
        raise
    finally:
        if env is not None:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
