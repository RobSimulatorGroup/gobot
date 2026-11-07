"""Adapt the existing SuperMimic recipe without changing its training clock."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .workflow import TrainingSpec


def _arguments(parser):
    parser.add_argument("--project", type=Path)
    parser.add_argument("--data", type=Path, help="directory containing converted 50 Hz NPZ clips")
    parser.add_argument("--library", type=Path, help="compact packed motion library")
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--stop-iteration", type=int, help="absolute control-step stop; preserves LR schedule")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--capture-graphs", action=argparse.BooleanOptionalAction, default=None)


def _configure(cfg, args):
    changes = {key: getattr(args, key) for key in ("project", "stop_iteration", "capture_graphs")
               if getattr(args, key) is not None}
    if args.output is not None:
        changes.update(log_root=args.output.parent, run_name=args.output.name)
    cfg = replace(cfg, **changes)
    data = {key: value for key, value in (("npz_dir", args.data), ("library", args.library), ("clips", args.clips))
            if value is not None}
    cfg = replace(cfg, data=replace(cfg.data, **data))
    if args.compile is not None:
        mode = "default" if args.compile else None
        cfg = replace(cfg, wm_trainer=replace(cfg.wm_trainer, compile=mode),
                      policy_trainer=replace(cfg.policy_trainer, compile=mode))
    return cfg


def training_spec():
    from gobot.rl.supermimic.config import TrainConfig

    def run(cfg):
        from gobot.rl.motion_tracking.runner import run as run_tracking
        return run_tracking(cfg)

    return TrainingSpec(
        config_factory=TrainConfig, run=run, supports_distributed=True,
        step_unit="control_steps", description="G1 SuperMimic motion tracking",
        common_paths={"backend": "backend", "device": "device", "num_envs": "env.num_envs",
                      "seed": "seed", "resume": "resume"},
        add_arguments=_arguments, configure=_configure,
    )
