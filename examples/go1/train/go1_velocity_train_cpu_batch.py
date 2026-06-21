"""Train the Go1 velocity policy with CPU batch defaults."""

from __future__ import annotations

from typing import Sequence

try:
    from .go1_velocity_train import build_arg_parser, run_training
except ImportError:
    from go1_velocity_train import build_arg_parser, run_training


def parse_args(argv: Sequence[str] | None = None):
    parser = build_arg_parser()
    parser.description = "Train the Go1 velocity policy with Gobot CPU batch defaults."
    parser.set_defaults(
        task="go1_flat",
        device="cpu",
        num_envs=64,
        sim_workers=0,
        render_video_interval=0,
        log_dir="logs/go1_velocity_cpu_batch",
        policy_out="policies/go1_velocity_cpu_batch.pt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
