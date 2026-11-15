"""One dance, one policy. Launch with Python or torchrun --nproc-per-node=8."""

import argparse
from gobot.rl.tracking import train_motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project",
        required=True,
        help="Prepared Gobot project with the dance reference",
    )
    p.add_argument("--motion", default="motion.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--num-envs", type=int, default=4096, help="Environments per GPU")
    p.add_argument("--updates", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    args = p.parse_args()
    train_motion(
        args.project,
        args.motion,
        output=args.output,
        num_envs=args.num_envs,
        updates=args.updates,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
