#!/usr/bin/env python3
"""Export a manifest-backed rsl_rl checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gobot.rl.policy import (
    ONNX_POLICY_MANIFEST_KEY,
    PolicyManifest,
    policy_manifest_from_checkpoint,
    write_policy_manifest_sidecar,
)

from .checkpoint_policy import CheckpointPolicy


SCRIPT_DIR = Path(__file__).resolve().parent


from gobot.rl.export import export_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR.parent / "policies" / "go1_velocity.pt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "policies" / "go1_velocity.onnx")
    args = parser.parse_args()

    export_policy(args.checkpoint.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
