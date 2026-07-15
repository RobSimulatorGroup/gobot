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


def _embed_manifest(output_path: Path, manifest: PolicyManifest) -> None:
    import onnx

    model = onnx.load(str(output_path))
    properties = {entry.key: entry.value for entry in model.metadata_props}
    properties[ONNX_POLICY_MANIFEST_KEY] = manifest.to_json()
    onnx.helper.set_model_props(model, properties)
    onnx.save(model, str(output_path))


def export_policy(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = policy_manifest_from_checkpoint(checkpoint)
    if manifest is None:
        raise RuntimeError(
            "checkpoint has no Gobot policy manifest; retrain with the current training script "
            "before exporting"
        )
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise RuntimeError("checkpoint has no actor_state_dict")
    policy = CheckpointPolicy(actor_state, manifest).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_obs = torch.zeros(1, policy.obs_size, dtype=torch.float32)
    batch_dim = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        policy,
        dummy_obs,
        output_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_shapes={"obs": {0: batch_dim}},
        opset_version=18,
        external_data=False,
    )
    _embed_manifest(output_path, manifest)
    sidecar = write_policy_manifest_sidecar(output_path, manifest)
    print(f"Exported {checkpoint_path} -> {output_path}")
    print(f"Policy manifest: {sidecar}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR.parent / "policies" / "go1_velocity.pt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "policies" / "go1_velocity.onnx")
    args = parser.parse_args()

    export_policy(args.checkpoint.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
