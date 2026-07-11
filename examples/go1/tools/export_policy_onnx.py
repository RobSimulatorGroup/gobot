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


SCRIPT_DIR = Path(__file__).resolve().parent


class CheckpointPolicy(torch.nn.Module):
    def __init__(
        self,
        actor_state: dict[str, torch.Tensor],
        manifest: PolicyManifest,
    ) -> None:
        super().__init__()
        self.obs_size = int(manifest.observation_spec["dim"])
        self.action_size = int(manifest.action_spec["dim"])
        layer_indices = _checkpoint_mlp_layer_indices(actor_state)
        if not layer_indices:
            raise RuntimeError("checkpoint actor_state_dict does not contain mlp.*.weight tensors")

        layers: list[torch.nn.Module] = []
        input_size = self.obs_size
        activation = str(manifest.model.get("activation", "elu"))
        for position, layer_index in enumerate(layer_indices):
            weight = actor_state[f"mlp.{layer_index}.weight"].to(dtype=torch.float32)
            bias = actor_state[f"mlp.{layer_index}.bias"].to(dtype=torch.float32)
            output_size, checkpoint_input_size = (int(value) for value in weight.shape)
            if checkpoint_input_size != input_size:
                raise RuntimeError(
                    f"checkpoint MLP layer {layer_index} input mismatch: "
                    f"checkpoint={checkpoint_input_size}, expected={input_size}"
                )
            linear = torch.nn.Linear(input_size, output_size)
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            if position + 1 < len(layer_indices):
                layers.append(_activation_module(activation))
            input_size = output_size
        if input_size != self.action_size:
            raise RuntimeError(
                f"checkpoint action dimension mismatch: checkpoint={input_size}, "
                f"manifest={self.action_size}"
            )
        self.mlp = torch.nn.Sequential(*layers)

        mean = _normalizer_tensor(actor_state, "obs_normalizer._mean")
        std = _normalizer_tensor(actor_state, "obs_normalizer._std")
        if std is None:
            variance = _normalizer_tensor(actor_state, "obs_normalizer._var")
            if variance is not None:
                std = torch.sqrt(torch.clamp(variance, min=1.0e-12))
        self.use_obs_normalizer = mean is not None
        if mean is None:
            mean = torch.zeros((self.obs_size,), dtype=torch.float32)
        if std is None:
            std = torch.ones((self.obs_size,), dtype=torch.float32)
        if mean.numel() != self.obs_size or std.numel() != self.obs_size:
            raise RuntimeError(
                "checkpoint observation normalizer dimension does not match policy manifest"
            )
        self.register_buffer("obs_mean", mean.reshape(1, -1))
        self.register_buffer("obs_std", std.reshape(1, -1).clamp_min(1.0e-6))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_obs_normalizer:
            obs = (obs - self.obs_mean) / self.obs_std
        return self.mlp(obs)


def _checkpoint_mlp_layer_indices(actor_state: dict[str, torch.Tensor]) -> list[int]:
    indices: list[int] = []
    for key, value in actor_state.items():
        if not key.startswith("mlp.") or not key.endswith(".weight") or value.ndim != 2:
            continue
        parts = key.split(".")
        if len(parts) == 3 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return sorted(indices)


def _normalizer_tensor(
    actor_state: dict[str, torch.Tensor],
    name: str,
) -> torch.Tensor | None:
    value = actor_state.get(name)
    return None if value is None else torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def _activation_module(name: str) -> torch.nn.Module:
    normalized = name.strip().lower()
    activations: dict[str, type[torch.nn.Module]] = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh,
    }
    activation = activations.get(normalized)
    if activation is None:
        raise RuntimeError(f"unsupported checkpoint activation {name!r}")
    return activation()


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
