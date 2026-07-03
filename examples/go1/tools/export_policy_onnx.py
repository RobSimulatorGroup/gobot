#!/usr/bin/env python3
"""Export a Go1 rsl_rl checkpoint to an ONNX inference graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gobot.rl.locomotion import velocity_actor_observation_schema

ACTION_SIZE = 12
HIDDEN_DIMS = (512, 256, 128)
TERRAIN_SCAN_GRID_SIZE = (1.6, 1.0)
TERRAIN_SCAN_GRID_RESOLUTION = 0.1
TERRAIN_SCAN_DIM = (
    int(round(TERRAIN_SCAN_GRID_SIZE[0] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
) * (
    int(round(TERRAIN_SCAN_GRID_SIZE[1] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
)
DEFAULT_OBS_SIZE = velocity_actor_observation_schema(ACTION_SIZE, TERRAIN_SCAN_DIM).dim
SCRIPT_DIR = Path(__file__).resolve().parent


class Go1Policy(torch.nn.Module):
    def __init__(self, actor_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.obs_size = _checkpoint_obs_dim(actor_state)
        self.use_obs_normalizer = "obs_normalizer._mean" in actor_state
        mean = actor_state.get("obs_normalizer._mean", torch.zeros((1, self.obs_size), dtype=torch.float32))
        std = actor_state.get("obs_normalizer._std", torch.ones((1, self.obs_size), dtype=torch.float32))
        self.register_buffer("obs_mean", mean.to(dtype=torch.float32))
        self.register_buffer("obs_std", std.to(dtype=torch.float32).clamp_min(1.0e-6))

        layers: list[torch.nn.Module] = []
        input_size = self.obs_size
        for layer_index, output_size in enumerate((*HIDDEN_DIMS, ACTION_SIZE)):
            linear = torch.nn.Linear(input_size, output_size)
            state_index = layer_index * 2
            linear.weight.data.copy_(actor_state[f"mlp.{state_index}.weight"].to(dtype=torch.float32))
            linear.bias.data.copy_(actor_state[f"mlp.{state_index}.bias"].to(dtype=torch.float32))
            layers.append(linear)
            if output_size != ACTION_SIZE:
                layers.append(torch.nn.ELU())
            input_size = output_size
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_obs_normalizer:
            obs = (obs - self.obs_mean) / self.obs_std.clamp_min(1.0e-6)
        return self.mlp(obs)


def _checkpoint_obs_dim(actor_state: dict[str, torch.Tensor]) -> int:
    normalizer_mean = actor_state.get("obs_normalizer._mean")
    if normalizer_mean is not None and len(normalizer_mean.shape) == 2:
        return int(normalizer_mean.shape[1])
    first_weight = actor_state.get("mlp.0.weight")
    if first_weight is not None and len(first_weight.shape) == 2:
        return int(first_weight.shape[1])
    return DEFAULT_OBS_SIZE


def export_policy(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint["actor_state_dict"]
    policy = Go1Policy(actor_state).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_obs = torch.zeros(1, policy.obs_size, dtype=torch.float32)
    torch.onnx.export(
        policy,
        dummy_obs,
        output_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        external_data=False,
    )
    print(f"Exported {checkpoint_path} -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR.parent / "policies" / "go1.pt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "policies" / "go1.onnx")
    args = parser.parse_args()

    export_policy(args.checkpoint.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
