#!/usr/bin/env python3
"""Export a CartPole rsl_rl checkpoint to an ONNX inference graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OBS_SIZE = 7
ACTION_SIZE = 1
HIDDEN_DIMS = (128, 128, 64)
SCRIPT_DIR = Path(__file__).resolve().parent


class CartPolePolicy(torch.nn.Module):
    def __init__(self, actor_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("obs_mean", actor_state["obs_normalizer._mean"].to(dtype=torch.float32))
        self.register_buffer("obs_std", actor_state["obs_normalizer._std"].to(dtype=torch.float32).clamp_min(1.0e-6))

        layers: list[torch.nn.Module] = []
        input_size = OBS_SIZE
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
        obs = (obs - self.obs_mean) / self.obs_std
        return self.mlp(obs)


def export_policy(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint["actor_state_dict"]
    policy = CartPolePolicy(actor_state).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_obs = torch.zeros(1, OBS_SIZE, dtype=torch.float32)
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
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR.parent / "policies" / "cartpole.pt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "policies" / "cartpole.onnx")
    args = parser.parse_args()

    export_policy(args.checkpoint.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
