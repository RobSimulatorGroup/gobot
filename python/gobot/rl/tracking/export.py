"""Export a tracking actor with its verified authoring assets for another machine."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from gobot.rl.policy import scene_bundle_digest
from .inference import load_policy


DEPLOYMENT_SCHEMA = "gobot_tracking_deployment_v1"


def export(checkpoint, output, *, project_path=None):
    """Run on the training host; retain exact weights and bind the source scene."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    env, _, _ = load_policy(checkpoint, backend="mujoco-cpu", project_path=project_path)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # load_policy has verified the training contract on this machine.
        source_contract = deepcopy(payload["infos"]["tracking_contract"])
        snapshot = json.loads(json.dumps(env.batch.snapshot, default=lambda value: value.tolist()))
        deployment = dict(
            schema=DEPLOYMENT_SCHEMA,
            source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            scene_bundle_digest=scene_bundle_digest(env.project_path, env.scene_path),
            physics_snapshot=snapshot,
        )
        output.mkdir(parents=True)
        (output / "config.json").write_bytes(checkpoint.with_name("config.json").read_bytes())
        torch.save(dict(actor_state_dict=payload["actor_state_dict"],
                        infos=dict(tracking_contract=source_contract, deployment=deployment)),
                   output / "policy.pt")
        (output / "export.json").write_text(json.dumps({
            key: value for key, value in deployment.items() if key != "physics_snapshot"
        }, indent=2) + "\n")
        return output / "policy.pt"
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-path", type=Path)
    print(export(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
