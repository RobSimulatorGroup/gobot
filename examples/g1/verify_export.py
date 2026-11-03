"""Compare an exported ONNX actor with its Torch policy on real Gobot rollouts."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
import torch
from gobot.rl.export import export_policy
from gobot.rl.checkpoint_policy import CheckpointPolicy
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.manifest import validate_policy
from .config import g1_amp_cfg
from .evaluate_amp import fixed_commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    export_policy(args.checkpoint, args.output)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = policy_manifest_from_checkpoint(payload)
    policy = CheckpointPolicy(payload['actor_state_dict'], manifest).eval()
    cfg = g1_amp_cfg(stage='flat', play=True); fixed_commands(cfg)
    env = make_velocity_env(cfg, num_envs=2, backend='mujoco-cpu', collect_step_extras=False)
    try:
        validate_policy(manifest, env, cfg)
        observations = []
        with torch.inference_mode():
            for _ in range(64):
                env.set_command((.8, 0., 0.))
                obs = env.get_observations()['actor'].clone()
                observations.append(obs)
                env.step(policy(obs).numpy())
            batch = torch.cat(observations)
            reference = policy(batch).numpy()
        session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
        actual = session.run(None, {session.get_inputs()[0].name: batch.numpy()})[0]
        error = float(np.max(np.abs(actual - reference)))
        if error > 1e-5:
            raise AssertionError(f'ONNX/Torch action mismatch: {error}')
        report = {'samples': len(batch), 'maximum_absolute_action_error': error, 'passed': True}
        args.output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    finally:
        env.close()

if __name__ == '__main__': main()
