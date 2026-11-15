"""Export a velocity actor after validating its scene and Torch/ONNX inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from gobot.rl.export import export_actor
from gobot.rl.checkpoint_policy import CheckpointPolicy
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.manifest import deployment_manifest, validate_policy
from .presets import Experiment, task_config
from .assessment import fixed_commands


def verify_export(checkpoint, output, *, project_path=None):
    import onnxruntime as ort

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    # Training checkpoints include NumPy RNG/state arrays in addition to tensors.
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
               np.dtypes.UInt32DType, np.dtypes.Int64DType,
               np.dtypes.Float32DType, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest = policy_manifest_from_checkpoint(payload)
    if manifest is None:
        raise ValueError('checkpoint has no Gobot policy manifest')
    torch.set_num_threads(4)
    cfg = task_config(Experiment(**payload['infos']['experiment']), play=True)
    if project_path is not None:
        cfg.project_path = Path(project_path).resolve()
    fixed_commands(cfg)
    env = make_velocity_env(cfg, num_envs=2, backend='mujoco-cpu', collect_step_extras=False)
    try:
        deployed = deployment_manifest(manifest, env, cfg)
        validate_policy(deployed, env, cfg)
        policy = CheckpointPolicy(payload['actor_state_dict'], manifest).eval()
        observations = []
        with torch.inference_mode():
            for _ in range(64):
                env.set_command((.8, 0., 0.))
                obs = torch.as_tensor(np.array(env.get_observations()['actor'], copy=True))
                observations.append(obs)
                env.step(policy(obs).numpy())
            batch = torch.cat(observations)
            reference = policy(batch).numpy()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.velocity-export-', dir=output.parent) as temporary:
            staged = Path(temporary) / output.name
            export_actor(payload['actor_state_dict'], deployed, staged)
            session = ort.InferenceSession(str(staged), providers=['CPUExecutionProvider'])
            actual = session.run(None, {session.get_inputs()[0].name: batch.numpy()})[0]
            error = float(np.max(np.abs(actual - reference)))
            if not np.isfinite(actual).all() or error > 1e-5:
                raise AssertionError(f'ONNX/Torch action mismatch: {error}')
            report = dict(candidate_only=True, samples=len(batch), maximum_absolute_action_error=error,
                          passed=True, checkpoint_iteration=payload.get('iter'),
                          source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          source_scene_digest=manifest.extras['training_scene_digest'],
                          actor_weights_changed=False)
            # Publish only after the contract and numerical checks have passed.
            shutil.copy2(staged.with_name(staged.name + '.manifest.json'),
                         output.with_name(output.name + '.manifest.json'))
            staged.replace(output)
            output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-path', type=Path)
    print(json.dumps(verify_export(**vars(parser.parse_args()))))


if __name__ == '__main__':
    main()
