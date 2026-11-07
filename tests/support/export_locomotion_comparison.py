"""Export identical G1 reset states, terrain and actor for independent mjlab evaluation.

Run with Gobot. Outputs are experiment artifacts, not training assets. The actor
may have been trained on a different model: --allow-model-change records that
explicit transfer while still validating joint order, observations and controls.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import torch
from gobot.rl.checkpoint_policy import CheckpointPolicy
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.experiment import Experiment, task_config, MJLAB_REFERENCE_REVISION
from gobot.rl.locomotion.manifest import validate_policy
from examples.locomotion.assessment import fixed_commands, evaluation_scene


def export(checkpoint, output, *, seeds=(71,72,73), device='cuda:0', allow_model_change=False):
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
    experiment = Experiment(**payload['infos']['experiment'])
    if (experiment.robot, experiment.recipe, experiment.stage) != ('g1','baseline','rough'):
        raise ValueError('reference comparison currently requires a G1 rough velocity PPO policy')
    manifest = policy_manifest_from_checkpoint(payload)
    policy = CheckpointPolicy(payload['actor_state_dict'], manifest).eval()
    torch.jit.trace(policy, torch.zeros(1, manifest.observation_spec['dim'])).save(str(output/'policy.ts'))
    scenarios = []
    for seed in seeds:
        for terrain,count,speeds in [('flat',32,(-.3,.8,1.5,2.5)), ('rough',70,(.8,))]:
            cfg = task_config(experiment, play=True)
            fixed_commands(cfg)
            cfg.spawn_jitter = .1
            cfg.terrain_out_of_bounds = False
            cfg.scene_path = evaluation_scene(cfg, Path(cfg.project_path)/'assets/evaluation'/f'comparison_{terrain}_{seed}',
                                              seed, continuous=terrain=='flat', seconds=20)
            env = make_velocity_env(cfg, backend='mujoco-warp', num_envs=count, device=device, seed=seed,
                                    collect_step_extras=False)
            try:
                validate_policy(manifest, env, cfg, allow_model_change=allow_model_change)
                # The reference authors its own robot. Only terrain crosses this boundary.
                source = ET.fromstring(env.compiled_artifact.content)
                target = ET.Element('mujoco')
                assets = ET.SubElement(target, 'asset')
                body = ET.SubElement(ET.SubElement(target, 'worldbody'), 'body', name='terrain')
                for asset in source.findall('./asset/hfield'):
                    assets.append(deepcopy(asset))
                for geom in source.findall('./worldbody/geom'):
                    geom = deepcopy(geom)
                    geom.set('group', '0')
                    body.append(geom)
                terrain_name = f'{terrain}_{seed}.xml'
                ET.ElementTree(target).write(output/terrain_name, encoding='unicode')
                rows,cols = env.terrain_grid_shape
                indices = np.arange(count)%(rows*cols)
                env.set_terrain_assignments(indices//cols, indices%cols, fixed=True)
                for speed in speeds:
                    cfg.spawn_offset_xy = (-3.25 if terrain=='rough' else -30.,0.) if speed>=0 else (30.,0.)
                    env.reset(seed=seed)
                    env.set_command((speed,0.,0.))
                    observation = env.get_observations()['actor'].cpu()
                    case = {'name':f'{terrain}_{seed}_{speed:g}', 'terrain':terrain_name, 'seed':seed,
                            'route':terrain=='rough', 'command':[speed,0.,0.], 'steps':1000, 'count':count,
                            'terrain_level':(indices//cols).tolist(), 'terrain_type':(indices%cols).tolist(),
                            'centers':env.terrain_origins[indices,:2].tolist(),
                            'qpos':env.provider.arrays['qpos'].cpu().tolist(),
                            'qvel':env.provider.arrays['qvel'].cpu().tolist(), 'observation':observation.tolist(),
                            'first_action':policy(observation).detach().tolist()}
                    scenarios.append(case)
            finally:
                env.close()
    report = {'joint_names':list(manifest.joint_names), 'reference_revision':MJLAB_REFERENCE_REVISION,
              'checkpoint_iteration':payload['iter'], 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'model_change_allowed':allow_model_change, 'scenarios':scenarios}
    (output/'scenarios.json').write_text(json.dumps(report)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[71,72,73])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-model-change', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.inference_mode():
        export(args.checkpoint, args.output, seeds=args.seeds, device=args.device, allow_model_change=args.allow_model_change)
