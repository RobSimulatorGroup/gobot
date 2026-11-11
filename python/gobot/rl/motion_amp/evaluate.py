"""Evaluate independent first episodes; optionally record Gobot's own renderer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import gobot
from .env import MotionAMPEnv
from .training import validate_manifest
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.checkpoint_policy import CheckpointPolicy


def evaluate(project, checkpoint, *, backend='mujoco-cpu', device=None, num_envs=64,
             seed=1001, video=None):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    saved = policy_manifest_from_checkpoint(payload)
    if saved is None:raise ValueError('AMP checkpoint is missing its policy manifest')
    motion = saved.extras['motion']
    velocity_reference = saved.extras.get('root_linear_velocity_reference', 'base_body_com')
    env = MotionAMPEnv(project, num_envs=num_envs, backend=backend, device=device, seed=seed, motion=motion,
                       velocity_reference=velocity_reference)
    writer, sensor = None, None
    try:
        manifest = validate_manifest(payload,env)
        policy = CheckpointPolicy(payload['actor_state_dict'],manifest).to(env.device).eval()
        state = env.reset(seed=seed)
        initial = env.batch.read_state().base_pose[:, :3].clone()
        active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        seconds = torch.zeros(num_envs, device=env.device)
        displacement = torch.zeros(num_envs, 3, device=env.device)
        failures = torch.zeros_like(active)
        trajectory = []
        if video is not None:
            import imageio.v2 as imageio
            video = Path(video)
            video.parent.mkdir(parents=True, exist_ok=True)
            camera = gobot.create_node('Camera3D', 'AMPEvaluationCamera')
            camera.set('up', (0.,0.,1.)); camera.set('fov_y',45.)
            camera.set('z_near',.05); camera.set('z_far',100.)
            sensor = gobot.render.CameraSensor(camera, root=env.batch.context.root,
                width=960, height=720, outputs=('rgb',), device='cpu')
            writer = imageio.get_writer(str(video), fps=round(1/env.control_dt), macro_block_size=16)
        with torch.no_grad():
            for step in range(env.max_steps):
                action = policy(state.obs['actor'])
                state = env.step(action)
                if not bool(state.info['transition_observation']['valid'].all()):
                    raise FloatingPointError('numerical failure during policy evaluation; no valid score can be reported')
                physical = env.batch.read_state()
                if bool(active[0]):
                    position = physical.base_pose[0,:3].cpu().numpy().copy()
                    trajectory.append({'time':(step+1)*env.control_dt, 'position':position.tolist(),
                        'velocity':physical.base_velocity[0,:3].cpu().tolist(), 'failure':bool(state.terminated[0])})
                    if writer is not None:
                        from PIL import Image, ImageDraw
                        env.batch.sync_scene(0)
                        camera.set('eye', tuple(position+(2.6,-3.6,1.0)))
                        camera.set('target', tuple(position+(0.,0.,-.1)))
                        frame = sensor.capture()
                        image = Image.fromarray(frame['rgb'].numpy().copy())
                        del frame
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0,0,960,42), fill=(18,24,30))
                        draw.text((14,12), f'Gobot AMP | {motion} | {(step+1)*env.control_dt:.2f}s | vx={physical.base_velocity[0,0]:.2f} m/s', fill='white')
                        writer.append_data(np.asarray(image))
                finished = active & state.done
                seconds[finished] = (step+1)*env.control_dt
                displacement[finished] = physical.base_pose[finished,:3]-initial[finished]
                failures[finished] = state.terminated[finished]
                active &= ~finished
                if not bool(active.any()):
                    break
                state = env.reset(mask=state.done)
        if bool(active.any()):
            raise RuntimeError('evaluation ended without first-episode terminal observations')
        forward = displacement[:,0]/seconds
        lateral = displacement[:,1]/seconds
        result = {'schema':'gobot_motion_amp_evaluation_v1', 'backend':backend, 'seed':seed,
            'velocity_reference':velocity_reference,
            'episodes':num_envs, 'failures':int(failures.sum()), 'full_length_episodes':int((~failures).sum()),
            'median_episode_seconds':float(seconds.median()), 'median_forward_speed':float(forward.median()),
            'median_absolute_lateral_speed':float(lateral.abs().median()),
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'initialization':'random reference phase, reference pose and velocity',
            'episode_seconds':seconds.cpu().tolist(), 'forward_speed':forward.cpu().tolist(),
            'displacement':displacement.cpu().tolist(), 'trajectory_lane_0':trajectory}
        return result
    finally:
        if writer is not None: writer.close()
        sensor = None
        if video is not None:
            import gc
            gc.collect()
        env.close()
        if video is not None:
            gobot.render._shutdown_headless_render_context()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('mujoco-cpu','mujoco-warp'), default='mujoco-cpu')
    parser.add_argument('--device')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    result = evaluate(args.project,args.checkpoint,backend=args.backend,device=args.device,
                      num_envs=args.num_envs,seed=args.seed,video=args.video)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))


if __name__ == '__main__':
    main()
