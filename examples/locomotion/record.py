"""Record a candidate using its experiment configuration and Gobot renderer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
import gobot
from PIL import Image, ImageDraw
from gobot.rl.checkpoint_policy import CheckpointPolicy
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.manifest import validate_policy
from .presets import Experiment, task_config
from .assessment import fixed_commands, evaluation_scene
from .budget import add_budget_arguments, evaluation_budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=12.)
    p.add_argument('--terrain-type', type=int, default=0)
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--speed', type=float, default=.8)
    p.add_argument('--transition', action='store_true')
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--allow-model-change', action='store_true', help='Record an explicit transfer to the current robot model')
    p.add_argument('--seed', type=int, default=101)
    add_budget_arguments(p)
    args = p.parse_args()
    if args.seconds <= 0 or args.level < 0 or args.terrain_type < 0:
        raise ValueError('recording duration must be positive and terrain indices nonnegative')
    with evaluation_budget(args, stage='record'):
        record(args)


def record(args):
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    experiment = Experiment(**payload['infos']['experiment'])
    cfg = task_config(experiment, play=True)
    fixed_commands(cfg)
    cfg.scene_path = evaluation_scene(cfg, Path(cfg.project_path)/'assets/evaluation/video', args.seed,
        continuous=args.continuous, speed=max(args.speed,3.), seconds=args.seconds)
    cfg.spawn_offset_xy = (-24., 0.) if args.continuous else (-3.25, 0.)
    env = make_velocity_env(cfg, backend='mujoco-warp', num_envs=1, device='cuda:0', seed=args.seed, collect_step_extras=False)
    writer = None
    sensor = view = camera = None
    try:
        manifest = policy_manifest_from_checkpoint(payload)
        validate_policy(manifest, env, cfg, allow_model_change=args.allow_model_change)
        policy = CheckpointPolicy(payload['actor_state_dict'], manifest).eval().cuda()
        env.set_terrain_assignments([0 if args.continuous else args.level], [0 if args.continuous else args.terrain_type], fixed=True)
        env.reset(seed=args.seed)
        root = env.context.root
        links, stack = [], [root]
        while stack:
            node = stack.pop()
            if node.type_name == 'Link3D': links.append(node)
            stack.extend(node.children)
        view = env.provider.create_robot_view(robot_name=cfg.robot_name, base_link=cfg.base_link,
            joint_names=cfg.joint_names, link_names=tuple(link.name for link in links),
            scene_context=env.context, scene_links=tuple(links))
        camera = gobot.create_node('Camera3D', 'EvaluationCamera')
        camera.set('up', (0., 0., 1.)); camera.set('fov_y', 45.)
        camera.set('z_near', .05); camera.set('z_far', 160.)
        sensor = gobot.render.CameraSensor(camera, root=root, width=960, height=720, outputs=('rgb',), device='cpu')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '960x720', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', str(args.output)], stdin=subprocess.PIPE)
        frames, trajectory = 0, []
        with torch.inference_mode():
            for step in range(int(args.seconds / env.step_dt)):
                elapsed = step * env.step_dt
                speed = args.speed
                if args.transition:
                    speed = 0. if elapsed < 2 or elapsed >= 16 else .8 if elapsed < 6 or elapsed >= 12 else 2.
                env.set_command((speed, 0., 0.))
                state = env.step(policy(env.get_observations()['actor']))
                physical = state.info['transition_state']
                fallen = bool(state.terminated[0])
                position = physical['base_position'][0].cpu().numpy()
                velocity = float(physical['base_linear_velocity_body'][0, 0])
                trajectory.append({'time': elapsed, 'position': position.tolist(), 'speed': velocity,
                                   'command': speed, 'fallen': fallen,
                                   'contact': physical['foot_contact'][0].cpu().tolist()})
                if fallen or bool(state.truncated[0]):
                    # The environment has reset; never show that reset as recovery.
                    break
                if step % 2 == 0:
                    view.sync_scene()
                    camera.set('eye', tuple(position + (2.6, -3.6, 1.0)))
                    camera.set('target', tuple(position + (0., 0., -.15)))
                    frame = sensor.capture()
                    image = Image.fromarray(frame['rgb'].numpy().copy())
                    del frame
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 960, 52), fill=(18, 24, 30))
                    draw.text((16, 10), f'Gobot {cfg.robot_name} candidate | t={elapsed:.2f}s | command={speed:.2f} m/s | measured={velocity:.2f} m/s', fill='white')
                    if frames == 0: image.save(args.output.with_suffix('.png'))
                    writer.stdin.write(np.asarray(image).tobytes())
                    frames += 1
                if not args.continuous:
                    level, kind = map(lambda x: int(x[0]), env.terrain_assignments)
                    origin = env.terrain_origins[level * env.terrain_grid_shape[1] + kind].cpu().numpy()
                    if position[0] - origin[0] >= 3.25: break
        writer.stdin.close()
        if writer.wait() != 0: raise RuntimeError('video encoder failed')
        writer = None
        args.output.with_suffix('.json').write_text(json.dumps({'checkpoint': str(args.checkpoint), 'frames': frames,
            'scenario': cfg.scene_path, 'model_change_allowed': args.allow_model_change, 'candidate_only': True, 'trajectory': trajectory}, indent=2) + '\n')
        print(f'Recorded {frames} frames to {args.output}; fallen={trajectory[-1]["fallen"] if trajectory else None}')
    finally:
        if writer is not None:
            if writer.stdin: writer.stdin.close()
            writer.wait()
        sensor = view = camera = None
        import gc
        gc.collect()
        env.close()
        gobot.render._shutdown_headless_render_context()

if __name__ == '__main__': main()
