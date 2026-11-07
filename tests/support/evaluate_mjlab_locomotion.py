"""Evaluate an exported actor in mjlab on shared terrain and reset states.

Run with mjlab dependencies. This process does not import Gobot. Inputs are
policy.ts, scenarios.json and terrain XML exported by the comparison driver.
This tests policy execution; it does not compare independently trained policies.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import subprocess
import sys
import types

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
# Load the velocity factory directly, without importing unrelated registered tasks.
import mjlab
if 'mjlab.tasks' not in sys.modules:
    tasks = types.ModuleType('mjlab.tasks')
    tasks.__path__ = [str(Path(mjlab.__file__).parent/'tasks')]
    sys.modules['mjlab.tasks'] = tasks
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.terrains import TerrainEntityCfg


def make_env(case, directory, device):
    from mujoco_warp._src import collision_convex
    collision_convex.MJ_MAX_EPAHORIZON = max(collision_convex.MJ_MAX_EPAHORIZON, 64)
    cfg = unitree_g1_rough_env_cfg(play=True)
    cfg.scene.num_envs = case['count']
    cfg.scene.terrain = TerrainEntityCfg(terrain_type='plane')
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.seed = case['seed']
    cfg.sim.njmax = 1500

    def terrain(spec):
        body = spec.body('terrain')
        for geom in list(body.geoms):
            spec.delete(geom)
        other = mujoco.MjSpec.from_file(str(directory/case['terrain']))
        for hfield in other.hfields:
            spec.add_hfield(name=hfield.name, nrow=hfield.nrow, ncol=hfield.ncol,
                            size=hfield.size, userdata=hfield.userdata)
        fields = ('name', 'type', 'size', 'pos', 'quat', 'contype', 'conaffinity', 'condim',
                  'priority', 'friction', 'solref', 'solimp', 'solmix', 'margin', 'gap', 'hfieldname')
        for geom in other.geoms:
            body.add_geom(**{name:getattr(geom,name) for name in fields}, group=0)

    cfg.scene.spec_fn = terrain
    command = cfg.commands['twist']
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (1e9, 1e9)
    for name in ('rel_standing_envs', 'rel_heading_envs', 'rel_world_envs', 'rel_forward_envs'):
        setattr(command, name, 0.)
    command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0., 0.)

    def capture(env, env_ids=None):
        data = env.scene['robot'].data
        env.comparison_state = (data.root_link_pos_w.clone(), data.root_link_lin_vel_b.clone())

    cfg.events = {'capture_evaluation': EventTermCfg(func=capture, mode='step')}
    return ManagerBasedRlEnv(cfg, device=device)


def evaluate(env, case, policy, joint_names):
    assert tuple(env.scene['robot'].joint_names) == tuple(joint_names)
    env.reset(seed=case['seed'])
    env.sim.data.qpos.copy_(torch.tensor(case['qpos'], device=env.device))
    env.sim.data.qvel.copy_(torch.tensor(case['qvel'], device=env.device))
    env.sim.forward()
    term = env.command_manager.get_term('twist')
    term.vel_command_b[:] = torch.tensor(case['command'], device=env.device)
    env.sim.sense()
    observation = env.observation_manager.compute(update_history=True)['actor']
    expected = torch.tensor(case['observation'], device=env.device)
    error = (observation-expected).abs()
    initial = {'observation_max_error': float(error.max()),
               'action_max_error': float((policy(observation)-torch.tensor(case['first_action'],device=env.device)).abs().max())}
    # A different observation order/scale would make this an invalid transfer test.
    if max(initial.values()) > 1e-4:
        raise ValueError(f"initial observation differs: {initial}; worst index {int(error.max(0).values.argmax())}")
    count = case['count']
    alive = torch.ones(count, dtype=torch.bool, device=env.device)
    fell = torch.zeros_like(alive)
    reached = torch.zeros_like(alive)
    off_route = torch.zeros_like(alive)
    errors = torch.zeros(count, device=env.device)
    tracking_samples = torch.zeros_like(errors)
    samples = torch.zeros_like(errors)
    centers = torch.tensor(case['centers'], device=env.device)
    command = torch.tensor(case['command'], device=env.device)
    direction = command[:2]/command[:2].norm()
    for step in range(case['steps']):
        term.vel_command_b[:] = command
        observation = env.observation_manager.compute(update_history=True)['actor']
        _, _, terminated, truncated, _ = env.step(policy(observation))
        position, velocity = env.comparison_state
        if not bool(torch.isfinite(position[alive]).all() & torch.isfinite(velocity[alive]).all()):
            raise RuntimeError('nonfinite physics in reference evaluation')
        samples += alive
        if step*env.step_dt >= 2.:
            errors += torch.where(alive, (velocity[:,:2]-command[:2]).norm(dim=1), 0.)
            tracking_samples += alive
        fell |= alive & terminated
        alive &= ~(terminated | truncated)
        if case['route']:
            relative = position[:,:2]-centers
            along = relative@direction
            across = (relative[:,0]*direction[1]-relative[:,1]*direction[0]).abs()
            crossed = (along>=3.25) & (across<3.5)
            wrong = ((across>=3.5) | (along< -3.5)) & ~crossed
            reached |= alive & crossed
            off_route |= alive & wrong
            alive &= ~(crossed | wrong)
        if not bool(alive.any()):
            break
        done = torch.nonzero(terminated | truncated).flatten()
        if len(done):
            env.reset(env_ids=done)
    valid = tracking_samples>0
    return {'name':case['name'], 'seed':case['seed'], 'command':case['command'], 'episodes':count,
            'route':case['route'], 'initial':initial, 'fall_rate':float(fell.float().mean()),
            'terrain_pass_rate':float(reached.float().mean()) if case['route'] else None,
            'off_route_rate':float(off_route.float().mean()),
            'velocity_error_after_warmup':float((errors[valid]/tracking_samples[valid]).mean()) if valid.any() else None,
            'episode_metrics':{'fell':fell.tolist(), 'reached':reached.tolist(), 'off_route':off_route.tolist(),
                               'duration_s':(samples*env.step_dt).tolist(), 'tracking_error_sum':errors.tolist(),
                               'tracking_samples':tracking_samples.tolist(), 'terrain_level':case['terrain_level'],
                               'terrain_type':case['terrain_type']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    import mjlab
    revision = subprocess.check_output(['git','-C',str(Path(mjlab.__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    manifest = json.loads((args.input/'scenarios.json').read_text())
    if revision != manifest['reference_revision']:
        raise ValueError('mjlab revision differs from comparison input')
    policy = torch.jit.load(str(args.input/'policy.ts'), map_location=args.device).eval()
    torch.set_num_threads(4)
    report = {'reference_revision':revision, 'protocol':'same_policy_same_terrain_same_reset',
              'versions':{name:importlib.metadata.version(name) for name in ('mujoco','mujoco-warp','torch','rsl-rl-lib')},
              'checkpoint_iteration':manifest['checkpoint_iteration'], 'capacities':{'njmax':1500,'nconmax':70,'epa_horizon':64}, 'scenarios':[]}
    env = None
    terrain = None
    try:
        with torch.inference_mode():
            for case in manifest['scenarios'][:args.limit]:
                if terrain != case['terrain']:
                    if env is not None:
                        env.close()
                    env = make_env(case, args.input, args.device)
                    terrain = case['terrain']
                result = evaluate(env, case, policy, manifest['joint_names'])
                report['scenarios'].append(result)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print('RESULT', result['name'], result['velocity_error_after_warmup'], result['fall_rate'], result['terrain_pass_rate'], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
