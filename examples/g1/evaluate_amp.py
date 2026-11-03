"""Deterministic locomotion screening and per-cell, pre-reset trajectory metrics."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict

from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.manifest import validate_policy
from gobot.rl.checkpoint_policy import CheckpointPolicy
from gobot.rl.policy import policy_manifest_from_checkpoint
from .config import g1_amp_cfg
from gobot.rl.locomotion.gait_metrics import BipedGaitMetrics


def _np(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def rollout(env, action_fn, *, command=(.8, 0., 0.), steps=1000, cell_exit=True, seed=101):
    """Count each episode once. All metrics use physical state before automatic reset."""
    if steps <= 0:
        raise ValueError('evaluation steps must be positive')
    count = env.num_envs
    gait = BipedGaitMetrics(count, env.step_dt)
    commands = np.broadcast_to(np.asarray(command, np.float32), (count, 3)).copy()
    env.reset(seed=seed)
    env.set_command(commands)
    origins = _np(env.locomotion_state()['base_position']).copy()
    terrain_origins = _np(env.terrain_origins)
    levels, types = map(_np, env.terrain_assignments)
    centers = terrain_origins[levels * env.terrain_grid_shape[1] + types, :2]
    direction = np.asarray(command[:2]) / max(np.linalg.norm(command[:2]), 1e-6)
    live = np.ones(count, bool)
    failed = np.zeros(count, bool)
    crossed = np.zeros(count, bool)
    samples = np.zeros(count)
    error = np.zeros(count)
    progress = np.zeros(count)
    yaw_progress = np.zeros(count)
    yaw_error = np.zeros(count)
    slip = np.zeros(count)
    flight = np.zeros(count)
    min_clearance = np.full(count, np.inf)
    stance = np.zeros((count, 2))
    touchdowns = np.zeros((count, 2), int)
    last_contact = np.ones((count, 2), bool)
    displacement = np.zeros((count, 2))
    reasons = np.zeros(count, int)
    with torch.inference_mode():
        for _ in range(steps):
            env.set_command(commands)
            state = env.step(action_fn(env.get_observations()))
            data = {k: _np(v) for k, v in state.info['transition_state'].items()}
            velocity = data['base_linear_velocity_body'][:, :2]
            contact = np.linalg.norm(data['foot_force'], axis=-1) > 10.
            gait.update(contact, data['foot_position'], live)
            error[live] += np.linalg.norm(velocity[live] - commands[live, :2], axis=1)
            yaw_error[live] += np.abs(data['base_angular_velocity_body'][live, 2] - commands[live, 2])
            progress[live] += (velocity @ direction)[live]
            yaw_progress[live] += data['base_angular_velocity_body'][live, 2] * np.sign(commands[live, 2])
            slip[live] += (np.square(data['foot_velocity'][live, :, :2]).sum(2) * contact[live]).sum(1)
            min_clearance[live] = np.minimum(min_clearance[live], data['base_clearance'][live])
            flight[live] += ~contact[live].any(1)
            stance[live] += contact[live]
            touchdowns[live] += contact[live] & ~last_contact[live]
            last_contact[live] = contact[live]
            samples[live] += 1
            displacement[live] = data['base_position'][live, :2] - origins[live, :2]
            terminal = _np(state.terminated).astype(bool)
            timeout = _np(state.truncated).astype(bool)
            failed |= live & terminal
            reasons[live & (terminal | timeout)] = _np(state.info['reset_reason'])[live & (terminal | timeout)]
            live &= ~(terminal | timeout)
            if cell_exit:
                relative = data['base_position'][:, :2] - centers
                # Assigned 8 m cells have a 0.5 m evaluation margin. Leaving through
                # another side is failure, never credited to the birth terrain.
                reached = (relative @ direction >= 3.) & (np.abs(relative).max(1) < 3.5)
                wrong_exit = (np.abs(relative[:, 0]) > 3.5) | (np.abs(relative[:, 1]) > 3.5)
                crossed |= live & reached
                failed |= live & wrong_exit & ~reached
                live &= ~(reached | wrong_exit)
            if not live.any():
                break
    divisor = np.maximum(samples, 1)
    speed = np.linalg.norm(commands[:, :2], axis=1)
    average_error = error / divisor
    survival = ~failed & ((reasons == 0) | (reasons == 2))
    admission = survival & (average_error <= np.maximum(.15, .2 * speed)) & (yaw_error / divisor <= .2)
    if cell_exit:
        admission &= crossed
    else:
        admission &= samples >= steps
    gait_values = gait.arrays()
    if not cell_exit and command[0] >= 1.6:
        admission &= (gait_values['flight_fraction'] >= .05) & (gait_values['alternation_fraction'] >= .9)
        admission &= gait_values['landings'] >= 8
    cells = []
    for level, kind in sorted(set(zip(levels.tolist(), types.tolist()))):
        mask = (levels == level) & (types == kind)
        cells.append({'level': level, 'terrain': env.terrain_type_names[kind], 'episodes': int(mask.sum()),
                      'admission': float(admission[mask].mean()), 'survival': float(survival[mask].mean()),
                      'velocity_error': float(average_error[mask].mean()),
                      'flight_fraction': float((flight / divisor)[mask].mean())})
    return {'seed': seed, 'command': list(command), 'steps': steps, 'cell_exit': cell_exit,
            'admission': float(admission.mean()), 'survival': float(survival.mean()),
            'velocity_error': float(average_error.mean()), 'yaw_error': float((yaw_error / divisor).mean()),
            'command_progress_m_s': float((progress / divisor).mean()),
            'command_yaw_progress_rad_s': float((yaw_progress / divisor).mean()),
            'flight_fraction': float((flight / divisor).mean()),
            'minimum_base_clearance_m': float(min_clearance.min()),
            'foot_slip_m2_s2': float((slip / divisor).mean()),
            'duty_factor': (stance / divisor[:, None]).mean(0).tolist(),
            'touchdowns_per_second': (touchdowns / (divisor[:, None] * env.step_dt)).mean(0).tolist(),
            'distance_m': float(np.linalg.norm(displacement, axis=1).mean()),
            'gait': {key: np.asarray(value).mean(axis=0).tolist() for key, value in gait_values.items()}, 'cells': cells}


def fixed_commands(cfg):
    cfg.command_curriculum = ()
    cfg.command.rel_standing_envs = cfg.command.rel_heading_envs = 0.
    cfg.command.rel_world_envs = cfg.command.rel_forward_envs = cfg.command.rel_run_envs = 0.
    cfg.command.heading_command = False
    cfg.command.resampling_time_range = (1e9, 1e9)
    cfg.command.ranges.lin_vel_x = cfg.command.ranges.lin_vel_y = cfg.command.ranges.ang_vel_z = (0., 0.)
    cfg.spawn_jitter = 0.
    cfg.randomize_reset_yaw = False
    cfg.observations.actor_noise = False
    cfg.push_enabled = False


def validate_training_env(env, actor, *, steps=1000):
    saved = env.training_state_dict()
    cfg = env.cfg_obj
    old_command, noise, push = deepcopy(cfg.command), cfg.observations.actor_noise, cfg.push_enabled
    jitter, yaw = cfg.spawn_jitter, cfg.randomize_reset_yaw
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    old_curriculum = cfg.command_curriculum
    initializer = getattr(env, '_reference_initializer', None)
    if initializer is not None: env.set_reference_initializer(None)
    actor.eval()
    try:
        fixed_commands(cfg)
        cfg.command_curriculum = ()
        levels, types = env.terrain_assignments
        env.set_terrain_assignments(levels, types, fixed=True)
        result = rollout(env, lambda obs: actor(TensorDict(obs, batch_size=[env.num_envs]), stochastic_output=False), steps=steps)
        result['protocol'] = 'training_screen_current_terrain_not_final_admission'
        return result
    finally:
        if initializer is not None: env.set_reference_initializer(initializer)
        cfg.command = old_command
        cfg.command_curriculum = old_curriculum
        cfg.observations.actor_noise, cfg.push_enabled = noise, push
        cfg.spawn_jitter, cfg.randomize_reset_yaw = jitter, yaw
        env.set_terrain_assignments(levels, types, fixed=False)
        env.load_training_state_dict(saved)
        torch.set_rng_state(rng)
        if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
        actor.train()


def prepare_evaluation_scene(cfg, seed, *, continuous=False, slope_degrees=0.):
    """Materialize inspectable Gobot scene/resources for held-out terrain geometry."""
    import math
    root = Path(cfg.project_path)
    directory = root / 'assets/evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{'continuous' if continuous else 'cells'}_{seed}_{slope_degrees:g}"
    resource = json.loads((root / ('terrain/flat_terrain.jres' if 'flat' in cfg.scene_path else 'terrain/rough_terrain.jres')).read_text())
    values = resource['__RESOURCE__']; values['seed'] = int(seed)
    if continuous:
        values['num_rows'] = values['num_cols'] = 1
        values['border_width'] = 2.
        values['patch_size']['matrix_data']['storage'] = [64., 64.]
        values['horizontal_scale'] = .2
        terrain = deepcopy(values['sub_terrains'][3 if slope_degrees else 0])
        terrain['name'] = 'continuous_slope' if slope_degrees else 'continuous_flat'
        terrain['type'] = 'PyramidSlope' if slope_degrees else 'Flat'
        terrain['slope_range']['matrix_data']['storage'] = [math.tan(math.radians(slope_degrees))] * 2
        values['sub_terrains'] = [terrain]
    (directory / f'{name}.jres').write_text(json.dumps(resource, indent=2) + '\n')
    scene = (root / 'g1_scene.jscn').read_text().replace('res://terrain/rough_terrain.jres', f'res://assets/evaluation/{name}.jres')
    (directory / f'{name}.jscn').write_text(scene)
    return f'res://assets/evaluation/{name}.jscn'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stage', choices=('flat', 'rough', 'mixed'), default='rough')
    p.add_argument('--command', type=float, nargs=3, default=(.8, 0., 0.))
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--episodes-per-cell', type=int, default=10)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--json-out', type=Path, required=True)
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--slope-degrees', type=float, default=0.)
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = g1_amp_cfg(stage=args.stage, play=True)
    fixed_commands(cfg)
    cfg.scene_path = prepare_evaluation_scene(cfg, args.seed, continuous=args.continuous, slope_degrees=args.slope_degrees)
    if args.continuous:
        count = args.episodes_per_cell
        cfg.spawn_offset_xy = (-24., 0.)
    else:
        count = 70 * args.episodes_per_cell
        direction = np.asarray(args.command[:2]) / max(np.linalg.norm(args.command[:2]), 1e-6)
        cfg.spawn_offset_xy = tuple(-3. * direction)
    env = make_velocity_env(cfg, backend='mujoco-warp', num_envs=count, device='cuda:0', seed=args.seed, collect_step_extras=False)
    try:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        manifest = policy_manifest_from_checkpoint(payload)
        validate_policy(manifest, env, cfg)
        policy = CheckpointPolicy(payload['actor_state_dict'], manifest).eval().cuda()
        rows, cols = env.terrain_grid_shape
        levels = np.repeat(np.arange(rows), cols * args.episodes_per_cell)
        types = np.tile(np.repeat(np.arange(cols), args.episodes_per_cell), rows)
        env.set_terrain_assignments(levels, types, fixed=True)
        result = rollout(env, lambda obs: policy(obs['actor']), command=args.command, steps=args.steps,
                         cell_exit=not args.continuous and np.linalg.norm(args.command[:2]) > .05, seed=args.seed)
        result['terrain_seed'] = args.seed
        result['slope_degrees'] = args.slope_degrees
        result['scenario'] = cfg.scene_path
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'cells'}), flush=True)
    finally:
        env.close()

if __name__ == '__main__':
    main()
