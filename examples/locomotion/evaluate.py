"""Evaluate every authored terrain cell using the public velocity task API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.manifest import validate_policy
from gobot.rl.policy import policy_manifest_from_checkpoint
from gobot.rl.checkpoint_policy import CheckpointPolicy
from .common import task_cfg

def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

def evaluate(robot, checkpoint, *, backend='mujoco-warp', device=None, command=(1., 0., 0.), steps=1000, episodes_per_cell=2, seed=123):
    if steps <= 0 or episodes_per_cell <= 0:
        raise ValueError('steps and episodes-per-cell must be positive')
    cfg = task_cfg(robot)
    cfg.observations.actor_noise = cfg.domain_randomization.enabled = cfg.push_enabled = False
    cfg.spawn_jitter = 0.
    cfg.randomize_reset_yaw = False
    cfg.command_curriculum = ()
    cfg.command.rel_standing_envs = cfg.command.rel_heading_envs = cfg.command.rel_forward_envs = cfg.command.rel_run_envs = 0.
    cfg.command.heading_command = False
    cfg.command.resampling_time_range = (1.e9, 1.e9)
    cfg.command.ranges.lin_vel_x = (command[0], command[0])
    cfg.command.ranges.lin_vel_y = (command[1], command[1])
    cfg.command.ranges.ang_vel_z = (command[2], command[2])
    # The generator uses one column per terrain type in curriculum mode.
    terrain = json.loads((Path(cfg.project_path) / 'terrain/rough_terrain.jres').read_text())['__RESOURCE__']
    rows, cols = terrain['num_rows'], len(terrain['sub_terrains'])
    count = rows * cols * episodes_per_cell
    device = device or ('cuda:0' if backend == 'mujoco-warp' else 'cpu')
    env = make_velocity_env(cfg, backend=backend, device=device, num_envs=count, seed=seed, collect_step_extras=False)
    try:
        if env.terrain_grid_shape != (rows, cols):
            raise ValueError('authored terrain grid does not match generated terrain')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        manifest = policy_manifest_from_checkpoint(payload)
        if manifest is None:
            raise ValueError('checkpoint has no policy manifest')
        validate_policy(manifest, env, cfg)
        policy = CheckpointPolicy(payload['actor_state_dict'], manifest).eval().to(device)
        levels = np.repeat(np.arange(rows), cols * episodes_per_cell)
        types = np.tile(np.repeat(np.arange(cols), episodes_per_cell), rows)
        env.set_terrain_assignments(levels, types, fixed=True)
        obs, _ = env.reset(seed=seed)
        live = np.ones(count, dtype=bool)
        reason = np.zeros(count, dtype=np.int64)
        samples = np.zeros(count, dtype=np.int64)
        progress = np.zeros(count)
        yaw_progress = np.zeros(count)
        error = np.zeros(count)
        slip = np.zeros(count)
        speed = np.linalg.norm(command[:2])
        direction = np.asarray(command[:2]) / max(speed, 1e-6)
        with torch.inference_mode():
            for _ in range(steps):
                action = policy(torch.as_tensor(obs['actor'], dtype=torch.float32, device=device))
                state = env.step(action)
                obs = state.obs
                failed = _numpy(state.terminated).astype(bool)
                truncated = _numpy(state.truncated).astype(bool)
                first_done = live & (failed | truncated)
                reasons = _numpy(state.info.get('reset_reason', np.where(failed, 1, 2)))
                reason[first_done] = reasons[first_done]
                live &= ~(failed | truncated)
                data = {key: _numpy(value) for key, value in env.locomotion_state().items()}
                samples[live] += 1
                velocity = data['base_linear_velocity_body'][:, :2]
                progress[live] += (velocity @ direction)[live]
                yaw_progress[live] += (data['base_angular_velocity_body'][:, 2] * np.sign(command[2]))[live]
                error[live] += np.linalg.norm(velocity - command[:2], axis=1)[live]
                slip[live] += (np.square(data['foot_velocity'][:, :, :2]).sum(axis=2) * (data['foot_contact'] > 0)).sum(axis=1)[live]
        divisor = np.maximum(samples, 1)
        survival = (reason == 0) | (reason == 2)
        admitted = survival & ((progress / divisor >= .5 * speed) if speed else True)
        if abs(command[2]) > 1e-6:
            admitted &= yaw_progress / divisor >= .5 * abs(command[2])
        cells = []
        for level in range(rows):
            for kind in range(cols):
                mask = (levels == level) & (types == kind)
                cells.append({'level': level, 'terrain': env.terrain_type_names[kind],
                    'episodes': int(mask.sum()), 'survival': float(survival[mask].mean()),
                    'admission': float(admitted[mask].mean()), 'command_progress_m_s': float((progress/divisor)[mask].mean()),
                    'velocity_error_m_s': float((error/divisor)[mask].mean()), 'foot_slip_m2_s2': float((slip/divisor)[mask].mean()),
                    'failures': {'terminated': int((reason[mask] == 1).sum()), 'out_of_bounds': int((reason[mask] == 3).sum())}})
        return {'robot': robot, 'backend': backend, 'seed': seed, 'command': command, 'steps': steps,
                'survival': float(survival.mean()), 'admission': float(admitted.mean()), 'cells': cells}
    finally:
        env.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=('go1', 'g1'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--backend', choices=('mujoco-cpu', 'mujoco-warp'), default='mujoco-warp')
    p.add_argument('--device')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--episodes-per-cell', type=int, default=2)
    p.add_argument('--command', type=float, nargs=3, default=(1., 0., 0.))
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--json-out', type=Path, required=True)
    args = p.parse_args()
    result = evaluate(args.robot, args.checkpoint, backend=args.backend, device=args.device,
        command=tuple(args.command), steps=args.steps, episodes_per_cell=args.episodes_per_cell, seed=args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Survival={result['survival']:.3f}, admission={result['admission']:.3f}; {args.json_out}")

if __name__ == '__main__':
    main()
