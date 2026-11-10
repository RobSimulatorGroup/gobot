"""Shared terrain and statistics for the pinned mjlab comparison runner.

Loads only standalone data/statistics helpers, without importing the Gobot engine.
"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def helper(name):
    path = Path(__file__).resolve().parents[2] / 'python/gobot/rl/locomotion' / f'{name}.py'
    module_name = f'comparison_{name}'
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def training_terrain(cfg, data):
    """Keep mjlab's terrain entity/curriculum, using the compiled shared geometry."""
    from mjlab.terrains.terrain_generator import SubTerrainCfg
    terrain = helper('terrain_reference')
    gen = cfg.scene.terrain.terrain_generator
    gen.seed = data['seed']
    gen.num_rows, gen.num_cols = data['grid_shape']
    gen.size = tuple(data['size'])
    gen.border_width = data['border_width']
    gen.curriculum = True
    # Only names/proportions are used by the entity and curriculum with this generator.
    class ReferencedSubTerrain(SubTerrainCfg):
        def function(self, difficulty, spec, rng):
            raise RuntimeError('shared terrain must use its exported geometry')

    gen.sub_terrains = {name: ReferencedSubTerrain(proportion=weight, size=gen.size)
                       for name, weight in zip(data['names'], data['proportions'], strict=True)}
    cfg.scene.terrain.max_init_terrain_level = data['max_init_level']

    class SharedTerrainGenerator:
        def __init__(self, config, device):
            self.terrain_origins = np.asarray(data['origins'], dtype=np.float32)
            self.flat_patches = {}
            self.flat_patch_radii = {}

        def compile(self, spec):
            terrain.add_to_spec(spec, data)

    with patch('mjlab.terrains.terrain_entity.TerrainGenerator', SharedTerrainGenerator):
        yield


def set_initial_terrain(env, data, seed):
    module = helper('terrain_reference')
    levels, types = module.initial_assignments(data, env.num_envs, seed)
    terrain = env.scene.terrain
    terrain.terrain_levels.copy_(torch.as_tensor(levels, device=env.device))
    terrain.terrain_types.copy_(torch.as_tensor(types, device=env.device))
    terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
    return module.assignment_digest(levels, types)


class TrainingLog:
    """Use the same completed-episode reductions/formatter as Gobot on every rank."""

    def __init__(self, env, runner, agent, directory):
        self.env, self.runner, self.agent = env, runner, agent
        self.directory = Path(directory)
        self.shared = helper('training_log')
        self.distributed = helper('distributed').DistributedTraining(
            str(env.device), runner.gpu_global_rank, runner.gpu_local_rank, runner.gpu_world_size)
        rewards = env.reward_manager
        self.reward_names = tuple(rewards.active_terms)
        self.cfg = SimpleNamespace(
            physics_dt=env.physics_dt, decimation=env.cfg.decimation,
            episode_length_s=env.max_episode_length_s, action_clip=agent.clip_actions,
            rewards=SimpleNamespace(**{name: rewards.get_term_cfg(name).weight for name in self.reward_names}))
        self.statistics = self.shared.TrainingStatistics(env.num_envs, env.action_manager.total_action_dim,
                                                        self.reward_names, self.cfg, env.device)
        self.feet = env.scene['robot'].find_sites(('left_foot','right_foot'))[0]
        self.pending = None
        self.collection_start = None
        self.collection_end = None
        self.update_end = None
        self.started = time.monotonic()
        self.iteration_seconds = 0.
        self.completed = 0
        self._attach()

    def clock(self):
        if str(self.env.device).startswith('cuda'):
            torch.cuda.synchronize(self.env.device)
        return time.monotonic()

    def _attach(self):
        env = self.env
        compute = env.reward_manager.compute

        def reward(dt):
            data = env.scene['robot'].data
            feet = env.scene['feet_ground_contact']
            swing = env.reward_manager.get_term_cfg('foot_swing_height').func
            height = env.scene['foot_height_scan'].data.heights
            peak = torch.where(feet.data.found == 0, torch.maximum(swing.peak_heights, height), swing.peak_heights)
            metrics = self.shared.step_statistics(torch,
                momentum=env.scene['robot/root_angmom'].data,
                foot_velocity=data.site_lin_vel_w[:, self.feet], contact=feet.data.found,
                first_contact=feet.compute_first_contact(dt), peak_height=peak, foot_force=feet.data.force,
                linear_velocity=data.root_link_lin_vel_b, angular_velocity=data.root_link_ang_vel_b,
                command=env.command_manager.get_command('twist'))
            result = compute(dt)
            reasons = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            terms = env.termination_manager._term_dones
            reasons.masked_fill_(terms['time_out'], 2)
            reasons.masked_fill_(terms['out_of_terrain_bounds'], 3)
            reasons.masked_fill_(env.reset_terminated, 1)
            self.pending = {'metrics': metrics, 'reward_terms': env.reward_manager._step_reward.clone(),
                            'reset_reason': reasons}
            self.action = env.action_manager.action.clone()
            return result
        env.reward_manager.compute = reward
        step = env.step

        def record(action):
            result = step(action)
            _, rewards, terminated, truncated, _ = result
            self.statistics.record(self.action, rewards, terminated | truncated, self.pending)
            return result
        env.step = record
        act = self.runner.alg.act

        def timed_act(obs):
            if self.collection_start is None:
                self.collection_start = self.clock()
            return act(obs)
        self.runner.alg.act = timed_act
        returns = self.runner.alg.compute_returns

        def timed_returns(obs):
            self.collection_end = self.clock()
            return returns(obs)
        self.runner.alg.compute_returns = timed_returns
        update = self.runner.alg.update

        def timed_update():
            result = update()
            self.update_end = self.clock()
            return result
        self.runner.alg.update = timed_update
        # Avoid the original rank-0 rolling window and repeated extras accumulation.
        self.runner.logger.process_env_step = lambda *args, **kwargs: None
        self.runner.logger.log = self.report

    def report(self, *, it, start_it, total_it, collect_time, learn_time, loss_dict,
               learning_rate, action_std, rnd_weight, **kwargs):
        terrain = self.env.scene.terrain
        facade = SimpleNamespace(
            terrain_assignments=(terrain.terrain_levels, terrain.terrain_types),
            terrain_type_names=tuple(terrain.cfg.terrain_generator.sub_terrains),
            cfg_obj=SimpleNamespace(command=self.env.command_manager.get_term('twist').cfg))
        summary = self.statistics.summarize(self.distributed, facade)
        losses = self.distributed.reduce([float(v) for v in loss_dict.values()] + [float(action_std.mean())])
        duration, collection, learning = self.distributed.reduce(
            [self.update_end - self.collection_start, self.collection_end - self.collection_start,
             self.update_end - self.collection_end], maximum=True)
        self.iteration_seconds += duration
        self.completed += 1
        size = self.env.num_envs * self.distributed.world_size * self.agent.num_steps_per_env
        metrics = {'iteration': it, 'total_steps': (it + 1) * size, 'steps_per_second': size / duration,
                   'collection_s': collection, 'learning_s': learning, 'iteration_s': duration,
                   'elapsed_s': time.monotonic() - self.started, 'world_size': self.distributed.world_size,
                   'total_num_envs': self.env.num_envs * self.distributed.world_size,
                   'action_std': losses[-1] / self.distributed.world_size, 'learning_rate': learning_rate,
                   'action_clip_fraction': None, 'joint_effort_saturation_fraction': None,
                   **summary,
                   **{k: v / self.distributed.world_size for k, v in zip(loss_dict, losses[:-1], strict=True)}}
        logger = self.runner.logger
        if self.distributed.main:
            # SolverStatistics supplies these globally reduced diagnostics before calling us.
            for values in logger.ep_extras:
                metrics['log'].update({k: float(v) for k,v in values.items() if k.startswith('Solver/')})
            with (self.directory / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + '\n')
            print(self.shared.format_training_log(metrics, target_iteration=total_it,
                  remaining_iterations=total_it-it-1, average_iteration_s=self.iteration_seconds/self.completed), flush=True)
            if logger.writer is not None:
                for key, value in metrics['log'].items():
                    if value is not None:
                        logger.writer.add_scalar(key, value, it)
                for key, value in metrics.items():
                    if isinstance(value, (float, int)):
                        logger.writer.add_scalar('Training/' + key, value, it)
        logger.ep_extras.clear()
        logger.tot_timesteps = metrics['total_steps']
        logger.tot_time += duration
        self.statistics.begin_update()
        self.collection_start = None
