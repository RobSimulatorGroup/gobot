"""Run the pinned mjlab G1 task and its PPO runner for the Gobot comparison."""
import argparse
from dataclasses import asdict
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

REFERENCE = 'dcdf6ce519abfd2ea45e9503ae6fb650b75bb869'
VERSIONS = {'torch': '2.11.0', 'mujoco': '3.17.0', 'mujoco-warp': '3.17.0',
            'warp-lang': '1.17.0', 'rsl-rl-lib': '5.5.2'}


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


class SolverStatistics:
    """Replace per-kernel prints with global per-update solve-call fractions."""

    def __init__(self, simulation):
        import torch
        import warp as wp
        import mujoco_warp as mjw
        self.bits = (int(mjw.OverflowType.ITERATIONS), int(mjw.OverflowType.LS_ITERATIONS))
        self.mask = self.bits[0] | self.bits[1]
        self.flags = wp.to_torch(simulation.wp_data.overflow)
        self.counts = torch.zeros(3, dtype=torch.float64, device=self.flags.device)
        for name in ('step', 'forward'):
            setattr(simulation, name, self.wrap(getattr(simulation, name)))

    def wrap(self, operation):
        def run(*args, **kwargs):
            # These flags are diagnostics only. Preserve every capacity-error bit.
            self.flags.bitwise_and_(~self.mask)
            result = operation(*args, **kwargs)
            for i, bit in enumerate(self.bits):
                self.counts[i].add_(((self.flags & bit) != 0).sum())
            self.counts[2].add_(self.flags.numel())
            return result
        return run

    def attach(self, runner):
        import torch
        log = runner.logger.log
        self.counts.zero_()

        def report(*args, **kwargs):
            counts = self.counts.clone()
            self.counts.zero_()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts)
            if runner.logger.writer is not None:
                rates = counts[:2] / counts[2].clamp(min=1)
                runner.logger.ep_extras.append({
                    'Solver/iterations_limit_fraction': rates[0],
                    'Solver/linesearch_limit_fraction': rates[1],
                })
            return log(*args, **kwargs)
        runner.logger.log = report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terrain-reference', type=Path, required=True, help='Authored terrain exported by export_training_terrain.py')
    parser.add_argument('--source', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('logs/g1_comparison_seed42'))
    parser.add_argument('--num-envs', type=positive, default=1024, help='environments per GPU')
    parser.add_argument('--iterations', type=positive, default=18000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ang-vel-axes', choices=('xyz', 'z'), default='xyz')
    parser.add_argument('--ang-vel-std', type=float, default=math.sqrt(.5))
    parser.add_argument('--save-interval', type=positive, default=500)
    parser.add_argument('--resume', type=Path, help='Continue this comparison checkpoint to --iterations total updates')
    parser.add_argument('--check', action='store_true', help='check dependencies and configuration without training')
    args = parser.parse_args()
    from mjlab_training import helper, training_terrain, set_initial_terrain, TrainingLog, configure_angular_tracking
    terrain_module = helper('terrain_reference')
    terrain_data = terrain_module.load(args.terrain_reference)
    source = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REFERENCE:
        raise RuntimeError(f'mjlab revision must be {REFERENCE}; found {revision}')
    if subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only', 'HEAD', '--', 'src', 'pyproject.toml'], text=True).strip():
        raise RuntimeError('mjlab training sources have modifications; preserve them and use a clean reference checkout')
    versions = {name: metadata.version(name) for name in VERSIONS}
    for name, expected in VERSIONS.items():
        if versions[name].split('+')[0] != expected:
            raise RuntimeError(f'{name} must be {expected}; found {versions[name]}')
    import torch
    import rsl_rl.runners  # Load the numerical stack before optional task dependencies.
    if torch.version.cuda != '12.8':
        raise RuntimeError(f'Torch CUDA build must be 12.8; found {torch.version.cuda}')
    sys.path.insert(0, str(source / 'src'))
    import mjlab
    # Load only the reference velocity task; no unrelated task autodiscovery.
    tasks = types.ModuleType('mjlab.tasks')
    tasks.__path__ = [str(source / 'src/mjlab/tasks')]
    sys.modules.setdefault('mjlab.tasks', tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
    from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from mujoco_warp._src import collision_convex

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    cfg = unitree_g1_rough_env_cfg()
    angular_tracking = configure_angular_tracking(cfg, args.ang_vel_axes, args.ang_vel_std)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed + rank
    import mujoco_warp as mjw
    class LoggedSimulationCfg(type(cfg.sim)):
        def apply_wp_opt(self, opt):
            super().apply_wp_opt(opt)
            # Set before the first CUDA graph capture: this flag is kernel-static.
            opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    cfg.sim = LoggedSimulationCfg(**{key: value for key, value in vars(cfg.sim).items()})
    cfg.sim.njmax = 1500
    cfg.sim.nconmax = 70
    collision_convex.MJ_MAX_EPAHORIZON = 64
    agent = unitree_g1_ppo_runner_cfg()
    agent.seed = args.seed + rank
    agent.max_iterations = args.iterations
    agent.save_interval = args.save_interval
    agent.logger = 'tensorboard'
    agent.upload_model = False
    agent.experiment_name = 'g1_comparison'
    settings = {'reference_revision': revision, 'versions': versions,
                'seed': args.seed, 'world_size': world_size, 'num_envs_per_gpu': args.num_envs,
                'total_num_envs': args.num_envs * world_size, 'iterations': args.iterations,
                'num_steps_per_env': agent.num_steps_per_env,
                'total_transitions': args.iterations * agent.num_steps_per_env * args.num_envs * world_size,
                'capacities': {'njmax': 1500, 'nconmax': 70, 'epa_horizon': 64},
                'solver_warnings': 'global_fraction_per_update',
                'statistics': 'all_workers_completed_episodes', 'contact_timing': 'physics_dt',
                'angular_tracking': angular_tracking,
                'terrain_sha256': terrain_data['sha256'],
                'allow_tf32': False, 'output': str(args.output.resolve())}
    if rank == 0:
        print(json.dumps(settings, indent=2), flush=True)
    if args.check:
        return
    output = args.output.resolve()
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        previous = json.loads((args.resume.parent / 'comparison.json').read_text())
        for key in ('reference_revision', 'versions', 'seed', 'world_size', 'num_envs_per_gpu',
                    'num_steps_per_env', 'capacities', 'allow_tf32', 'terrain_sha256', 'contact_timing', 'angular_tracking'):
            if previous.get(key) != settings[key]:
                raise ValueError(f'resume changes {key}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if int(checkpoint['iter']) + 1 >= args.iterations:
            raise ValueError('--iterations must exceed the checkpoint completed updates')
        del checkpoint
    if rank == 0:
        if output.exists() and any(output.iterdir()) and args.resume is None:
            raise RuntimeError(f'output is not empty: {output}; choose a new output directory')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'comparison.json').write_text(json.dumps(settings, indent=2) + '\n')
        (output / 'training_terrain.json').write_text(json.dumps(terrain_data, indent=2) + '\n')
        dump_yaml(output / 'params/env.yaml', asdict(cfg))
        dump_yaml(output / 'params/agent.yaml', asdict(agent))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.cuda.set_device(local_rank)
    configure_torch_backends(allow_tf32=False)
    with training_terrain(cfg, terrain_data):
        env = ManagerBasedRlEnv(cfg, device=f'cuda:{local_rank}')
    try:
        solver_statistics = SolverStatistics(env.sim)
        initial_digest = set_initial_terrain(env, terrain_data, args.seed + rank)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = VelocityOnPolicyRunner(wrapped, asdict(agent), str(output), f'cuda:{local_rank}')
        initial_digests = [initial_digest]
        if torch.distributed.is_initialized():
            initial_digests = [None] * world_size
            torch.distributed.all_gather_object(initial_digests, initial_digest)
        if rank == 0:
            (output / 'terrain.json').write_text(json.dumps({'sha256': terrain_data['sha256'],
                'seed': terrain_data['seed'], 'initial_assignments': initial_digests}, indent=2) + '\n')
            # Store the resolved terrain/config after applying the shared generation input.
            dump_yaml(output / 'params/env.yaml', asdict(cfg))
        TrainingLog(env, runner, agent, output)
        solver_statistics.attach(runner)
        if args.resume is not None:
            runner.load(str(args.resume))
            runner.current_learning_iteration += 1
        remaining = args.iterations - runner.current_learning_iteration
        if remaining <= 0:
            raise ValueError('--iterations must exceed the checkpoint completed updates')
        runner.add_git_repo_to_log(str(source / 'src/mjlab/scripts/train.py'))
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    finally:
        env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
