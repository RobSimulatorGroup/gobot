"""Train Go1 or G1 from Gobot scenes with a shared PPO workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from gobot.rl import RslRlVecEnvWrapper
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.ppo import rsl_rl_train_cfg
from gobot.rl.locomotion.manifest import build_policy_manifest, validate_policy
from gobot.rl.locomotion.logger import _install_cpu_logger_bookkeeping
from gobot.rl.policy import POLICY_MANIFEST_KEY, policy_manifest_from_checkpoint, write_policy_manifest_sidecar
from .common import task_cfg

class VelocityRunner(OnPolicyRunner):
    def __init__(self, *args, manifest, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest = manifest
    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos[POLICY_MANIFEST_KEY] = self.manifest.metadata()
        infos['env_state'] = self.env.training_state_dict()
        super().save(path, infos=infos)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=('go1', 'g1'), required=True)
    parser.add_argument('--backend', choices=('mujoco-cpu', 'mujoco-warp'), default='mujoco-warp')
    parser.add_argument('--device')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--sim-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-step-extras', action='store_true')
    args = parser.parse_args(argv)
    device = args.device or ('cuda:0' if args.backend == 'mujoco-warp' else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = task_cfg(args.robot)
    log_dir = args.log_dir or Path(cfg.project_path) / 'logs' / cfg.name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if args.resume and checkpoint is None:
        checkpoint = log_dir / 'model_final.pt'
    extra = {'sim_workers': args.sim_workers} if args.backend == 'mujoco-cpu' else {}
    env = make_velocity_env(cfg, backend=args.backend, device=device, num_envs=args.num_envs,
                            seed=args.seed, collect_step_extras=not args.no_step_extras, **extra)
    wrapper = RslRlVecEnvWrapper(env, device=device)
    try:
        train_cfg = rsl_rl_train_cfg(experiment_name=cfg.name, max_iterations=args.iterations, save_interval=args.save_interval)
        train_cfg['seed'] = args.seed
        manifest = build_policy_manifest(wrapper, cfg, train_cfg, Path(cfg.project_path))
        runner = VelocityRunner(wrapper, train_cfg, log_dir=str(log_dir), device=device, manifest=manifest)
        _install_cpu_logger_bookkeeping(runner.logger)
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            saved_manifest = policy_manifest_from_checkpoint(payload)
            if saved_manifest is None:
                raise ValueError('checkpoint has no policy manifest')
            validate_policy(saved_manifest, env, cfg)
            infos = runner.load(str(checkpoint), map_location=device)
            if not infos or 'env_state' not in infos:
                raise ValueError('checkpoint has no curriculum/RNG state')
            print('Restored training state:', wrapper.load_training_state_dict(infos['env_state']))
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        output = log_dir / 'model_final.pt'
        runner.save(str(output))
        write_policy_manifest_sidecar(output, manifest)
        print(f'Saved {output}')
    finally:
        wrapper.close()

if __name__ == '__main__':
    main()
