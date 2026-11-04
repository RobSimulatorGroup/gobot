"""Train a registered Gobot task, with torchrun for synchronous multi-GPU runs."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
from gobot.rl import tasks
from gobot.rl.supermimic.config import TrainConfig
from gobot.rl.motion_tracking.runner import run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',nargs='?',default='Gobot-Tracking-Flat-G1-SuperMimic-v0')
    p.add_argument('--list',action='store_true')
    p.add_argument('--task-module',action='append',default=[])
    p.add_argument('--config',type=Path,help='SuperMimic-Reproduced dataclass-tree JSON; rates are unscaled base rates')
    p.add_argument('--project',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--data',type=Path,help='directory containing converted 50 Hz NPZ clips')
    p.add_argument('--library',type=Path,help='compact packed library')
    p.add_argument('--clips',nargs='+')
    p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'))
    p.add_argument('--device')
    p.add_argument('--num-envs',type=int)
    p.add_argument('--seed',type=int)
    p.add_argument('--stop-iteration',type=int,help='absolute control-step stopping point; does not shorten the LR schedule')
    p.add_argument('--resume',type=Path)
    p.add_argument('--compile',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--capture-graphs',action=argparse.BooleanOptionalAction,default=None)
    a=p.parse_args(argv)
    for module in a.task_module:
        importlib.import_module(module)
    if a.list:
        for name in tasks.list_tasks():
            print(f'{name}\t{tasks.get_task(name).description}')
        return
    cfg = TrainConfig.from_dict(json.loads(a.config.read_text())) if a.config else TrainConfig()
    for key in ('project','backend','device','seed','stop_iteration','resume','capture_graphs'):
        if getattr(a,key) is not None:
            setattr(cfg,key,getattr(a,key))
    if a.output:
        cfg.log_root,cfg.run_name = a.output.parent,a.output.name
    if a.data: cfg.data.npz_dir=a.data
    if a.library: cfg.data.library=a.library
    if a.clips: cfg.data.clips=a.clips
    if a.num_envs is not None: cfg.env.num_envs=a.num_envs
    if cfg.backend=='mujoco-cpu' and a.device is None: cfg.device='cpu'
    if a.compile is not None:
        cfg.wm_trainer.compile=cfg.policy_trainer.compile='default' if a.compile else None
    print(json.dumps(run(cfg,a.task),indent=2))

if __name__=='__main__':
    main()
