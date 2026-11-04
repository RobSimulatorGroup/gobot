"""Evaluate a trained or pinned public actor on complete motion clips."""
import argparse
import json
from pathlib import Path
from gobot.rl.supermimic.config import TrainConfig
from gobot.rl.motion_tracking.evaluate import load_policy, benchmark
from gobot.rl.motion_tracking.convert import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clips',nargs='+')
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'),default='mujoco-warp')
    p.add_argument('--device')
    p.add_argument('--release',action='store_true')
    p.add_argument('--nconmax',type=int,default=35)
    p.add_argument('--njmax',type=int,default=250)
    a=p.parse_args()
    device=a.device or ('cpu' if a.backend=='mujoco-cpu' else 'cuda:0')
    pi,raw=load_policy(a.checkpoint,device,release=a.release)
    cfg=TrainConfig.from_dict(raw)
    cfg.project=a.project;cfg.data.npz_dir=a.data;cfg.data.library=None;cfg.data.clips=a.clips;cfg.data.manifest=None;cfg.data.limit=None
    cfg.device=device;cfg.backend=a.backend;cfg.env.num_envs=a.num_envs;cfg.env.replay_capacity=32
    cfg.nconmax=a.nconmax;cfg.njmax=a.njmax
    result=benchmark(cfg,pi)
    result['checkpoint_sha256']=digest(a.checkpoint)
    result['training_result']=not a.release
    result['capacities']={'nconmax':cfg.nconmax,'njmax':cfg.njmax}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
