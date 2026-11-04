"""Evaluate the pinned public actor using the untouched reference benchmark."""
import argparse
import json
from pathlib import Path
import sys
import types


def main():
    import torch
    import mjlab
    from .check_release_policy import load_release
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--nconmax',type=int,default=35)
    p.add_argument('--njmax',type=int,default=250)
    a=p.parse_args()
    ck,sha=load_release(a.checkpoint)
    package=types.ModuleType('mjlab.tasks');package.__path__=[str(Path(mjlab.__file__).parent/'tasks')]
    sys.modules['mjlab.tasks']=package
    from supermimic.policy.nets import Policy,PolicyCfg
    from supermimic.eval import benchmark
    if (a.nconmax,a.njmax)!=(35,250):
        from supermimic.envs.tracking_env_cfg import make_env_cfg
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.utils.torch import configure_torch_backends
        def make_diagnostic_env(args,device):
            configure_torch_backends()
            cfg=make_env_cfg(args)
            cfg.sim.nconmax=a.nconmax;cfg.sim.njmax=a.njmax
            return ManagerBasedRlEnv(cfg,device=device)
        benchmark.make_env=make_diagnostic_env
    run_benchmark,save_results=benchmark.run_benchmark,benchmark.save_results
    policy=Policy(PolicyCfg(**ck['config']['policy'])).to('cuda:0').eval()
    policy.load_state_dict(ck['policy'])
    torch.set_num_threads(2)
    results=run_benchmark(policy,[str(p) for p in a.data],num_envs=a.num_envs)
    save_results(results,a.output)
    import importlib.metadata as metadata
    info={'capacities':{'nconmax':a.nconmax,'njmax':a.njmax},'checkpoint_sha256':sha,'gobot_imported':'gobot' in sys.modules,
          'software':{p:metadata.version(p) for p in ('torch','mujoco','mujoco-warp','warp-lang')}}
    a.output.with_suffix('.provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    print(a.output.read_text())

if __name__=='__main__':main()
