"""Real optimizer/checkpoint round trips for PPO and conditional AMP (small batches)."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import tempfile
import torch
from examples.locomotion.experiment import Experiment
from examples.locomotion.train import run
from examples.locomotion.assessment import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'),default='mujoco-cpu');a=p.parse_args()
    device='cuda:0' if a.backend=='mujoco-warp' else 'cpu'
    with tempfile.TemporaryDirectory(prefix='gobot-training-test-') as directory:
        root=Path(directory)
        for robot,recipe in [('go1','natural'),('g1','baseline'),('g1','natural')]:
            exp=Experiment(robot=robot,recipe=recipe,stage='flat_gait' if robot=='g1' and recipe=='natural' else 'flat_control',
                           iterations=2,num_envs=4,hours=.1,eval_steps=20,eval_envs=2)
            if recipe=='natural' and robot=='g1' and a.backend=='mujoco-warp':exp=replace(exp,reference_reset_probability=.5)
            out=root/f'{robot}_{recipe}'
            run(exp,out,backend=a.backend,device=device,no_eval=True,require_gate=False)
            original=torch.load(out/'model_final.pt',map_location='cpu',weights_only=False)
            run(replace(exp,iterations=1),out/'resumed',backend=a.backend,device=device,resume=out/'model_final.pt',no_eval=True,require_gate=False)
            restored=torch.load(out/'resumed/model_final.pt',map_location='cpu',weights_only=False)
            assert restored['iter']==original['iter']+1
            if robot=='g1' and recipe=='natural':
                assert restored['motion_prior']['update_count']==3
                assert restored['motion_prior']['replay_size']>0
                assert torch.isfinite(restored['motion_prior']['mean']).all()
            result=evaluate(out/'model_final.pt',out/'eval.json',steps=20,count=2,device=device,backend=a.backend)
            assert not result['passed'] # Untrained policies cannot be admitted by a smoke test.
            print(robot,recipe,'optimizer, resume and independent evaluation passed',flush=True)

if __name__=='__main__':main()
