"""Measure held-out world-model, critic and physical-rollout adaptation errors."""
import argparse
import json
from pathlib import Path
import torch

from gobot.rl.motion_tracking import MotionTrackingCfg,MotionRandomizationCfg,make_motion_tracking_env
from gobot.rl.motion_tracking.validation import validate_adaptation
from examples.locomotion.budget import add_budget_arguments,evaluation_budget
from examples.locomotion.supermimic_reference import write_json
from .tracking_experiment import TrackingExperiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--thresholds',type=Path)
    p.add_argument('--num-envs',type=int,default=64);p.add_argument('--steps',type=int,default=128)
    p.add_argument('--seed',type=int,default=100071)
    p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'),default='mujoco-warp')
    add_budget_arguments(p);a=p.parse_args()
    if a.steps<32 or a.num_envs<1:p.error('validation requires positive environments and at least 32 steps')
    recipe=TrackingExperiment.load(a.config)
    project=Path(__file__).resolve().parent
    manifest=json.loads((project/'assets/motions/tracking/manifest.json').read_text())
    split='validation' if recipe.split=='train' else 'coverage40'
    files=tuple(str(project/'assets/motions/tracking'/r['file']) for r in manifest['clips'] if split=='coverage40' or r['split']==split)
    cfg=MotionTrackingCfg(project_path=project,motion_files=files,split=split,eval_mode=False,reset_noise=True,
        gamma=recipe.learner_config().gamma,goal_frames=recipe.learner_config().goal_frames,
        deterministic=recipe.deterministic,randomization=MotionRandomizationCfg(**recipe.randomization))
    thresholds=json.loads(a.thresholds.read_text()) if a.thresholds else None
    torch.set_num_threads(4)
    from contextlib import nullcontext
    with evaluation_budget(a,stage='tracking_adaptation_validation') if a.backend=='mujoco-warp' else nullcontext():
        env=make_motion_tracking_env(cfg,num_envs=a.num_envs,backend=a.backend,seed=a.seed)
        try:
            result=validate_adaptation(env,a.checkpoint,a.baseline,thresholds=thresholds,steps=a.steps,seed=a.seed)
            a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result)
            print(json.dumps(result,indent=2))
        finally:env.close()


if __name__=='__main__':main()
