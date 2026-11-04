"""Run an independent reference/Gobot task trace with the same released actor."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import types


def run(mode, project, data, checkpoint, output, steps, device, compact=False, episode_length=10.):
    import torch
    from .check_release_policy import load_release
    torch.set_num_threads(2)
    ck,sha=load_release(checkpoint)
    if mode=='reference':
        import mjlab
        package=types.ModuleType('mjlab.tasks');package.__path__=[str(Path(mjlab.__file__).parent/'tasks')]
        sys.modules['mjlab.tasks']=package
        from supermimic.envs.tracking_env_cfg import make_env,SuperMimicEnvCfgArgs
        from supermimic.envs.state_io import StateIO
        from supermimic.policy.nets import Policy,PolicyCfg
        env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(data)],num_envs=2,episode_length_s=episode_length,
            action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
            sampling_mode='start',replay_capacity=64,compact_library=compact),device=device)
        command=env.command_manager.get_term('motion')
        command.forced_frame[:]=0
        state_io=StateIO(env)
        read=state_io.read
        get_frame=lambda:command.frame
        buffer=env.recorder_manager.get_term('replay').buffer
        sampler=command
    else:
        from gobot.rl.supermimic.config import TrainConfig
        from gobot.rl.supermimic.policy import Policy,PolicyCfg
        from gobot.rl.motion_tracking.env import MotionTrackingEnv
        cfg=TrainConfig()
        cfg.project=project;cfg.device=device;cfg.backend='mujoco-warp' if device.startswith('cuda') else 'mujoco-cpu'
        cfg.env.num_envs=2;cfg.env.episode_length_s=episode_length;cfg.data.npz_dir=data.parent;cfg.data.clips=[data.stem];cfg.env.replay_capacity=64
        cfg.env.action_noise_std=0.;cfg.env.obs_noise=cfg.env.reset_noise=cfg.env.domain_randomization=False
        cfg.env.sampling_mode='start';cfg.env.compact_library=compact
        # Same backend precision configuration as mjlab.
        torch.backends.cuda.matmul.fp32_precision='tf32';torch.backends.cudnn.fp32_precision='tf32'
        env=MotionTrackingEnv(cfg)
        env.forced_frame[:]=0
        read=env.learning_state;get_frame=lambda:env.frame
        buffer=env.buffer;sampler=env.sampler
    try:
        policy=Policy(PolicyCfg(**ck['config']['policy'])).to(device).eval()
        policy.load_state_dict(ck['policy'])
        rows=[]
        with torch.no_grad():
            obs,_=env.reset()
            for i in range(steps):
                action=policy(obs['policy'],obs['goal'])
                row={'state':read().clone(),'frame':get_frame().clone(),'observation':obs['policy'].clone(),
                     'goal':obs['goal'].clone(),'action':action.clone()}
                obs,reward,terminated,timeout,info=env.step(action)
                row.update(reward=reward.clone(),terminated=terminated.clone(),timeout=timeout.clone())
                rows.append({k:v.cpu() for k,v in row.items()})
        result={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        result['replay']={key:getattr(buffer,key).cpu().clone() for key in
            ('s','a','a_exec','xi','r','done','frame','clip','ep_id','terminal','ep_start','valid','ptr','ep_counter')}
        result['sampler']={key:getattr(sampler,key).cpu().clone() for key in ('bin_return','bin_memory','ep_len')}
        torch.save(result,output)
        import importlib.metadata as metadata
        info={'mode':mode,'compact':compact,'episode_length_s':episode_length,'steps':steps,'checkpoint_sha256':sha,'gobot_imported':'gobot' in sys.modules,
              'software':{p:metadata.version(p) for p in ('torch','mujoco','mujoco-warp','warp-lang')}}
        Path(output).with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        return info
    finally:
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('reference','gobot'),required=True)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compact',action='store_true')
    p.add_argument('--episode-length',type=float,default=10.)
    p.add_argument('--steps',type=int,default=64);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(run(a.mode,a.project,a.data,a.checkpoint,a.output,a.steps,a.device,a.compact,a.episode_length),indent=2))
