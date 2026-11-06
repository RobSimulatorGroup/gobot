"""Replay a selected evaluation lane and render actual/reference side by side.

Replays its original shard (same batch, seed and protocol). Captures scene link
poses immediately after physics, before autoreset, then renders them offline.
Failure recordings end at termination; success recordings require every clip frame.
Rendering is offline, and neither mode presents autoreset as physical recovery.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
from .evaluate_bones import digest, load_plan, atomic_json


def replay_outcome(expected, outcome, *, terminated, done, steps):
    """Reject a truncated success or a replay that disagrees with its chosen label."""
    if outcome not in ('failure', 'success'):
        raise ValueError('unknown requested outcome')
    if expected['failed'] != (outcome == 'failure'):
        raise ValueError('selected clip does not have the requested benchmark outcome')
    if not 1 <= steps <= expected['frames'] or not done:
        raise ValueError('replay did not reach a valid episode boundary')
    if outcome == 'failure' and not terminated:
        raise ValueError('this replay did not reproduce failure')
    success = not terminated and steps == expected['frames']
    if outcome == 'success' and not success:
        raise ValueError('this replay did not complete the entire clip without termination')
    return {
        'replay_failure_step': steps if terminated else -1,
        'failure_reproduced': terminated,
        'same_failure_step': terminated and steps == expected['fail_step'],
        'requested_outcome': outcome,
        'success': success,
        'completed_steps': steps,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--clip',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--outcome',choices=('failure','success'),default='failure')
    a=p.parse_args()
    import numpy as np
    import torch
    import gobot
    from PIL import Image, ImageDraw, ImageFont
    from gobot.rl.supermimic.config import TrainConfig
    from gobot.rl.motion_tracking.env import MotionTrackingEnv
    from gobot.rl.motion_tracking.evaluate import load_policy
    from gobot.rl.locomotion.math import find_node_by_name
    from gobot.sim.providers.base import RobotBatchSpec
    plan,_=load_plan(a.evaluation)
    full=json.loads((a.evaluation/'full.json').read_text())
    expected=next(row for row in full['clips'] if row['name']==a.clip)
    if expected['failed'] != (a.outcome=='failure'):
        raise ValueError('selected clip does not have the requested benchmark outcome')
    if digest(plan['checkpoint'])!=plan['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    shard_index,part=next((i,part) for i,part in enumerate(plan['shards']) if any(r['name']==a.clip for r in part))
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed']+shard_index)
    policy,raw=load_policy(plan['checkpoint'],'cuda:0')
    policy.requires_grad_(False)
    cfg=TrainConfig.from_dict(raw)
    cfg.project=Path(plan['project']);cfg.data.npz_dir=Path(plan['data'])/'npz'
    cfg.data.library=cfg.data.manifest=cfg.data.limit=None
    cfg.data.clips=[r['name'] for r in part]
    for row in part:
        if digest(cfg.data.npz_dir/(row['name']+'.npz'))!=row['npz_sha256']:
            raise ValueError('motion hash changed')
    cfg.device='cuda:0';cfg.backend='mujoco-warp';cfg.deterministic=plan['deterministic']
    cfg.nconmax=plan['nconmax'];cfg.njmax=plan['njmax']
    cfg.env.num_envs=min(plan['max_envs'],len(part))
    cfg.env.replay_capacity=32;cfg.env.episode_length_s=1e5
    cfg.env.obs_noise=cfg.env.reset_noise=cfg.env.domain_randomization=False
    cfg.env.action_noise_std=0.;cfg.env.sampling_mode='start';cfg.env.compact_library=False
    # Exactly match the evaluator's reseeding after policy construction.
    torch.manual_seed(plan['seed']+shard_index)
    env=MotionTrackingEnv(cfg)
    camera=sensor=view=None
    writer=None
    try:
        provider=env.batch._provider
        provider._overflow_check_interval=1
        provider.assert_no_overflow()
        if env.batch.artifact.digest!=plan['physics_digest'] or env.fk.scene.digest!=plan['kinematics_digest']:
            raise ValueError('recording scene differs from evaluation')
        library=env.library
        clip_index=next(i for i,c in enumerate(library.clips) if c.name==a.clip)
        if clip_index>=env.num_envs:
            raise ValueError('recording currently requires the selected clip in the first queue wave')
        lane=clip_index
        current=env.ids.clone();nxt=current+env.num_envs
        obs,_=env.reset(frames=library.clip_start[current])
        env.forced_frame[:]=library.clip_start[nxt.clamp_max(library.num_clips-1)]
        names=tuple(env.fk.body_names)
        robot=find_node_by_name(env.batch.context.root,env.profile['robot_name'])
        nodes=tuple(find_node_by_name(robot,name) for name in names)
        if any(node is None for node in nodes):raise ValueError('render links differ')
        view=provider.create_robot_view(RobotBatchSpec(env.profile['robot_name'],env.profile['base_link'],env.joint_names,names))
        root_index=names.index('pelvis')
        states=[];references=[];trajectory=[];last_pose=None
        original=env.batch.step_control
        def capture_physics(*args,**kwargs):
            nonlocal last_pose
            out=original(*args,**kwargs)
            last_pose=view.read_state().link_pose[lane].detach().cpu().numpy().copy()
            return out
        env.batch.step_control=capture_physics
        terminated=False;done=False;total_reward=0.
        with torch.no_grad():
            for step in range(expected['frames']):
                target=int(env.frame[lane])
                qpos,qvel=library.ref_qpos_qvel(torch.tensor([target],device=env.device))
                ref=env.fk(qpos,qvel)
                ref_pose=torch.cat((ref['pos'][0],ref['quat'][0,:, [1,2,3,0]]),-1).cpu().numpy()
                obs,reward,term,timeout,info=env.step(policy(obs['policy'],obs['goal']),record=False)
                provider.assert_no_overflow();provider.assert_finite()
                terminated=bool(term[lane]);done=bool(term[lane]|timeout[lane])
                total_reward+=float(reward[lane])
                # One pose per 20ms; video samples every second pose (25 fps).
                if not np.isfinite(last_pose).all():raise ValueError('nonfinite link pose')
                states.append(last_pose.copy());references.append(ref_pose)
                trajectory.append({'step':step+1,'time':(step+1)*.02,'reference_frame':target,
                                   'terminated':terminated,'reward':float(reward[lane]),
                                   'base_position':last_pose[root_index,:3].tolist()})
                if done:break
        outcome = replay_outcome(expected, a.outcome, terminated=terminated, done=done, steps=len(states))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        poses=np.stack(states);refs=np.stack(references)
        np.savez_compressed(a.output.with_suffix('.poses.npz'),actual=poses,reference=refs,body_names=np.array(names))
        # Render after replay, so camera/scene presentation cannot affect simulation.
        camera=gobot.create_node('Camera3D','EvaluationReplayCamera')
        camera.set('up',(0.,0.,1.));camera.set('fov_y',45.)
        camera.set('z_near',.05);camera.set('z_far',200.)
        sensor=gobot.render.CameraSensor(camera,root=env.batch.context.root,width=640,height=560,outputs=('rgb',),device='cpu')
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        writer=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x640','-r','25','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
        font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
        small=ImageFont.truetype(str(font_path),16) if font_path.exists() else font
        indices=list(range(0,len(states),2))
        if indices[-1]!=len(states)-1:indices.append(len(states)-1)
        images=[]
        for k,index in enumerate(indices):
            canvas=Image.new('RGB',(1280,640),(16,23,34))
            for col,pose in enumerate((poses[index],refs[index])):
                env.batch.context.apply_link_poses(nodes,pose)
                base=pose[root_index,:3].copy()
                target=base.copy();target[2]=.65
                camera.set('eye',tuple(target+np.array([1.9,-2.4,1.05])))
                camera.set('target',tuple(target))
                frame=sensor.capture()
                image=Image.fromarray(frame['rgb'].numpy().copy())
                del frame
                canvas.paste(image,(col*640,80))
            draw=ImageDraw.Draw(canvas)
            draw.text((16,9),'GOBOT POLICY  |  physics replay',font=font,fill=(245,245,245))
            draw.text((656,9),'REFERENCE MOTION  |  target pose',font=font,fill=(120,200,255))
            draw.text((16,42),a.clip,font=small,fill=(215,225,235))
            draw.text((900,43),f't = {(index+1)*.02:.2f}s   step {index+1}',font=small,fill='white')
            if index==len(states)-1:
                color=(147,30,35) if terminated else (24,112,70)
                label=(f'FAILURE at step {len(states)}  |  replay ends before automatic reset' if terminated
                       else f'SUCCESS  |  completed all {len(states)} frames without termination')
                draw.rectangle((0,596,1280,640),fill=color)
                draw.text((16,605),label,font=font,fill='white')
            writer.stdin.write(np.asarray(canvas).tobytes())
            if k in (0,len(indices)//2,len(indices)-1):
                canvas.save(a.output.with_name(a.output.stem+f'_frame{k:04d}.png'))
                images.append(canvas.copy())
            if index==len(states)-1:
                # Labeled terminal freeze for readability, not additional physics.
                for _ in range(25):writer.stdin.write(np.asarray(canvas).tobytes())
        writer.stdin.close()
        if writer.wait()!=0:raise RuntimeError('video encoder failed')
        writer=None
        strip=Image.new('RGB',(1280,640*len(images)))
        for i,img in enumerate(images):strip.paste(img,(0,640*i))
        strip.save(a.output.with_suffix('.contact.png'))
        report={'checkpoint_sha256':plan['checkpoint_sha256'],'clip':a.clip,'shard':shard_index,
                'num_envs':env.num_envs,'original_benchmark':expected,
                **outcome, 'replay_norm_return':total_reward/(.1*len(states)),
                'frames':len(indices)+25,'fps':25,'terminal_freeze_seconds':1,
                'rendering':'Gobot CameraSensor; actual post-physics pre-reset poses vs reference FK; follow cameras',
                'trajectory':trajectory}
        atomic_json(a.output.with_suffix('.json'),report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('trajectory','original_benchmark')},indent=2))
    finally:
        if writer is not None:
            if writer.stdin and not writer.stdin.closed:writer.stdin.close()
            writer.wait()
        sensor=view=camera=None
        gc.collect()
        env.close()
        gobot.render._shutdown_headless_render_context()


if __name__=='__main__':main()
