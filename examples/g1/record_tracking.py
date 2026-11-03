"""Fixed tracking preview windows rendered by Gobot, with terminal frames retained."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import torch
import numpy as np
from PIL import Image, ImageDraw
import gobot
from gobot.rl.motion_tracking import MotionTrackingCfg,make_motion_tracking_env
from gobot.rl.motion_tracking.evaluate import load_policy
from gobot.rl.supermimic.geometry import unflatten_state
from gobot.rl.experiment_identity import file_digest
from examples.locomotion.budget import add_budget_arguments,evaluation_budget

WINDOWS={'walk1_subject1':4.,'run1_subject2':32.,'sprint1_subject2':8.,'dance1_subject2':8.}


def record(checkpoint,output,clip,start,seconds):
    project=Path(__file__).resolve().parent
    cfg=MotionTrackingCfg(project_path=project,motion_files=(str(project/f'assets/motions/tracking/{clip}.csv'),),autoreset=False)
    env=make_motion_tracking_env(cfg,num_envs=1,backend='mujoco-warp')
    sensor=camera=writer=None
    try:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        policy=load_policy(env,payload);mdp=env.supermimic_mdp
        env.reset_to_frames(torch.tensor([round(start/env.step_dt)],device=env.device))
        env.batch.sync_scene()
        root=env.batch.context.root
        ground=gobot.create_box_visual('preview_ground',(100.,100.,.02),(0.,0.,-.015))
        ground.surface_color=(.28,.32,.37,1.)
        root.add_child(ground)
        # Visual-only meter lines; the compiled collision plane is unchanged.
        for i in range(-15,16):
            for axis in range(2):
                size=(.012,40.,.002) if axis==0 else (40.,.012,.002)
                position=(float(i),0.,-.003) if axis==0 else (0.,float(i),-.003)
                line=gobot.create_box_visual(f'preview_grid_{axis}_{i}',size,position)
                line.surface_color=(.42,.46,.50,1.);root.add_child(line)
        camera=gobot.create_node('Camera3D','TrackingCamera')
        camera.set('up',(0.,0.,1.));camera.set('fov_y',45.)
        camera.set('z_near',.05);camera.set('z_far',160.)
        sensor=gobot.render.CameraSensor(camera,root=root,width=960,height=720,outputs=('rgb',),device='cpu')
        output.parent.mkdir(parents=True,exist_ok=True)
        writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','960x720','-r','25','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20',
            '-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
        history=None;trajectory=[];images=0
        with torch.inference_mode():
            for step in range(round(seconds/env.step_dt)):
                state=env.state.info['supermimic_state']
                obs=mdp.obs(unflatten_state(state['s']),state['a_prev'],state['xi'],state['f'])
                history=obs[:,None].repeat(1,int(payload['args']['hist']),1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action=policy(history,mdp.goal(state['f'],int(payload['args']['goal_frames'])))
                result=env.step(action)
                failed=bool(result.terminated[0]);ended=bool(result.truncated[0])
                physical=env.batch.read_state();position=physical.base_pose[0,:3].cpu().numpy()
                t=(step+1)*env.step_dt
                trajectory.append({'elapsed_s':t,'source_time_s':start+t,'failed':failed,'clip_ended':ended,
                    'root_position':position.tolist(),'reward':float(result.reward[0])})
                if step%2==0 or failed or ended:
                    env.batch.sync_scene()
                    camera.set('eye',tuple(position+(2.6,-3.6,.9)))
                    camera.set('target',tuple(position+(0.,0.,-.1)))
                    frame=sensor.capture();image=Image.fromarray(frame['rgb'].numpy().copy());del frame
                    draw=ImageDraw.Draw(image);draw.rectangle((0,0,960,65),fill=(18,24,30))
                    draw.text((14,10),f'Gobot 29DoF G1 | {clip} | checkpoint {checkpoint.name}',fill='white')
                    draw.text((14,32),f'Tracking preview | source {start+t:.2f}s | '+('TERMINATED' if failed else 'clip end' if ended else 'running'),fill='white')
                    if images==0:image.save(output.with_suffix('.png'))
                    writer.stdin.write(np.asarray(image).tobytes());images+=1
                if failed or ended:break
        writer.stdin.close()
        if writer.wait()!=0:raise RuntimeError('tracking video encoder failed')
        writer=None
        report={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':file_digest(checkpoint),
                'task':'motion_tracking','clip':clip,'source_start_s':start,'requested_seconds':seconds,
                'preview_only':True,'backend':'mujoco-warp','physics_digest':env.batch.artifact.digest,
                'frames':images,'failure_time_s':trajectory[-1]['source_time_s'] if trajectory and trajectory[-1]['failed'] else None,
                'trajectory':trajectory}
        output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        if writer:
            if writer.stdin:writer.stdin.close()
            writer.wait()
        sensor=camera=None;gc.collect();env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clips',nargs='+',choices=tuple(WINDOWS),default=list(WINDOWS))
    p.add_argument('--seconds',type=float,default=8.)
    add_budget_arguments(p)
    a=p.parse_args()
    if not 0<a.seconds<=30:p.error('preview duration must be in (0,30] seconds')
    torch.set_num_threads(4)
    try:
        with evaluation_budget(a,stage='tracking_preview'):
            reports=[]
            for clip in a.clips:reports.append(record(a.checkpoint,a.output/f'{clip}.mp4',clip,WINDOWS[clip],a.seconds))
            (a.output/'manifest.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='trajectory'} for r in reports],indent=2)+'\n')
    finally:
        gobot.render._shutdown_headless_render_context()


if __name__=='__main__':main()
