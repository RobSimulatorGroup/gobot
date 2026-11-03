"""Independent validation and held-out admission for both locomotion robots."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import torch
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.gait_metrics import BipedGaitMetrics, QuadrupedGaitMetrics
from gobot.rl.locomotion.manifest import validate_policy
from gobot.rl.checkpoint_policy import CheckpointPolicy
from gobot.rl.policy import policy_manifest_from_checkpoint
from .experiment import Experiment, task_config
from .budget import add_budget_arguments, evaluation_budget
from gobot.rl.locomotion.capabilities import assess_capabilities


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def fixed_commands(cfg):
    cfg.command.source = "external"
    cfg.command_curriculum = cfg.run_command_curriculum = ()
    for key in ('rel_standing_envs','rel_heading_envs','rel_world_envs','rel_forward_envs','rel_run_envs'):
        setattr(cfg.command,key,0.)
    cfg.command.heading_command = False
    cfg.command.resampling_time_range = (1e9,1e9)
    cfg.command.ranges.lin_vel_x = cfg.command.ranges.lin_vel_y = cfg.command.ranges.ang_vel_z = (0.,0.)
    cfg.spawn_jitter = 0.
    cfg.randomize_reset_yaw = False
    cfg.observations.actor_noise = False
    cfg.push_enabled = cfg.domain_randomization.enabled = False
    cfg.terrain_curriculum = False


def evaluation_scene(cfg, directory, seed, *, continuous=False, speed=3., seconds=30., slope=0.):
    """Write evaluation geometry as ordinary Gobot authoring resources."""
    root = Path(cfg.project_path).resolve()
    directory = Path(directory).resolve()
    # Resources must remain inside the project for res:// resolution.
    directory.mkdir(parents=True,exist_ok=True)
    is_g1 = cfg.robot_name == 'g1'
    resource_name = 'flat_terrain.jres' if 'flat' in cfg.scene_path else ('rough_terrain.jres' if is_g1 else 'practical_terrain.jres')
    resource = json.loads((root/'terrain'/resource_name).read_text())
    values = resource['__RESOURCE__']; values['seed'] = int(seed)
    if continuous:
        values['num_rows'] = values['num_cols'] = 1
        values['patch_size']['matrix_data']['storage'] = [max(64.,speed*seconds+20.)]*2
        values['border_width'] = 2.
        terrain = deepcopy(values['sub_terrains'][3 if slope else 0])
        terrain.update(type='PyramidSlope' if slope else 'Flat', name='continuous_slope' if slope else 'continuous_flat')
        terrain['slope_range']['matrix_data']['storage'] = [math.tan(math.radians(slope))]*2
        values['sub_terrains'] = [terrain]
    name=f'{seed}_{"continuous" if continuous else "cells"}_{slope:g}'
    target = directory/f'{name}.jres'
    target.write_text(json.dumps(resource,indent=2)+'\n')
    scene = (root/cfg.scene_path.removeprefix('res://')).read_text()
    import re
    scene = re.sub(r'res://terrain/[^"\s]+\.jres','res://'+str(target.relative_to(root)),scene)
    scene_path=directory/f'{name}.jscn'; scene_path.write_text(scene)
    return 'res://'+str(scene_path.relative_to(root))


def rollout(env, policy, *, command, steps, route=False, seed=71, sequence=None, reference=None):
    count = env.num_envs
    feet = len(env.cfg_obj.robot.feet)
    gait = (BipedGaitMetrics if feet == 2 else QuadrupedGaitMetrics)(count,env.step_dt)
    env.reset(seed=seed)
    start=array(env.locomotion_state()['base_position'])[:,:2].copy()
    levels, types = map(array,env.terrain_assignments)
    centers=array(env.terrain_origins)[levels*env.terrain_grid_shape[1]+types,:2]
    command=np.asarray(command,dtype=np.float32)
    direction=command[:2]/max(np.linalg.norm(command[:2]),1e-6)
    alive=np.ones(count,bool); failed=np.zeros(count,bool); reached=np.zeros(count,bool)
    samples=np.zeros(count); errors=np.zeros(count); yaw=np.zeros(count); slip=np.zeros(count); progress=np.zeros(count)
    deviation=np.zeros(count); clearance=np.full(count,np.inf); reasons=np.zeros(count,int)
    command_speed_sum=np.zeros(count); stop_speed=np.zeros(count); stop_samples=np.zeros(count)
    with torch.inference_mode():
        for step in range(steps):
            current=command
            if sequence:
                for begin, value in sequence:
                    if step*env.step_dt >= begin: current=np.asarray(value,dtype=np.float32)
            env.set_command(current)
            state=env.step(policy(env.get_observations()['actor']))
            physical={key:array(value) for key,value in state.info['transition_state'].items()}
            contact=np.linalg.norm(physical['foot_force'],axis=-1)>10.
            gait.update(contact,physical['foot_position'],alive)
            velocity=physical['base_linear_velocity_body'][:,:2]
            errors[alive]+=np.linalg.norm(velocity-current[:2],axis=1)[alive]
            yaw[alive]+=np.abs(physical['base_angular_velocity_body'][:,2]-current[2])[alive]
            slip[alive]+=(np.square(physical['foot_velocity'][...,:2]).sum(-1)*contact).sum(1)[alive]
            progress[alive]+=(velocity@direction)[alive]
            command_speed_sum[alive]+=np.linalg.norm(current[:2])
            clearance[alive]=np.minimum(clearance[alive],physical['base_clearance'][alive])
            samples[alive]+=1
            displacement=physical['base_position'][:,:2]-start
            if not sequence and abs(current[2])<1e-6 and np.linalg.norm(current[:2])>.05:
                cross=np.abs(displacement[:,0]*direction[1]-displacement[:,1]*direction[0])
                deviation[alive]=np.maximum(deviation[alive],cross[alive])
            if np.linalg.norm(current)<.05:
                begin=max((t for t,v in sequence if t<=step*env.step_dt),default=0.) if sequence else 0.
                if step*env.step_dt-begin>=2:
                    stop_speed[alive]+=np.linalg.norm(velocity,axis=1)[alive];stop_samples[alive]+=1
            term=array(state.terminated).astype(bool); timeout=array(state.truncated).astype(bool)
            reasons[alive & (term|timeout)] = array(state.info['reset_reason'])[alive & (term|timeout)]
            failed |= alive & (term | (timeout & (reasons!=2)))
            alive &= ~(term|timeout)
            if route:
                relative=physical['base_position'][:,:2]-centers
                crossed=(relative@direction>=3.) & (np.abs(relative).max(1)<3.5)
                wrong=(np.abs(relative).max(1)>3.5) & ~crossed
                reached |= alive & crossed
                failed |= alive & wrong
                alive &= ~(crossed|wrong)
            if not alive.any(): break
    divisor=np.maximum(samples,1)
    speed=command_speed_sum/divisor
    velocity_error=errors/divisor; yaw_error=yaw/divisor
    success=~failed & (reached if route else samples>=steps)
    success &= velocity_error<=np.maximum(.15,.2*speed)
    success &= yaw_error<=.2
    if not sequence and np.linalg.norm(command[:2])>.05:
        success &= progress/divisor >= .8*np.linalg.norm(command[:2])
        if abs(command[2])<1e-6: success &= deviation<=.5
    if stop_samples.any(): success &= stop_speed/np.maximum(stop_samples,1)<=.1
    mobility=success.copy()
    gait_data=gait.arrays()
    natural=np.ones(count,bool)
    if not sequence and command[0]>.2 and abs(command[1])+abs(command[2])<.05:
        if feet == 2:
            natural &= gait_data['landings']>=4
            natural &= gait_data['alternation_fraction']>=.9
            if command[0]>=1.6: natural &= gait_data['flight_fraction']>=.05
            if reference is not None:
                candidates=[r for r in reference['ranges'] if r['gait']==int(command[0]>=1.6) and abs(r['speed']-command[0])<=.25]
                if not candidates:
                    natural[:]=False
                else:
                    target=min(candidates,key=lambda r:abs(r['speed']-command[0]))
                    for key in ('cadence_hz','stride_length_m'):
                        natural &= (gait_data[key]>=target[key][0]) & (gait_data[key]<=target[key][1])
        else:
            key='walk_fraction' if command[0]<.7 else 'trot_fraction' if command[0]<1.9 else 'gallop_fraction'
            fraction=gait_data[key]
            if .7<=command[0]<=.9:fraction=gait_data['walk_fraction']+gait_data['trot_fraction']
            if 1.9<=command[0]<=2.1:fraction=gait_data['trot_fraction']+gait_data['gallop_fraction']
            natural &= (gait_data['cycles']>=3)&(fraction>=.7)
            natural &= gait_data['simultaneous_landings'] <= .1*np.maximum(gait_data['landings'],1)
            if command[0]>=2.1: natural &= gait_data['flight_fraction']>=.05
    cells=[]
    for level,kind in sorted(set(zip(levels.tolist(),types.tolist()))):
        mask=(levels==level)&(types==kind)
        cells.append({'level':level,'terrain':env.terrain_type_names[kind], 'episodes':int(mask.sum()),
                      'admission':float(mobility[mask].mean()),'natural_admission':float((mobility&natural)[mask].mean())})
    return {'command':command.tolist(),'sequence':sequence,'seed':seed,'steps':steps,'route':route,'admission':float(mobility.mean()),
            'natural_admission':float((mobility&natural).mean()),'survival':float((~failed).mean()),
            'velocity_error':float(velocity_error.mean()),'yaw_error':float(yaw_error.mean()),
            'cross_track_max_m':float(deviation.max()),'foot_slip_m2_s2':float((slip/divisor).mean()),
            'minimum_base_clearance_m':float(clearance.min()),'gait':{k:np.asarray(v).mean(axis=0).tolist() for k,v in gait_data.items()},
            'failure_reasons':{str(i):int((reasons==i).sum()) for i in np.unique(reasons)},'cells':cells,
            'episode_metrics':{'admitted':mobility.tolist(),'natural':natural.tolist(),
                               'velocity_error':velocity_error.tolist(),'yaw_error':yaw_error.tolist()}}


def rank_report(report):
    results=report['scenarios']
    worst=min(c['admission'] for r in results for c in r['cells'])
    average=float(np.mean([r['admission'] for r in results]))
    violation=max(max(r['velocity_error']/max(.15,.2*np.linalg.norm(r['command'][:2])),r['yaw_error']/.2,
                      r['cross_track_max_m']/.5) for r in results)
    natural=float(np.mean([r['natural_admission'] for r in results]))
    return [float(report.get('stage_passed',report['passed'])),worst,average,-violation,natural,-float(np.mean([r['foot_slip_m2_s2'] for r in results]))]


def evaluate(checkpoint, output, *, steps=None, count=None, seed=71, final=False, device='cuda:0', backend='mujoco-warp', scenario_stage=None, slope=0., robust=False):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    exp=Experiment(**payload['infos']['experiment'])
    if scenario_stage is not None:exp=replace(exp,stage=scenario_stage)
    cfg=task_config(exp,play=True);fixed_commands(cfg)
    cfg.episode_length_s=1e9
    if robust:
        cfg.domain_randomization.enabled=True
        cfg.push_enabled=True
    folder=Path(cfg.project_path)/'assets/evaluation'/Path(output).stem
    seconds=(steps or exp.eval_steps)*.02
    cfg.scene_path=evaluation_scene(cfg,folder,seed,continuous=exp.stage.startswith('flat'),seconds=seconds,slope=slope)
    is_flat=exp.stage.startswith('flat')
    count=count or exp.eval_envs
    if not is_flat:
        cells=10*7
        count=cells*(10 if final else math.ceil(count/cells))
    env=make_velocity_env(cfg,num_envs=count,backend=backend,device=device,seed=seed,collect_step_extras=False)
    try:
        validate_policy(policy_manifest_from_checkpoint(payload),env,cfg)
        policy=CheckpointPolicy(payload['actor_state_dict'],policy_manifest_from_checkpoint(payload)).eval().to(device)
        rows,cols=env.terrain_grid_shape
        indices=np.arange(count)%(rows*cols)
        env.set_terrain_assignments(indices//cols,indices%cols,fixed=True)
        slow=exp.stage=='flat_control'
        commands=[(.8,0,0),(-.1 if slow else -.3,0,0),(0,.15 if slow else .3,0),
                  (0,0,.3 if slow else .6),(0,0,0)]
        if final: commands += [(0,-.3,0),(0,0,-.6),(.5,0,0),(1.2,0,0)]
        if is_flat and exp.stage=='flat_gait':
            commands += [(1.8 if exp.robot=='g1' else 1.4,0,0),(2.2 if exp.robot=='g1' else 2.8,0,0)]
        reference=None
        if exp.robot=='g1' and exp.stage=='flat_gait':
            from examples.g1.reference_gaits import build_reference
            reference=build_reference(Path(exp.motion_dataset),split='holdout' if final else 'validation')
        results=[]
        for command in commands:
            speed=math.hypot(*command[:2])
            route=not is_flat and speed>.05
            if route:cfg.spawn_offset_xy=tuple(-3*np.asarray(command[:2])/speed)
            elif speed>.05:
                margin=max(64.,3.*seconds+20.)/2-10.
                cfg.spawn_offset_xy=tuple(-margin*np.asarray(command[:2])/speed)
            else:cfg.spawn_offset_xy=(0.,0.)
            duration=max(steps or exp.eval_steps, math.ceil((6/speed+2)/env.step_dt)) if route else (steps or exp.eval_steps)
            results.append(rollout(env,policy,command=command,steps=duration,route=route,seed=seed,reference=reference))
        if final and is_flat and slope==0.:
            top=2.2 if exp.robot=='g1' else 3.
            sequence=[(0,(0,0,0)),(3,(.5,0,0)),(8,(1.2,0,0)),(13,(top,0,0)),(20,(.8,0,0)),(25,(0,0,0))]
            cfg.spawn_offset_xy=(-max(64.,3.*seconds+20.)/2+10.,0.)
            results.append(rollout(env,policy,command=(.8,0,0),steps=1500,seed=seed,sequence=sequence))
        threshold=.95 if final else .9
        passed=all(r['admission']>=threshold and min(c['admission'] for c in r['cells'])>=.9 for r in results)
        if exp.stage=='flat_gait': passed &= all(r['natural_admission']>=threshold for r in results if r['command'][0]>.2)
        report={'schema_version':1,'robot':exp.robot,'stage':exp.stage,'checkpoint':str(checkpoint),
                'protocol':'heldout_admission' if final else 'validation', 'slope_degrees':slope, 'robustness':robust, 'passed':bool(passed),'scenarios':results}
        if exp.schema_version >= 2:
            report.update(assess_capabilities(results, exp.stage, final=final,
                natural=exp.stage=='flat_gait', run_speed=1.6 if exp.robot=='g1' else 1.9))
        report['score']=rank_report(report)
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=71);p.add_argument('--steps',type=int);p.add_argument('--num-envs',type=int)
    p.add_argument('--final',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--scenario-stage',choices=('flat_control','flat_gait','rough','mixed'))
    p.add_argument('--slope',type=float,default=0.)
    p.add_argument('--robust',action='store_true')
    p.add_argument('--backend',default='mujoco-warp',choices=('mujoco-cpu','mujoco-warp'))
    add_budget_arguments(p)
    a=p.parse_args();torch.set_num_threads(4)
    with evaluation_budget(a, stage='assessment'):
        result=evaluate(a.checkpoint,a.output,steps=a.steps,count=a.num_envs,seed=a.seed,final=a.final,device=a.device,backend=a.backend,scenario_stage=a.scenario_stage,slope=a.slope,robust=a.robust)
    print(json.dumps({k:v for k,v in result.items() if k!='scenarios'}))

if __name__=='__main__':main()
