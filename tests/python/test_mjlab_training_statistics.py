"""The reference runner logs completed episodes before automatic reset overwrites buffers."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

spec = importlib.util.spec_from_file_location('mjlab_training_support',
    Path(__file__).resolve().parents[1] / 'support/mjlab_training.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class Scene(dict):
    @property
    def terrain(self):
        return self['terrain']


def test_reference_logger_preserves_terminal_terms_and_uses_global_episode_window(tmp_path):
    names = ('upright', 'foot_swing_height')
    terms = torch.zeros(2, 2)
    dones = {'time_out': torch.zeros(2, dtype=torch.bool),
             'out_of_terrain_bounds': torch.zeros(2, dtype=torch.bool)}
    body = NS(root_link_lin_vel_b=torch.tensor([[.5,.2,0.],[.5,.2,0.]]),
              root_link_ang_vel_b=torch.zeros(2,3), site_lin_vel_w=torch.zeros(2,2,3))
    terrain = NS(terrain_levels=torch.tensor([1,3]), terrain_types=torch.tensor([0,1]),
                 cfg=NS(terrain_generator=NS(sub_terrains={'flat':None,'stairs':None})))
    feet = NS(data=NS(found=torch.ones(2,2),force=torch.zeros(2,2,3)),
              compute_first_contact=lambda dt:torch.ones(2,2,dtype=torch.bool))
    scene = Scene(robot=NS(data=body,find_sites=lambda names:([0,1],names)), terrain=terrain)
    scene.update({'feet_ground_contact':feet,'robot/root_angmom':NS(data=torch.zeros(2,3)),
                  'foot_height_scan':NS(data=NS(heights=torch.ones(2,2)*.1))})
    swing = NS(peak_heights=torch.ones(2,2)*.2)
    reward_cfgs={name:NS(weight=1.,func=swing) for name in names}
    ranges=NS(lin_vel_x=(-1.,1.),lin_vel_y=(-1.,1.),ang_vel_z=(-.5,.5))
    command=torch.tensor([[.5,0.,0.],[.5,0.,0.]])
    env=NS(device='cpu',num_envs=2,physics_dt=.005,max_episode_length_s=.08,cfg=NS(decimation=4),scene=scene,
           action_manager=NS(total_action_dim=1,action=torch.zeros(2,1)),
           command_manager=NS(get_command=lambda name:command,get_term=lambda name:NS(cfg=NS(ranges=ranges))),
           termination_manager=NS(_term_dones=dones), reset_terminated=torch.zeros(2,dtype=torch.bool))
    counter=0
    def compute(dt):
        terms[:,0]=torch.tensor([2.,4.]) if counter==1 else torch.tensor([6.,8.])
        terms[:,1]=0.
        return terms.sum(1)*dt
    env.reward_manager=NS(active_terms=names,get_term_cfg=lambda name:reward_cfgs[name],
                          compute=compute,_step_reward=terms)
    def step(action):
        nonlocal counter
        counter+=1
        env.action_manager.action=action
        env.reset_terminated[:]=torch.tensor([True,False]) if counter==1 else torch.tensor([False,True])
        dones['time_out'][:]=torch.tensor([False,False]) if counter==1 else torch.tensor([True,False])
        reward=env.reward_manager.compute(.02)
        # Mimic the reset path: the terminal rate buffer and peak heights are reused.
        terms.zero_();swing.peak_heights.zero_()
        return {},reward,env.reset_terminated,dones['time_out'],{}
    env.step=step
    scalars=[]
    logger=NS(writer=NS(add_scalar=lambda *args:scalars.append(args)),ep_extras=[],tot_time=0.)
    alg=NS(act=lambda obs:torch.ones(2,1),compute_returns=lambda obs:None,update=lambda:{'value':.2})
    runner=NS(gpu_global_rank=0,gpu_local_rank=0,gpu_world_size=1,alg=alg,logger=logger)
    agent=NS(clip_actions=None,num_steps_per_env=1)
    log=support.TrainingLog(env,runner,agent,tmp_path)
    for i in range(2):
        action=alg.act({});env.step(action);alg.compute_returns({});alg.update()
        logger.log(it=i,start_it=0,total_it=2,collect_time=1.,learn_time=1.,loss_dict={'value':.2},
                   learning_rate=.001,action_std=torch.ones(1),rnd_weight=None)
    rows=[json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert rows[0]['episode_reward']==pytest.approx(2*.02)
    assert rows[1]['episode_reward']==pytest.approx((6+4+8)*.02/2)
    assert rows[1]['episode_steps']==1.5
    assert rows[1]['episodes_completed']==2
    assert rows[1]['log']['Episode_Reward/upright']==pytest.approx(rows[1]['episode_reward']/.08)
    assert rows[1]['log']['Metrics/twist/error_vel_xy']==pytest.approx(.2)
    assert rows[1]['log']['Episode_Termination/time_out']==1
    assert rows[1]['log']['Episode_Termination/fell_over']==1
    assert rows[1]['total_steps']==4
    assert logger.ep_extras==[]
    assert scalars
