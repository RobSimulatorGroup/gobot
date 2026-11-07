"""Export reference data using mjlab dependencies and PYTHONPATH=mjlab/src.

This script does not import Gobot. Run after deliberately reviewing a new mjlab
revision; numerical expectations are produced by the reference reward functions.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import mujoco
import re
from mjlab.scene import Scene
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg, unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg, unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from unittest.mock import Mock, patch
from copy import deepcopy
from mjlab.rl.runner import MjlabOnPolicyRunner

root=Path(__file__).resolve().parents[2]
reference=root/'mjlab'
source_files=['src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml','src/mjlab/asset_zoo/robots/unitree_go1/go1_constants.py','src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py','src/mjlab/rl/runner.py','src/mjlab/tasks/velocity/velocity_env_cfg.py','src/mjlab/tasks/velocity/mdp/rewards.py','src/mjlab/tasks/velocity/mdp/curriculums.py','src/mjlab/tasks/velocity/config/go1/rl_cfg.py','src/mjlab/tasks/velocity/config/g1/rl_cfg.py','src/mjlab/tasks/velocity/config/go1/env_cfgs.py','src/mjlab/tasks/velocity/config/g1/env_cfgs.py']
out={'reference_revision':'dcdf6ce519abfd2ea45e9503ae6fb650b75bb869','source_sha256':{name:hashlib.sha256((reference/name).read_bytes()).hexdigest() for name in source_files},'tasks':{}}
for robot,rough,flat,agent in [('go1',unitree_go1_rough_env_cfg,unitree_go1_flat_env_cfg,unitree_go1_ppo_runner_cfg),('g1',unitree_g1_rough_env_cfg,unitree_g1_flat_env_cfg,unitree_g1_ppo_runner_cfg)]:
    for terrain,factory in [('rough',rough),('flat',flat)]:
        cfg=factory()
        command=cfg.commands['twist']
        out['tasks'][f'{robot}_{terrain}']={'agent':asdict(agent()),'weights':{k:v.weight for k,v in cfg.rewards.items()},'physics_dt':cfg.sim.mujoco.timestep,'decimation':cfg.decimation,'episode_length_s':cfg.episode_length_s,'actor_terms':list(cfg.observations['actor'].terms),'noise':{k:[v.noise.n_min,v.noise.n_max] for k,v in cfg.observations['actor'].terms.items() if v.noise is not None},'commands':{k:getattr(command,k) for k in ['rel_heading_envs','rel_standing_envs','rel_forward_envs','heading_command','resampling_time_range']},'command_ranges':asdict(command.ranges),'terminations':list(cfg.terminations),'action_scale':cfg.actions['joint_pos'].scale,'solver':{name:getattr(cfg.sim.mujoco,name) for name in ('solver','integrator','cone','impratio','iterations','ls_iterations','ccd_iterations')}}
        with patch('mjlab.rl.runner.OnPolicyRunner.__init__', return_value=None) as constructor:
            MjlabOnPolicyRunner(None, asdict(agent()), device='cpu')
            out['tasks'][f'{robot}_{terrain}']['runtime_agent'] = deepcopy(constructor.call_args.args[1])

out['profiles']={}
for robot,factory in [('go1',unitree_go1_flat_env_cfg),('g1',unitree_g1_flat_env_cfg)]:
    cfg=factory()
    scene=Scene(cfg.scene,device='cpu')
    model=scene.compile()
    profile={}
    for actuator in range(model.nu):
        joint=int(model.actuator_trnid[actuator,0])
        name=model.joint(joint).name.removeprefix('robot/')
        qadr=int(model.jnt_qposadr[joint]);dadr=int(model.jnt_dofadr[joint])
        matches=[value for pattern,value in cfg.actions['joint_pos'].scale.items() if re.fullmatch(pattern,name)]
        assert len(matches)==1,name
        profile[name]={'kp':float(model.actuator_gainprm[actuator,0]),
                       'kd':float(-model.actuator_biasprm[actuator,2]),
                       'armature':float(model.dof_armature[dadr]),
                       'effort_limit':float(model.actuator_forcerange[actuator,1]),
                       'default_position':float(model.key('init_state').qpos[qadr]),
                       'action_scale':matches[0]}
    out['profiles'][robot]=profile

# Independent numerical probes: different command speeds, tilt, landing/sliding,
# joint-limit violations, and nonzero vertical/roll/pitch velocity.
g=torch.Generator().manual_seed(171)
rand=lambda *shape:torch.randn(*shape,generator=g)
B,J,F=4,12,4
command=torch.tensor([[0.,0.,0.],[1.,0.,.2],[-.4,.1,-.1],[2.,0.,0.]])
quat=rand(B,4);quat/=quat.norm(dim=-1,keepdim=True)
lin=rand(B,3);ang=rand(B,3)
joints=rand(B,J);previous=rand(B,J);action=rand(B,J)
foot_vel=rand(B,F,3);height=rand(B,F).abs()*.1;peak=rand(B,F).abs()*.1
found=torch.tensor([[1.,0.,1.,0.],[0.,0.,1.,1.],[1.,1.,1.,1.],[0.,1.,0.,1.]])
first=torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,1],[0,0,0,1]],dtype=torch.bool)
force=rand(B,F,3)*20;history=rand(B,F,4,3)*6
stand=torch.full((J,),.05);walk=torch.full((J,),.3);run=torch.full((J,),.6)
feet=NS(data=NS(found=found,force=force,current_air_time=torch.tensor([[0.,.2,0.,.6],[.3,.1,0.,0.],[0.,0.,0.,0.],[.05,0.,.49,0.]])),compute_first_contact=lambda dt:first)
scan=Mock(spec=TerrainHeightSensor);scan.data=NS(heights=height)
robot=NS(data=NS(root_link_lin_vel_b=lin,root_link_ang_vel_b=ang,root_link_quat_w=quat,body_link_quat_w=quat[:,None,:],gravity_vec_w=torch.tensor([[0.,0.,-1.]]).expand(B,-1),body_link_ang_vel_w=quat_apply(quat,ang)[:,None,:],joint_pos=joints,default_joint_pos=torch.zeros(B,J),soft_joint_pos_limits=torch.tensor([-.9,.9]).expand(B,J,2),site_lin_vel_w=foot_vel))
env=NS(scene={'robot':robot,'feet':feet,'height':scan,'collisions':NS(data=NS(force_history=history,found=found)),'angmom':NS(data=rand(B,3))},action_manager=NS(action=action,prev_action=previous),command_manager=NS(get_command=lambda name:command),extras={'log':{}},step_dt=.02,num_envs=B,device='cpu')
asset=SceneEntityCfg('robot');asset.body_ids=[0];asset.joint_ids=slice(None);asset.site_ids=slice(None)
pose=mdp.variable_posture.__new__(mdp.variable_posture);pose.std_standing=stand;pose.std_walking=walk;pose.std_running=run;pose.default_joint_pos=torch.zeros(B,J)
upright=mdp.upright.__new__(mdp.upright)
swing=mdp.feet_swing_height.__new__(mdp.feet_swing_height);swing.peak_heights=peak.clone();swing.step_dt=.02
rates={
 'track_linear_velocity':mdp.track_linear_velocity(env,.5,'twist'),
 'track_angular_velocity':mdp.track_angular_velocity(env,2**-.5,'twist'),
 'upright':upright(env,.2**.5,asset_cfg=asset),
 'pose':pose(env,None,None,None,asset,'twist',walking_threshold=.05,running_threshold=1.5),
 'body_ang_vel':mdp.body_angular_velocity_penalty(env,asset),
 'angular_momentum':mdp.angular_momentum_penalty(env,'angmom'),
 'dof_pos_limits':mdp.joint_pos_limits(env,asset),
 'action_rate_l2':mdp.action_rate_l2(env),
 'air_time':mdp.feet_air_time(env,'feet',command_name='twist'),
 'foot_clearance':mdp.feet_clearance(env,.1,'height','twist',.05,asset),
 'foot_swing_height':swing(env,'feet','height',.1,'twist',.05),
 'foot_slip':mdp.feet_slip(env,'feet','twist',.05,asset),
 'soft_landing':mdp.soft_landing(env,'feet','twist',.05),
 'self_collisions':mdp.self_collision_cost(env,'collisions'),
 'shank_collision':mdp.self_collision_cost(env,'collisions'),
 'trunk_head_collision':mdp.self_collision_cost(env,'collisions'),
}
# This probe deliberately enables every term; production weights are checked separately.
weights={name:float(i+1)/10 for i,name in enumerate(rates)}
inputs={'command':command,'quaternion':quat,'linear_body':lin,'angular_body':ang,'joint_position':joints,'action':action,'previous_action':previous,'foot_velocity':foot_vel,'foot_height':height,'peak_before':peak,'foot_contact':found,'first_contact':first,'contact_force':force,'force_history':history,'air_time':feet.data.current_air_time,'angular_momentum':env.scene['angmom'].data,'pose_standing':stand,'pose_walking':walk,'pose_running':run}
out['reward_probe']={'inputs':{k:v.tolist() for k,v in inputs.items()},'rates':{k:v.tolist() for k,v in rates.items()},'weights':weights,'reward':sum(weights[k]*v for k,v in rates.items()).mul(.02).tolist()}
(root/'tests/fixtures/mjlab_velocity.json').write_text(json.dumps(out,indent=2)+'\n')
print('Exported',len(out['tasks']),'task configurations and',len(rates),'independent reward terms')

# Keep a self-contained physical model for scene compiler regression tests.
from mjlab.entity import Entity
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
spec = Entity(get_g1_robot_cfg()).spec
for geom in list(spec.geoms):
    if not geom.contype and not geom.conaffinity:
        spec.delete(geom)
for mesh in list(spec.meshes):
    spec.delete(mesh)
spec.compile()
(root/'tests/fixtures/mjlab_g1.xml').write_text(
    '<!-- mjlab '+out['reference_revision']+'; visual-only meshes omitted. -->\n'+spec.to_xml())
