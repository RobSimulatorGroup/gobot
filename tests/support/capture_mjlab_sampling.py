"""Compare real training data before PPO updates, using the pinned reference."""
import argparse,hashlib,random,sys,types
from pathlib import Path
import numpy as np
import torch
import rsl_rl.runners
p=argparse.ArgumentParser(description=__doc__);p.add_argument('runtime',choices=('gobot','mjlab'));p.add_argument('--count',type=int,default=64);p.add_argument('--output',type=Path,required=True);p.add_argument('--terrain-reference',type=Path,required=True);a=p.parse_args()
root=a.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
random.seed(42);np.random.seed(42);torch.manual_seed(42)
sys.path.insert(0,str(Path('tests/support').resolve()))
from mjlab_training import helper,training_terrain,set_initial_terrain
terrain=helper('terrain_reference').load(a.terrain_reference)
records={}
def save(name,**values):
 records[name]={k:v.detach().cpu().clone() for k,v in values.items()}
 records[name]['rng']=torch.cuda.get_rng_state()
 print(name,hashlib.sha256(records[name]['rng'].numpy().tobytes()).hexdigest(),flush=True)
if a.runtime=='gobot':
 import gobot.rl.locomotion.warp_env as w
 from gobot.rl.locomotion.experiment import Experiment,task_config
 cfg=task_config(Experiment(robot='g1'));cfg.episode_length_s=.2;cfg.push_interval_range_s=(.04,.08);cfg.command.resampling_time_range=(.04,.08)
 env=w.WarpVelocityEnv(cfg,num_envs=a.count,device='cuda:0',seed=42,collect_step_extras=False,reset_on_create=False)
 def snapshot(name):
  save(name,qpos=env._arrays['qpos'],qvel=env._arrays['qvel'],command=env._command,
       command_time=env._command_time_left,standing=env._is_standing_env,heading=env._heading_target,
       heading_env=env._is_heading_env,forward_env=env._is_forward_env,
       ctrl=env._arrays['ctrl'][:,env._actuator_ids],push_time=env._push_time_left,actor=env.get_observations()['actor'],critic=env.get_observations()['critic'])
 save('startup',bias=env._encoder_bias,friction=env.provider.model_array('geom_friction')[:,env._foot_geom_ids],
      com=env.provider.model_array('body_ipos')[:,env._torso_body_id],push_time=env._push_time_left)
 levels,kinds=helper('terrain_reference').initial_assignments(terrain,a.count,42)
 env.set_terrain_assignments(levels,kinds,fixed=False);env.reset()
 snapshot('reset')
 try:
  for step in range(15):
   state=env.step(torch.zeros(a.count,29,device='cuda:0'));snapshot(f'step_{step}')
   records[f'step_{step}']['reward']=state.reward.cpu().clone()
   records[f'step_{step}']['done']=state.done.cpu().clone()
 finally:env.close()
else:
 sys.path.insert(0,str(Path('mjlab/src').resolve()));import mjlab
 tasks=types.ModuleType('mjlab.tasks');tasks.__path__=[str(Path(mjlab.__file__).parent/'tasks')];sys.modules.setdefault('mjlab.tasks',tasks)
 from mjlab.envs import ManagerBasedRlEnv
 from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
 from mjlab.utils.torch import configure_torch_backends
 from mujoco_warp._src import collision_convex
 collision_convex.MJ_MAX_EPAHORIZON=64
 cfg=unitree_g1_rough_env_cfg();cfg.seed=42;cfg.scene.num_envs=a.count;cfg.sim.njmax=1500;cfg.sim.nconmax=70;cfg.episode_length_s=.2;cfg.events['push_robot'].interval_range_s=(.04,.08);cfg.commands['twist'].resampling_time_range=(.04,.08)
 configure_torch_backends(allow_tf32=False)
 with training_terrain(cfg,terrain):env=ManagerBasedRlEnv(cfg,device='cuda:0')
 robot=env.scene['robot'];term=env.command_manager.get_term('twist')
 actuator_order=torch.tensor([next(i for i,j in enumerate(env.sim.mj_model.actuator_trnid[:,0]) if env.sim.mj_model.joint(int(j)).name.endswith('/'+name)) for name in robot.joint_names],device='cuda:0')
 def snapshot(name):
  obs=env.observation_manager.compute()
  save(name,qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,command=term.command,command_time=term.time_left,
       standing=term.is_standing_env,heading=term.heading_target,heading_env=term.is_heading_env,forward_env=term.is_forward_env,
       ctrl=env.sim.data.ctrl[:,actuator_order],push_time=env.event_manager._interval_term_time_left[0],actor=obs['actor'],critic=obs['critic'])
 geom=env.event_manager.get_term_cfg('foot_friction').params['asset_cfg'].geom_ids
 body=env.event_manager.get_term_cfg('base_com').params['asset_cfg'].body_ids
 save('startup',bias=robot.data.encoder_bias,friction=env.sim.model.geom_friction[:,robot.indexing.geom_ids[geom]],
      com=env.sim.model.body_ipos[:,robot.indexing.body_ids[body]].squeeze(1),push_time=env.event_manager._interval_term_time_left[0])
 set_initial_terrain(env,terrain,42);env.reset();snapshot('reset')
 try:
  for step in range(15):
   obs,reward,terminated,truncated,_=env.step(torch.zeros(a.count,29,device='cuda:0'));snapshot(f'step_{step}')
   records[f'step_{step}']['reward']=reward.cpu().clone()
   records[f'step_{step}']['done']=(terminated|truncated).cpu().clone()
 finally:env.close()
torch.save(records,root/f'{a.runtime}.pt')
