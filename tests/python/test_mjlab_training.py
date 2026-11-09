"""Compare Gobot's standard velocity training with independently exported mjlab data."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from gobot.rl.locomotion.config import VELOCITY_REWARD_TERM_NAMES
from gobot.rl.locomotion.cpu_env import CpuVelocityEnv, _TASK_PARAM
from examples.locomotion.presets import Experiment, MJLAB_REFERENCE_REVISION, task_config
from gobot.rl.locomotion.training import agent_config
from gobot.rl.locomotion.warp_env import WarpVelocityEnv, _quat_apply

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=json.loads((ROOT/'tests/fixtures/mjlab_velocity.json').read_text())


def test_reference_sources_are_identified():
    assert REFERENCE['reference_revision']==MJLAB_REFERENCE_REVISION
    for name,digest in REFERENCE['source_sha256'].items():
        source=ROOT/'mjlab'/name
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest()==digest, f'regenerate reference after reviewing {name}'


@pytest.mark.parametrize('robot',['go1','g1'])
@pytest.mark.parametrize('terrain',['flat','rough'])
def test_standard_velocity_configuration_matches_reference(robot,terrain):
    recipe=Experiment(robot=robot,stage=terrain)
    cfg=task_config(recipe)
    agent=agent_config(recipe)
    ref=REFERENCE['tasks'][f'{robot}_{terrain}']
    assert cfg.physics_dt==ref['physics_dt'] and cfg.decimation==ref['decimation']
    assert cfg.episode_length_s==ref['episode_length_s']
    solver_names={'impratio':'impedance_ratio','ls_iterations':'line_search_iterations','ccd_iterations':'convex_collision_iterations'}
    for name,value in ref['solver'].items():
        actual=cfg.mujoco_solver_settings[solver_names.get(name,name)]
        assert (actual.lower() if isinstance(actual,str) else actual)==value,name
    assert cfg.observations.history_length==1
    assert cfg.observations.include_height_scan==('height_scan' in ref['actor_terms'])
    assert not cfg.gait.enabled
    assert cfg.capture_transitions and cfg.action_clip is None
    assert not hasattr(cfg, "post_step_sensing")
    for key in VELOCITY_REWARD_TERM_NAMES:
        assert getattr(cfg.rewards,key)==ref['weights'].get(key,0),key
    for key,value in ref['noise'].items():
        assert list(cfg.observations.actor_noise_ranges[key])==value,key
    for key,value in ref['commands'].items():
        actual=getattr(cfg.command,key)
        assert (list(actual) if isinstance(actual,tuple) else actual)==value,key
    for key,value in ref['command_ranges'].items():
        actual=getattr(cfg.command.ranges,key)
        assert (list(actual) if isinstance(actual,tuple) else actual)==value,key
    assert [stage.step for stage in cfg.command_curriculum]==[0,120000,240000]
    assert agent['algorithm']['class_name']=='gobot.rl.locomotion.transition_ppo:TransitionPPO'
    for key,value in ref['agent']['algorithm'].items():
        if key == 'class_name': continue  # Gobot bootstraps from the final state.
        assert agent['algorithm'].get(key)==value,key
    for role in ('actor','critic'):
        expected=ref['agent'][role]
        assert list(agent[role]['hidden_dims'])==expected['hidden_dims']
        assert agent[role]['activation']==expected['activation']
        assert agent[role]['obs_normalization']==expected['obs_normalization']
    distribution=agent['actor']['distribution_cfg']
    assert distribution['init_std']==1. and distribution['std_type']=='scalar'
    assert agent['num_steps_per_env']==ref['agent']['num_steps_per_env']
    assert recipe.save_interval==ref['agent']['save_interval']


def _probe():
    probe=REFERENCE['reward_probe']
    inputs={key:np.asarray(value,dtype=np.float32) for key,value in probe['inputs'].items()}
    cfg=task_config(Experiment(robot='go1',stage='flat'))
    for name,weight in probe['weights'].items():setattr(cfg.rewards,name,weight)
    cfg.rewards.upright_reference='world'
    cfg.fall_angle=None
    cfg.illegal_contact.enabled=False
    cfg.observations.actor_noise=False
    return inputs,cfg


def _cpu_probe():
    inp,cfg=_probe();count,joints=inp['joint_position'].shape
    env=CpuVelocityEnv.__new__(CpuVelocityEnv)
    env.cfg_obj=cfg;env.num_envs=count;env.num_actions=joints;env._height_scan_dim=1;env.step_dt=.02
    env._reward_term_names=VELOCITY_REWARD_TERM_NAMES
    quat=torch.tensor(inp['quaternion'])
    linear=_quat_apply(quat,torch.tensor(inp['linear_body'])).numpy()
    angular=_quat_apply(quat,torch.tensor(inp['angular_body'])).numpy()
    history=inp['force_history']
    collision=(np.linalg.norm(history,axis=-1)>10).any(axis=1).sum(-1).astype(np.float32)
    params=np.zeros(len(_TASK_PARAM),np.float32)
    for name,value in {'step_dt':.02,'lin_vel_std2':.25,'ang_vel_std2':.5,'upright_std2':.2,'command_threshold':.05,'height_scan_max_distance':5.}.items():
        params[_TASK_PARAM[name]]=value
    state=NS(task_params=params,reward_weights=np.array([REFERENCE['reward_probe']['weights'][key] for key in VELOCITY_REWARD_TERM_NAMES],np.float32),
        command=inp['command'],base_quaternion=inp['quaternion'],base_linear_velocity=linear,base_angular_velocity=angular,
        base_linear_velocity_body=inp['linear_body'],base_angular_velocity_body=inp['angular_body'],projected_gravity=np.tile([0.,0.,-1.],(count,1)),
        joint_position=inp['joint_position'],joint_velocity=np.zeros((count,joints),np.float32),joint_lower_limit=-np.ones(joints,np.float32),joint_upper_limit=np.ones(joints,np.float32),
        previous_action=inp['previous_action'],action=inp['action'],submitted_action=inp['action'],last_action=inp['action'],default_joint_position=np.zeros(joints,np.float32),
        pose_std_standing=inp['pose_standing'],pose_std_walking=inp['pose_walking'],pose_std_running=inp['pose_running'],
        foot_contact=inp['foot_contact'],foot_height=inp['foot_height'],foot_velocity=inp['foot_velocity'],
        foot_peak_height=np.where(inp['foot_contact']==0,np.maximum(inp['peak_before'],inp['foot_height']),inp['peak_before']),first_contact=inp['first_contact'],
        landing_force=np.linalg.norm(inp['contact_force'],axis=-1)*inp['first_contact'],foot_air_time=inp['air_time'],foot_contact_force=inp['contact_force'],
        self_collision_count=collision,shank_collision_count=collision,trunk_head_collision_count=collision,
        illegal_contact_count=np.zeros(count),height_scan=np.ones((count,1),np.float32),base_height=np.ones(count,np.float32),angular_momentum=inp['angular_momentum'],
        reward_terms=np.zeros((count,len(VELOCITY_REWARD_TERM_NAMES)),np.float32),reward=np.zeros(count,np.float32),terminated=np.zeros(count,np.uint8),
        actor_obs=np.zeros((count,48),np.float32),critic_obs=np.zeros((count,72),np.float32),encoder_bias=np.zeros((count,joints),np.float32))
    for key in ('velocity_error','foot_slip','base_clearance','terrain_normal_error'):setattr(state,key,np.zeros(count,np.float32))
    env.backend=NS(state=state)
    env._run_velocity_task_numpy()
    return env,state


def test_cpu_reward_terms_match_mjlab():
    _,state=_cpu_probe()
    probe=REFERENCE['reward_probe']
    for i,key in enumerate(VELOCITY_REWARD_TERM_NAMES):
        expected=np.asarray(probe['rates'][key])*probe['weights'][key]
        np.testing.assert_allclose(state.reward_terms[:,i],expected,rtol=2e-5,atol=3e-6,err_msg=key)
    np.testing.assert_allclose(state.reward,probe['reward'],rtol=2e-5,atol=3e-6)


def test_torch_reward_terms_match_mjlab_without_cuda():
    raw,cfg=_probe();inp={k:torch.tensor(v) for k,v in raw.items()};count,joints=inp['joint_position'].shape
    env=WarpVelocityEnv.__new__(WarpVelocityEnv)
    env.cfg_obj=cfg;env.num_envs=count;env.num_actions=joints;env.step_dt=.02
    env._root_velocity_body=lambda:(inp['linear_body'],inp['angular_body'])
    world_ang=_quat_apply(inp['quaternion'],inp['angular_body'])
    env._root_link_velocity_world=lambda:(inp['linear_body'],world_ang)
    env._foot_velocity_world=lambda:inp['foot_velocity']
    env._arrays={'qpos':inp['joint_position'],'xquat':inp['quaternion'][:,None,:],
                 'cvel':torch.cat((world_ang,torch.zeros(count,3)),1)[:,None,:], 'sensordata':inp['angular_momentum']}
    env._base_body_id=env._torso_body_id=0
    env._joint_qpos_ids=torch.arange(joints);env._default_joint_pos=torch.zeros(joints)
    env._terrain_up_world=torch.tensor([[0.,0.,1.]]).expand(count,-1)
    env._pose_std_standing=inp['pose_standing'];env._pose_std_walking=inp['pose_walking'];env._pose_std_running=inp['pose_running']
    env._joint_limits=torch.tensor([[-1.,1.]]).expand(joints,-1)
    env._action=inp['action'];env._previous_action=inp['previous_action'];env._command=inp['command']
    env._feet_contact={'found':inp['foot_contact'],'force':inp['contact_force']}
    env._foot_height=inp['foot_height'];env._foot_peak_height=inp['peak_before'].clone()
    env._current_contact_time=inp['first_contact']*.005
    env._current_air_time=inp['air_time']
    env._contact_histories={key:inp['force_history'] for key in ('self_collision','shank_ground_touch','trunk_ground_touch')}
    env._sensor_slices=(slice(0,0),slice(0,0),slice(0,3))
    env._reward_term_names=VELOCITY_REWARD_TERM_NAMES
    env._reward_weights=torch.tensor([REFERENCE['reward_probe']['weights'][key] for key in VELOCITY_REWARD_TERM_NAMES])
    env._reward_terms=torch.zeros(count,len(VELOCITY_REWARD_TERM_NAMES));env._reward=torch.zeros(count);env._episode_returns=torch.zeros(count)
    env._compute_rewards()
    probe=REFERENCE['reward_probe']
    for i,key in enumerate(VELOCITY_REWARD_TERM_NAMES):
        expected=torch.tensor(probe['rates'][key])*probe['weights'][key]
        torch.testing.assert_close(env._reward_terms[:,i],expected,rtol=2e-5,atol=3e-6,msg=key)
    torch.testing.assert_close(env._reward,torch.tensor(probe['reward']),rtol=2e-5,atol=3e-6)


def test_actor_encoder_bias_does_not_change_critic():
    env,state=_cpu_probe()
    state.encoder_bias[:]=.013
    actor,critic=env._build_generic_velocity_observations(state)
    np.testing.assert_allclose(actor[:,9:21]-critic[:,9:21],.013,atol=1e-7)


def test_full_ppo_updates_match_reference_runner():
    """Same non-timeout transitions: optimizer, normalization and Adam match mjlab."""
    from rsl_rl.runners import OnPolicyRunner
    torch.set_num_threads(2)
    from rsl_rl.algorithms import PPO
    from tensordict import TensorDict
    from gobot.rl.locomotion.training import build_algorithm, update_policy
    from gobot.rl.locomotion.distributed import DistributedTraining

    class Environment:
        num_envs=4
        num_actions=12
        max_episode_length=1000
        device='cpu'
        cfg={}
        def __init__(self):
            self.episode_length_buf=torch.zeros(4,dtype=torch.long)
            self.count=0
            self.values=torch.linspace(-.5,.5,4*48).reshape(4,48)
        def get_observations(self):
            return TensorDict({'actor':self.values.clone(),'critic':torch.cat((self.values,self.values[:,:24]),1)},batch_size=[4])
        def step(self,action):
            self.count+=1
            self.episode_length_buf+=1
            reward=1-action.square().mean(-1)*.05
            timeout=self.episode_length_buf>=self.max_episode_length
            failure=torch.zeros(4,dtype=torch.bool)
            failure[1]=self.count%11==0
            done=timeout|failure
            self.values=torch.sin(self.values+.1+action.mean(-1,keepdim=True)*.01)
            self.values[done]=0
            self.episode_length_buf[done]=0
            return self.get_observations(),reward,done,{'time_outs':timeout&~failure}

    def run(reference):
        torch.manual_seed(617)
        config=deepcopy(REFERENCE['tasks']['go1_rough']['runtime_agent']) if reference else agent_config(Experiment(robot='go1'))
        env=Environment()
        if reference:
            runner=OnPolicyRunner(env,config,device='cpu')
            alg=runner.alg
            runner.learn(2,init_at_random_ep_len=False)
        else:
            alg=build_algorithm(env,config,'cpu',DistributedTraining('cpu'))
            observation=env.get_observations()
            alg.train_mode()
            for _ in range(2):
                observation,_=update_policy(alg,env,observation,config['num_steps_per_env'])
        assert isinstance(alg, PPO)
        return alg.save()

    expected=run(True)
    actual=run(False)
    def compare(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:compare(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b,strict=True):compare(x,y)
        else:assert a==b
    compare(actual,expected)


@pytest.mark.parametrize("robot,dimensions", [("go1", (48,72)), ("g1", (99,111))])
def test_flat_observations_and_reset_command_timers(robot,dimensions):
    from gobot.rl.locomotion import make_velocity_env
    cfg=task_config(Experiment(robot=robot,stage='flat'))
    cfg.observations.actor_noise=False
    cfg.domain_randomization.enabled=False
    cfg.push_enabled=False
    cfg.command.resampling_time_range=(1.,1.)
    env=make_velocity_env(cfg,backend='mujoco-cpu',num_envs=2,seed=31,max_episode_length=2)
    try:
        assert (env.num_obs,env.num_privileged_obs)==dimensions
        env.step(np.zeros((2,env.num_actions),np.float32))
        np.testing.assert_allclose(env.backend.state.command_time_left, .98, atol=1.e-7)
        result=env.step(np.zeros((2,env.num_actions),np.float32))
        assert result.truncated.all()
        np.testing.assert_allclose(env.backend.state.command_time_left, 1., atol=0)
        assert np.isfinite(result.obs['actor']).all()
        assert 'height_scan' not in env.observation_spec.names
    finally:env.close()


@pytest.mark.parametrize('robot',['go1','g1'])
def test_robot_controls_match_reference(robot):
    import re
    cfg=task_config(Experiment(robot=robot))
    expected=REFERENCE['profiles'][robot]
    assert set(cfg.robot.joint_names)==set(expected)
    for index,name in enumerate(cfg.robot.joint_names):
        reference=expected[name]
        for field in ('kp','kd','armature','effort_limit'):
            assert getattr(cfg.robot,field)[index]==pytest.approx(reference[field],rel=1e-6), (name,field)
        assert cfg.robot.default_joint_pos[index]==pytest.approx(reference['default_position'],abs=1e-6), name
        scales=[value for pattern,value in cfg.action_scale.items() if re.fullmatch(pattern,name)]
        assert len(scales)==1
        assert scales[0]==pytest.approx(reference['action_scale'],rel=1e-6),name


@pytest.mark.parametrize('robot', ['go1', 'g1'])
def test_external_commands_do_not_delay_observations_or_reset(robot):
    from gobot.rl.locomotion import make_velocity_env
    cfg=task_config(Experiment(robot=robot,stage='flat'))
    cfg.command.source='external'
    cfg.observations.actor_noise=cfg.domain_randomization.enabled=cfg.push_enabled=False
    cfg.randomize_reset_yaw=False
    cfg.spawn_jitter=0.
    cfg.reset_z_range=(.03,.03)
    env=make_velocity_env(cfg,backend='mujoco-cpu',num_envs=2,seed=19,max_episode_length=2)
    try:
        for step in range(2):
            env.set_command((.5,0.,0.))
            state=env.step(np.full((2,env.num_actions),.1,dtype=np.float32))
            physical=env.locomotion_state()
            from gobot.rl.locomotion.math import find_node_by_name
            imu_offset = np.asarray(find_node_by_name(env.robot, 'imu').position)
            expected = physical['base_linear_velocity_body'] + np.cross(
                physical['base_angular_velocity_body'], imu_offset)
            np.testing.assert_allclose(state.obs['actor'][:,:3], expected, atol=1e-6)
            np.testing.assert_allclose(state.obs['actor'][:,9:9+env.num_actions],
                physical['joint_position']-env.default_joint_pos,atol=1e-6)
        assert state.truncated.all()
        np.testing.assert_allclose(state.obs['actor'][:,:6],0.,atol=1e-6)
    finally:env.close()


def test_g1_locomotion_model_matches_reference_physics():
    """Mass, inertia, joints, contacts and actuators, beyond configuration equality."""
    import gobot
    import mujoco
    from examples.g1.prepare_tracking import audit_model
    from gobot.rl.locomotion.scene import configure_robot_drives
    cfg = task_config(Experiment(robot='g1'))
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(cfg.project_path))
        context.load_scene(cfg.robot_resource_path)
        configure_robot_drives(context.root, cfg)
        artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        reference = mujoco.MjModel.from_xml_path(str(ROOT/'tests/fixtures/mjlab_g1.xml'))
        assert (reference.nq, reference.nv, reference.nu) == (36, 35, 29)
        report = audit_model(reference, artifact, 'g1_')
        assert report['passed'], report['failed_checks']
        candidate = mujoco.MjModel.from_xml_string(artifact['content'])
        np.testing.assert_allclose(candidate.site('g1_pelvis_imu_site').pos,
                                   reference.site('imu_in_pelvis').pos, atol=1e-8)
        data = json.loads((Path(cfg.project_path)/'g1.jscn').read_text())
        meshes = [node for node in data['__NODES__'] if node['type'] == 'MeshInstance3D']
        assert len(meshes) == 35 and all(node['properties']['mesh'] for node in meshes)
        for resource in data['__EXT_RESOURCES__']:
            assert (Path(cfg.project_path)/resource['__PATH__'].removeprefix('res://')).is_file()
    finally:
        context.clear_scene()
