"""Compare sampling with the pinned mjlab implementation, without a GPU import."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import numpy as np
import pytest

from examples.locomotion.presets import Experiment, task_config
from gobot.rl.locomotion.warp_env import WarpVelocityEnv

ROOT=Path(__file__).resolve().parents[2]/'mjlab/src/mjlab'


def reference_method(file, class_name, name, namespace):
    tree=ast.parse((ROOT/file).read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name) if class_name else tree
    fn=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    namespace={'torch':torch,'math':math,**namespace}
    exec(compile(ast.fix_missing_locations(module),str(ROOT/file),'exec'),namespace)
    return namespace[name]


def probe(count=19):
    env=WarpVelocityEnv.__new__(WarpVelocityEnv)
    env.cfg_obj=task_config(Experiment(robot='g1'))
    env.num_envs=count;env._torch_device=torch.device('cpu');env._generator=torch.default_generator
    env.step_dt=.02
    env._command=torch.zeros(count,3);env._command_world=torch.zeros_like(env._command)
    env._command_time_left=torch.zeros(count);env._heading_target=torch.zeros(count)
    for key in ('standing','heading','world','forward','run'):
        setattr(env,f'_is_{key}_env',torch.zeros(count,dtype=torch.bool))
    env._command_mode=torch.zeros(count,dtype=torch.long);env._command_counter=torch.zeros(count,dtype=torch.long)
    return env


def test_command_sampling_and_rng_match_reference_with_partial_resampling():
    env=probe();ids=torch.tensor([1,3,6,11,15])
    ref=NS(device="cpu",cfg=env.cfg_obj.command,vel_command_b=torch.zeros(19,3),vel_command_w=torch.zeros(19,3),heading_target=torch.zeros(19),
        is_heading_env=torch.zeros(19,dtype=torch.bool),is_standing_env=torch.zeros(19,dtype=torch.bool),
        is_world_env=torch.zeros(19,dtype=torch.bool),is_forward_env=torch.zeros(19,dtype=torch.bool),
        time_left=torch.zeros(19),command_counter=torch.zeros(19,dtype=torch.long))
    sample=reference_method(Path('tasks/velocity/mdp/velocity_command.py'),'UniformVelocityCommand','_resample_command',{})
    ref._resample_command=lambda ids:sample(ref,ids)
    resample=reference_method(Path('managers/command_manager.py'),'CommandTerm','_resample',{})
    torch.manual_seed(42);resample(ref,ids);expected_rng=torch.get_rng_state()
    torch.manual_seed(42);env._resample_commands(ids)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for actual,expected in [(env._command,ref.vel_command_b),(env._command_world,ref.vel_command_w),
        (env._command_time_left,ref.time_left),(env._heading_target,ref.heading_target),
        (env._is_heading_env,ref.is_heading_env),(env._is_forward_env,ref.is_forward_env),
        (env._is_standing_env,ref.is_standing_env),(env._command_counter,ref.command_counter)]:
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_uniform_events_use_reference_shapes_and_consume_zero_width_draws():
    env=probe()
    sample=reference_method(Path('utils/lab_api/math.py'),None,'sample_uniform',{})
    calls=[((19,1,1),.3,1.2),((19,29),-.015,.015),((19,6),0.,0.),((19,29),0.,0.)]
    torch.manual_seed(42);expected=[sample(torch.tensor(lo),torch.tensor(hi),shape,'cpu') for shape,lo,hi in calls];rng=torch.get_rng_state()
    torch.manual_seed(42);actual=[env._sample_uniform(shape,lo,hi) for shape,lo,hi in calls]
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('count', [3, 7, 64, 1024])
def test_initial_terrain_assignments_and_rng_match_native_reference(count):
    env = probe(count)
    env._spawn_rows, env._spawn_cols = 10, 7
    origins = torch.arange(210, dtype=torch.float32).reshape(10, 7, 3)
    env._spawn_origins = origins.reshape(-1, 3)
    env._all_env_ids = torch.arange(count)
    proportions = np.array([.1, .2, .1, .15, .2, .15, .1])
    env._terrain_config = {'sub_terrains': [{'proportion': p} for p in proportions]}
    ref = NS(cfg=NS(max_init_terrain_level=env.cfg_obj.max_init_terrain_level), _device='cpu')
    file = Path('terrains/terrain_entity.py')
    allocation = reference_method(file, None, '_proportional_counts', {'np': np})
    initialize = reference_method(file, 'TerrainEntity', '_compute_env_origins_curriculum',
                                  {'np': np, '_proportional_counts': allocation})
    torch.manual_seed(42)
    expected = initialize(ref, count, origins, proportions)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    env._initialize_terrain_assignments()
    assert torch.equal(torch.get_rng_state(), rng)
    for actual, target in ((env._terrain_levels, ref.terrain_levels),
                           (env._terrain_types, ref.terrain_types), (env._env_origins, expected)):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)


def test_push_samples_next_timer_then_one_six_axis_kick_without_mutating_live_velocity():
    env=probe(4);env._push_time_left=torch.tensor([0.,1.,.01,2.]);env._push_count=torch.zeros(4,dtype=torch.long)
    env._arrays={'qpos':torch.tensor([[0.,0.,1.,1.,0.,0.,0.]]).repeat(4,1),'qvel':torch.zeros(4,6)}
    linear=torch.ones(4,3);angular=torch.zeros(4,3)
    env._root_link_velocity_world=lambda:(linear,angular)
    ids=torch.tensor([0,2]);ranges=env.cfg_obj.push_velocity_ranges
    sample=reference_method(Path('utils/lab_api/math.py'),None,'sample_uniform',{})
    se3=reference_method(Path('envs/mdp/events.py'),None,'_sample_se3_range',{'_SE3_KEYS':('x','y','z','roll','pitch','yaw'),'sample_uniform':sample})
    torch.manual_seed(42)
    timer=torch.rand(2)*2+1
    kick=se3(ranges,(2,6),'cpu');expected_rng=torch.get_rng_state()
    torch.manual_seed(42);env._apply_pushes()
    torch.testing.assert_close(env._push_time_left[ids],timer,rtol=0,atol=0)
    torch.testing.assert_close(env._arrays['qvel'][ids],torch.cat((linear[ids],angular[ids]),1)+kick,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    torch.testing.assert_close(linear,torch.ones_like(linear),rtol=0,atol=0)
    torch.testing.assert_close(angular,torch.zeros_like(angular),rtol=0,atol=0)
    assert env._push_count.tolist()==[1,0,1,0]


def test_cpu_reset_clears_targets_like_reference_entity_reset():
    from gobot.rl.locomotion import make_velocity_env
    cfg=task_config(Experiment(robot='g1',stage='flat'),play=True)
    env=make_velocity_env(cfg,backend='mujoco-cpu',num_envs=2,collect_step_extras=False)
    try:
        original=env.backend.reset_robot_states
        calls=[]
        def capture(*args,**kwargs):
            calls.append(kwargs['joint_position_targets'].copy())
            return original(*args,**kwargs)
        env.backend.reset_robot_states=capture
        env.reset(seed=17)
        assert len(calls)==1
        torch.testing.assert_close(torch.tensor(calls[0]), torch.zeros(2,env.num_actions))
    finally:env.close()
