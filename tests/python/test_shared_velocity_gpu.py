"""CPU/Warp short stepping, observations and resets for velocity training."""
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.g1.config import g1_velocity_cfg
from gobot.rl.locomotion import make_velocity_env

def cfg():
    value = g1_velocity_cfg()
    value.observations.actor_noise = value.domain_randomization.enabled = value.push_enabled = False
    value.spawn_jitter = 0.
    value.reset_z_range = (.03, .03)
    value.randomize_reset_yaw = False
    value.command.heading_command = False
    value.command.rel_standing_envs = value.command.rel_heading_envs = value.command.rel_forward_envs = 0.
    value.command.ranges.lin_vel_x = (.5, .5)
    value.command.ranges.lin_vel_y = value.command.ranges.ang_vel_z = (0., 0.)
    return value

def collect(backend):
    env = make_velocity_env(cfg(), backend=backend, num_envs=2, seed=19)
    try:
        env.set_terrain_assignments([0, 0], [0, 0])
        obs, _ = env.reset(seed=19)
        initial = obs['actor'].detach().cpu().numpy() if backend.endswith('warp') else obs['actor'].copy()
        history = []
        for _ in range(4):
            state = env.step(np.zeros((2, 29), dtype=np.float32))
            data = env.locomotion_state()
            data["reward_terms"] = env._reward_terms if backend.endswith("warp") else env.backend.state.reward_terms
            history.append({name: value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else value.copy() for name, value in data.items()})
        if backend.endswith('warp'):
            model = env.provider
            pointer = model.arrays['qpos'].data_ptr()
            env.reset([1], seed=19)
            assert model.arrays['qpos'].data_ptr() == pointer
            ids = list(env._layout.actuator_ids)
            np.testing.assert_allclose(model.arrays['ctrl'][1, ids].cpu().numpy(), 0.)
            model.assert_finite()
            model.assert_no_overflow()
        return initial, history
    finally:
        env.close()

def standard_velocity_backends(robots=('go1', 'g1')):
    from examples.locomotion.presets import Experiment, task_config

    for robot in robots:
        traces=[]
        for backend in ('mujoco-cpu', 'mujoco-warp'):
            config=task_config(Experiment(robot=robot,stage='flat'))
            config.observations.actor_noise=config.domain_randomization.enabled=config.push_enabled=False
            config.spawn_jitter=0.;config.reset_z_range=(.03,.03);config.randomize_reset_yaw=False
            config.command.source='external';config.command_curriculum=()
            env=make_velocity_env(config,backend=backend,num_envs=2,seed=19,max_episode_length=4)
            def array(value):
                return value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value).copy()
            try:
                env.set_terrain_assignments([0,0],[0,0],fixed=True)
                env.reset(seed=19)
                trace=[]
                for step in range(4):
                    env.set_command((.5,0.,0.))
                    state=env.step(np.full((2,env.num_actions),.01*step,dtype=np.float32))
                    physical=env.locomotion_state()
                    trace.append({**{key:array(physical[key]) for key in ('base_position','joint_position','joint_velocity')},
                                  'actor':array(state.obs['actor']), 'critic':array(state.obs['critic']),
                                  'reward':array(state.reward),'terminated':array(state.terminated),'truncated':array(state.truncated)})
                if backend=='mujoco-warp':
                    env.provider.assert_finite();env.provider.assert_no_overflow()
                traces.append(trace)
            finally:env.close()
        errors={}
        for cpu,gpu in zip(*traces,strict=True):
            for key in cpu:
                if cpu[key].dtype==bool:
                    np.testing.assert_array_equal(cpu[key],gpu[key])
                else:
                    # Contact impulses can differ between solvers; test numerical
                    # closeness, not long-horizon trajectory or policy equivalence.
                    tolerance=1e-3 if key=='reward' else 2e-4
                    np.testing.assert_allclose(cpu[key],gpu[key],rtol=2e-4,atol=tolerance,err_msg=f'{robot}/{key}')
                    errors[key]=max(errors.get(key,0.),float(np.abs(cpu[key]-gpu[key]).max()))
        print(robot,'standard CPU/Warp four-step errors',errors,flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--robot', choices=('go1','both'), default='both')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(77)
    if args.robot=='go1':
        standard_velocity_backends(('go1',))
        return
    cpu_obs, cpu = collect('mujoco-cpu')
    gpu_obs, gpu = collect('mujoco-warp')
    np.testing.assert_allclose(cpu_obs, gpu_obs, atol=2e-4, rtol=2e-4)
    for step, (frame_cpu, frame_gpu) in enumerate(zip(cpu, gpu, strict=True)):
        print("step", step, {name: float(np.max(np.abs(frame_cpu[name] - frame_gpu[name]))) for name in ("base_position", "joint_position", "joint_velocity", "angular_momentum")}, flush=True)
        for name in ('base_position', 'base_quaternion', 'joint_position', 'joint_velocity', 'angular_momentum', 'reward_terms'):
            np.testing.assert_allclose(frame_cpu[name], frame_gpu[name], atol=2e-3, rtol=2e-3, err_msg=name)
    print('G1 CPU/Warp reset observations and short-horizon state parity passed')
    standard_velocity_backends()

if __name__ == '__main__':
    main()
