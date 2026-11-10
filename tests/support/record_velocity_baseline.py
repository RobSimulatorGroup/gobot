"""Record native training behavior in separate Gobot and pinned mjlab processes.

Both processes use the same authored terrain, seed and action sequence. No
runtime states or RNG states are copied between implementations. The probe
exercises startup randomization, noisy observations, command resampling, pushes,
partial resets, timeout/failure overlap and curriculum boundaries.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'dcdf6ce519abfd2ea45e9503ae6fb650b75bb869'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', choices=('gobot', 'mjlab'), required=True)
    p.add_argument('--terrain', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--updates', type=int, default=0)
    p.add_argument('--robot', choices=('g1', 'go1'), default='g1')
    args = p.parse_args()
    if args.num_envs < 8 or args.updates < 0:
        p.error('the event probe needs at least eight environments and nonnegative updates')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mujoco_warp._src import collision_convex
    collision_convex.MJ_MAX_EPAHORIZON = 64
    reference = args.runtime == 'mjlab'
    from mjlab_training import training_terrain, helper
    terrain_data = helper('terrain_reference').load(args.terrain)
    if reference:
        assert subprocess.check_output(['git', '-C', str(ROOT/'mjlab'), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
        assert not subprocess.check_output(['git', '-C', str(ROOT/'mjlab'), 'diff', '--name-only', 'HEAD', '--', 'src'], text=True).strip()
        sys.path.insert(0, str(ROOT/'mjlab/src'))
        tasks = types.ModuleType('mjlab.tasks')
        tasks.__path__ = [str(ROOT/'mjlab/src/mjlab/tasks')]
        sys.modules['mjlab.tasks'] = tasks
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        if args.robot == 'g1':
            from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg as factory
            from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg as agent_factory
        else:
            from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg as factory
            from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg as agent_factory
        cfg = factory()
        cfg.seed = 42
        cfg.scene.num_envs = args.num_envs
        cfg.sim.njmax = 1500
        cfg.sim.nconmax = 70
        with training_terrain(cfg, terrain_data):
            env = ManagerBasedRlEnv(cfg, device='cuda:0')
        initial_rng = torch.cuda.get_rng_state()
        wrapped = RslRlVecEnvWrapper(env)
        agent = asdict(agent_factory())
        term = env.command_manager.get_term('twist')
        qpos, qvel, ctrl = env.sim.data.qpos, env.sim.data.qvel, env.sim.data.ctrl
    else:
        from gobot.rl.locomotion import make_velocity_env
        from gobot.rl import RslRlVecEnvWrapper
        from examples.locomotion.presets import Experiment, task_config
        from gobot.rl.locomotion.training import agent_config
        cfg = task_config(Experiment(robot=args.robot))
        env = make_velocity_env(cfg, backend='mujoco-warp', device='cuda:0', num_envs=args.num_envs,
                                seed=42, reset_on_create=False, collect_step_extras=False)
        helper('terrain_reference').verify_environment(env, terrain_data)
        initial_rng = torch.cuda.get_rng_state()
        env.reset()
        wrapped = RslRlVecEnvWrapper(env, device='cuda:0')
        agent = agent_config(Experiment(robot=args.robot))
        qpos, qvel, ctrl = (env.provider.arrays[k] for k in ('qpos', 'qvel', 'ctrl'))

    def snapshot():
        obs = wrapped.get_observations()
        values = dict(actor=obs['actor'], critic=obs['critic'], qpos=qpos, qvel=qvel, ctrl=ctrl,
                      episode_length=env.episode_length_buf,
                      rng=torch.cuda.get_rng_state(), cpu_rng=torch.get_rng_state())
        if reference:
            terrain = env.scene.terrain
            values.update(command=term.vel_command_b, command_time=term.time_left,
                          displacement=term.commanded_displacement_w,
                          levels=terrain.terrain_levels, types=terrain.terrain_types,
                          bias=env.scene['robot'].data.encoder_bias,
                          heading=term.heading_target,
                          push_time=env.event_manager._interval_term_time_left[0])
        else:
            values.update(command=env._command, command_time=env._command_time_left,
                          displacement=env._commanded_displacement,
                          levels=env._terrain_levels, types=env._terrain_types,
                          bias=env._encoder_bias, heading=env._heading_target,
                          push_time=env._push_time_left)
        return {k:v.detach().cpu().clone() for k,v in values.items()}

    import mujoco
    model = env.sim.mj_model if reference else env.provider._mj_model
    mujoco.mj_saveModel(model, str(args.output.with_suffix('.mjb')))
    # Actuator array order differs between independently authored models.
    actuator_order = torch.as_tensor(model.actuator_trnid[:, 0].argsort(), device='cuda:0')
    raw_snapshot = snapshot
    def snapshot():
        result = raw_snapshot()
        result['ctrl'] = ctrl[:, actuator_order].detach().cpu().clone()
        return result

    frames = [snapshot()]
    try:
        with torch.inference_mode(args.updates == 0):
            if args.updates:
                if reference:
                    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
                    runner = VelocityOnPolicyRunner(wrapped, agent, device='cuda:0')
                    alg = runner.alg
                else:
                    from gobot.rl.locomotion.training import build_algorithm
                    from gobot.rl.locomotion.distributed import DistributedTraining
                    alg = build_algorithm(wrapped, agent, 'cuda:0', DistributedTraining('cuda:0'))
                initial_algorithm = deepcopy(alg.save())
                # Native runners randomize episode lengths after constructing PPO.
                wrapped.episode_length_buf = torch.randint_like(wrapped.episode_length_buf, high=wrapped.max_episode_length)
                original = wrapped.step
                def record(action):
                    result = original(action)
                    frame = snapshot()
                    frame.update(action=action.cpu().clone(), reward=result[1].cpu().clone(),
                                 done=result[2].bool().cpu().clone(), timeout=result[3]['time_outs'].cpu().clone())
                    frames.append(frame)
                    return result
                wrapped.step = record
                if reference:
                    runner.learn(args.updates, init_at_random_ep_len=False)
                else:
                    from gobot.rl.locomotion.training import update_policy
                    obs = wrapped.get_observations()
                    alg.train_mode()
                    for _ in range(args.updates):
                        obs, _ = update_policy(alg, wrapped, obs, agent['num_steps_per_env'])
            else:
                for step in range(64):
                    if step in (16, 32, 48):
                        env.reset()
                        if not reference:
                            wrapped._sync_steps_from_core()
                    if step == 4:
                        (term.time_left if reference else env._command_time_left)[:4] = 0.
                    if step == 8:
                        (env.event_manager._interval_term_time_left[0] if reference else env._push_time_left)[4:8] = 0.
                    if step in (12, 40):
                        wrapped.episode_length_buf[:4] = env.max_episode_length - 1
                        qpos[1, 3:7] = torch.tensor([0., 1., 0., 0.], device='cuda:0')
                    if step == 20:
                        env.reset(env_ids=torch.tensor([2, 5], device='cuda:0'))
                        if not reference:
                            wrapped._sync_steps_from_core()
                    if step == 30:
                        env.common_step_counter = 120000
                        env.reset(env_ids=torch.tensor([0, 3], device='cuda:0'))
                        if not reference:
                            wrapped._sync_steps_from_core()
                    action = .1 * torch.sin(torch.arange(env.num_envs * wrapped.num_actions, device='cuda:0').reshape(env.num_envs, -1) * .1 + step * .2)
                    _, reward, done, extras = wrapped.step(action)
                    frame = snapshot()
                    frame.update(reward=reward.cpu().clone(), done=done.bool().cpu().clone(), timeout=extras['time_outs'].cpu().clone())
                    frames.append(frame)
        payload = dict(runtime=args.runtime, robot=args.robot, revision=REVISION,
                       versions={name: importlib.metadata.version(name) for name in ('torch', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib')},
                       protocol='training' if args.updates else 'events_with_full_resets_16_32_48',
                       terrain_sha256=terrain_data['sha256'],
                       initialization_rng=initial_rng, frames=frames)
        if args.updates:
            payload.update(initial_algorithm=initial_algorithm, algorithm=alg.save())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
        print(f'Recorded {len(frames)} frames: {args.output}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
