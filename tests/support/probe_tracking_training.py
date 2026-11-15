"""Audit actual native/Gobot runner startup, rollout, GAE and PPO updates.

The mjlab branch imports no Gobot. Run in separate processes with the same
numerical packages. Instrumentation copies tensors but never draws randomness.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
# Resolve numerical dependencies before optional upstream viewer packages.
import rsl_rl.runners
import mujoco
import mujoco_warp
import warp
import scipy


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def rng(device):
    return dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(device).clone())


def probe(args):
    for path in args.extra_python_path:
        sys.path.append(str(path.resolve()))
    device = args.device
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    out = args.output.resolve()
    if world > 1:
        out = out / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    seed = args.seed + rank
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trace = {'engine': args.engine, 'seed': seed, 'num_envs': args.num_envs, 'updates': [],
             'deterministic_physics': args.deterministic_physics,
             'self_contact_reduction': args.self_contact_reduction,
             'no_self_collision': args.no_self_collision, 'replay': str(args.replay) if args.replay else None}
    replay = args.replay
    if replay and replay.is_dir():
        replay = replay / f"rank{rank}" / 'trace.pt'
    golden = torch.load(replay, map_location='cpu', weights_only=False) if replay else None
    if golden is not None and (golden['num_envs'] != args.num_envs or len(golden['updates']) < args.updates):
        raise ValueError('replay batch size/update count differs')
    def device_obs(obs):
        from tensordict import TensorDict
        return TensorDict({k:v.clone().to(device) for k,v in obs.items()}, batch_size=[args.num_envs])
    def restore_rng(value):
        torch.set_rng_state(value['cpu'])
        torch.cuda.set_rng_state(value['cuda'], device)

    group = None
    if args.engine == 'mjlab':
        from mjlab_tracking_reference import reference, configure_deterministic_physics
        reference(args.source)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
        from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
        # MotionTrackingOnPolicyRunner inherits this initialization/learn loop;
        # its overrides only export/upload at save, disabled by this probe.
        from mjlab.rl import MjlabOnPolicyRunner
        cfg = unitree_g1_flat_tracking_env_cfg()
        cfg.seed = seed
        cfg.scene.num_envs = args.num_envs
        cfg.sim.nconmax, cfg.sim.njmax = 128, 500
        cfg.scene.sensors[0].reduce = args.self_contact_reduction
        cfg.commands['motion'].motion_file = str(args.motion.resolve())
        configure_deterministic_physics(cfg.sim, args.deterministic_physics)
        env = ManagerBasedRlEnv(cfg, device=device)
        wrapper = RslRlVecEnvWrapper(env)
        factory = MjlabOnPolicyRunner
        agent = asdict(unitree_g1_tracking_ppo_runner_cfg())
        agent['seed'] = seed
        def state():
            cmd = env.command_manager.get_term('motion')
            return dict(frames=cpu(cmd.time_steps), episode_length=cpu(env.episode_length_buf),
                        sampler=cpu(cmd.bin_failed_count), qpos=cpu(env.sim.data.qpos[:]),
                        qvel=cpu(env.sim.data.qvel[:]))
    else:
        from gobot.rl.tracking.training import Config, Runner, contract
        from gobot.rl.tracking.distributed import DistributedTraining
        from gobot.rl.tracking.env import TrackingEnv
        from gobot.rl.training.workflow import update_config
        from gobot.rl.rsl_rl import RslRlVecEnvWrapper
        group = DistributedTraining.from_environment(device)
        group.initialize()
        cfg = update_config(Config(), json.loads((args.project/'task.json').read_text()))
        cfg.project_path = args.project.resolve()
        cfg.num_envs, cfg.seed, cfg.device = args.num_envs, seed, device
        cfg.deterministic_physics = args.deterministic_physics
        cfg.self_contact_reduction = args.self_contact_reduction
        env = TrackingEnv(cfg)
        wrapper = RslRlVecEnvWrapper(env, device=device)
        agent = deepcopy(cfg.agent)
        agent['seed'] = seed
        def factory(wrapper, agent, output, device):
            return Runner(wrapper, agent, output, device, 'diagnostic', contract(env), group)
        def state():
            s=env.batch.read_state()
            return dict(frames=cpu(env.frame), episode_length=cpu(env.steps), sampler=cpu(env.sampler.failed),
                        base=cpu(s.base_pose), joints=cpu(s.joint_position))
    agent['num_steps_per_env'] = args.steps
    agent['logger'] = 'tensorboard'
    agent['upload_model'] = False
    if args.no_self_collision:
        # An explicitly labeled diagnostic ablation, never the production reward.
        if args.engine == 'mjlab':
            env.reward_manager._term_cfgs[env.reward_manager.active_terms.index('self_collisions')].weight = 0.
        else:
            env.cfg_obj.reward_weights['self_collisions'] = 0.
    if golden is not None:
        wrapper.get_observations = lambda: device_obs(golden['after_reset']['obs'])
    trace['after_reset'] = dict(rng=rng(device), obs=cpu(wrapper.get_observations()), state=state())
    runner = factory(wrapper, agent, str(out/'runner'), device)
    alg = runner.alg
    if golden is not None:
        alg.load(golden["after_runner"]["models"], None, True)
    trace['after_runner'] = dict(rng=rng(device), models=cpu(alg.save()), state=state(), config=cpu(agent))
    update = {'acts': [], 'steps': [], 'gradients': []}
    old_act, old_step = alg.act, wrapper.step
    old_returns, old_update = alg.compute_returns, alg.update
    old_opt_step = alg.optimizer.step
    def act(obs):
        if not update['acts']:
            if golden is not None:
                restore_rng(golden['updates'][len(trace['updates'])]['start']['rng'])
            update['start'] = dict(rng=rng(device), state=state(), obs=cpu(obs), models=cpu(alg.save()))
        before = rng(device)
        action = old_act(obs)
        update['acts'].append(dict(obs=cpu(obs), rng=before, action=cpu(action), values=cpu(alg.transition.values),
                                   log_prob=cpu(alg.transition.actions_log_prob), distribution=cpu(alg.transition.distribution_params)))
        return action
    def step(action):
        if golden is not None:
            record = golden['updates'][len(trace['updates'])]['steps'][len(update['steps'])]
            restore_rng(record['rng'])
            result = (device_obs(record['obs']), record['rewards'].clone().to(device),
                      record['dones'].clone().to(device), {'time_outs':record['time_outs'].clone().to(device)})
            update['steps'].append(cpu(record))
            return result
        result = old_step(action)
        obs, reward, done, extras = result
        terms = env.reward_manager.term_rewards if args.engine == 'mjlab' else torch.stack(tuple(env.state.info['reward_terms'].values()),-1)
        update['steps'].append(dict(obs=cpu(obs), rewards=cpu(reward), dones=cpu(done),
                time_outs=cpu(extras['time_outs']), rng=rng(device), state=state(), terms=cpu(terms)))
        return result
    def returns(obs):
        old_returns(obs)
        storage=alg.storage
        update['storage']={k:cpu(getattr(storage,k)) for k in (
            'observations','actions','rewards','dones','values','returns','advantages','actions_log_prob','distribution_params')
            if hasattr(storage,k)}
        update['pre_update_rng']=rng(device)
    def optimizer_step(*a,**kw):
        # Save each minibatch's clipped gradient and parameter state. Only a few
        # diagnostic updates are intended; production training is untouched.
        names=[('actor.'+n,p) for n,p in alg.actor.named_parameters()]+[('critic.'+n,p) for n,p in alg.critic.named_parameters()]
        update['gradients'].append(dict(lr=alg.learning_rate, values={n:cpu(p.grad) for n,p in names}))
        return old_opt_step(*a,**kw)
    def train_update():
        result=old_update()
        update['losses']=cpu(result)
        update['post_update']=cpu(alg.save())
        update['post_update_rng']=rng(device)
        trace['updates'].append(deepcopy(update))
        update.clear(); update.update(acts=[],steps=[],gradients=[])
        print('Probe update completed',len(trace['updates']),flush=True)
        return result
    alg.act, wrapper.step = act, step
    alg.compute_returns, alg.update = returns, train_update
    alg.optimizer.step = optimizer_step
    runner.save = lambda *a, **kw: None
    try:
        runner.learn(args.updates, init_at_random_ep_len=True)
        torch.save(trace,out/'trace.pt')
        print('Probe saved:',out,flush=True)
    finally:
        env.close()
        if group: group.close()
        elif world>1 and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    p.add_argument('--engine',choices=['mjlab','gobot'],required=True)
    p.add_argument('--source',type=Path,default=root/'mjlab')
    p.add_argument('--project',type=Path,default=root/'logs/tracking-comparison/gobot-aligned-scene')
    p.add_argument('--motion',type=Path,default=root/'logs/tracking-comparison/dance2-reference/motion.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-python-path',type=Path,action='append',default=[])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--replay',type=Path,help='Replay exactly the recorded observations, rewards, masks and external RNG stream to isolate PPO')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--updates',type=int,default=1)
    p.add_argument('--no-self-collision',action='store_true')
    p.add_argument('--deterministic-physics',action='store_true')
    p.add_argument('--self-contact-reduction',choices=('none','maxforce'),default='none')
    probe(p.parse_args())
