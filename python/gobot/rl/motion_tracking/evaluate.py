"""Frozen teacher inference and full-clip scoring shared by port comparisons."""
from __future__ import annotations
import torch
from gobot.rl.supermimic import networks
from gobot.rl.supermimic.geometry import unflatten_state
from gobot.rl.supermimic.contracts import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY


def load_policy(env, checkpoint, device=None):
    mdp, args = env.supermimic_mdp, checkpoint['args']
    if int(args['goal_frames']) != env.cfg.goal_frames:
        raise ValueError('checkpoint future-goal protocol differs from the environment')
    networks.AMP_DTYPE = torch.bfloat16 if args.get('amp', 'none') == 'bf16' else None
    policy = networks.Policy(mdp.obs_dim, mdp.goal_frame_dim*int(args['goal_frames']),
        hist=int(args['hist']), z_dim=int(args['z_dim']), num_joints=mdp.num_joints,
        hidden=tuple(args.get('test_hidden_dims') or args['policy_hidden'])).to(device or env.device)
    policy.load_state_dict(checkpoint['pi'], strict=True)
    return policy.eval()


@torch.inference_mode()
def evaluate(env, checkpoint, *, max_steps=None, progress=None, start_offset=0, device=None):
    """Each lane evaluates one clip; early truncation cannot count as completion."""
    mdp, args = env.supermimic_mdp, checkpoint['args']
    motion = mdp.motion
    count = len(motion.names)
    if env.num_envs != count:
        raise ValueError('evaluation requires exactly one environment per clip')
    if max_steps is not None and max_steps <= 0 or start_offset < 0:
        raise ValueError('invalid evaluation duration or start offset')
    frames = motion.clip_start+start_offset
    if bool((frames>=motion.clip_end).any()):
        raise ValueError('evaluation offset exceeds one or more clips')
    policy = load_policy(env, checkpoint, device)
    policy_device = next(policy.parameters()).device
    _, info = env.reset_to_frames(frames)
    current = info[SUPERMIMIC_STATE_KEY]
    returns = torch.zeros(count, device=env.device)
    steps = torch.zeros(count, dtype=torch.long, device=env.device)
    done = torch.zeros(count, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(done)
    history = None
    iteration = 0
    while not bool(done.all()):
        obs = mdp.obs(unflatten_state(current['s']),current['a_prev'],current['xi'],current['f'])
        history = obs[:, None].repeat(1,int(args['hist']),1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        goal = mdp.goal(current['f'],int(args['goal_frames']))
        action = policy(history.to(policy_device),goal.to(policy_device)).to(env.device)
        state = env.step(action)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        alive = ~done
        if not bool(torch.isfinite(transition['s1']).all() and torch.isfinite(transition['r']).all()):
            raise FloatingPointError('non-finite physical state or tracking reward')
        returns += torch.where(alive,transition['r'],0.)
        steps += alive
        failed |= alive & transition['term']
        done |= alive & transition['done']
        current = state.info[SUPERMIMIC_STATE_KEY]
        iteration += 1
        if progress and (iteration % 100 == 0 or bool(done.all())):
            progress(iteration, steps, failed)
        if max_steps is not None and iteration >= max_steps:
            break
    expected = motion.clip_len-1-start_offset
    normalized = returns / (expected.float()*mdp.max_step_reward)
    complete = (steps>=expected)&~failed
    per_clip = {name:{'norm_return':float(normalized[i]), 'steps':int(steps[i]),
        'expected_steps':int(expected[i]),'failed':bool(failed[i]),'full_clip':bool(complete[i]) and start_offset==0,
        'window_complete':bool(complete[i]),'failure_time_s':(start_offset+int(steps[i]))*.02 if failed[i] else None}
        for i,name in enumerate(motion.names)}
    return {'protocol':'g1_tracking_deterministic_v1' if env.cfg.deterministic else 'g1_tracking_v1','start_offset':start_offset,'screening':max_steps is not None or start_offset!=0,
            'norm_return':float(normalized.mean()),'norm_return_framew':float(returns.sum()/(expected.sum()*mdp.max_step_reward)),
            'n_failed':int(failed.sum()),'n_full_clips':int(complete.sum()) if start_offset==0 else 0,
            'frame_coverage':float(steps.sum()/expected.sum()), 'per_clip':per_clip}
