"""Export independent official CPU expectations without importing Gobot.

All reference functions execute from the verified checkouts. torch.compile is
disabled in this process for a deterministic, eager FP32 math comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .reference import LOCK, audit, numerical_packages, sha256


def generate(reference: Path, mjlab: Path, output: Path):
    report = audit(reference, mjlab)
    numerical_packages(reference, mjlab)
    import torch
    from tensordict import TensorDict
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.compile = lambda fn, *args, **kwargs: fn
    from supermimic.representations import create_representation, goal_window_indices
    from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
    from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
    from supermimic.learners.policy_learner import ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg
    from supermimic.learners.learner_utils import _optimize_step
    from supermimic.tasks.task_spec import TaskSpec
    from supermimic.replay_buffer import FieldSpec, ReplayBuffer
    from supermimic.utils.tracking_reward import tracking_reward
    from mjlab.utils.lab_api.math import matrix_from_quat
    from .probes import DifferentiableProbeEnv
    from types import SimpleNamespace

    def snapshot(module):
        return {k: v.detach().clone() for k, v in module.state_dict().items()}
    def grads(module):
        return {k: v.grad.detach().clone() for k, v in module.named_parameters() if v.grad is not None}
    def detach(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: detach(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(detach(v) for v in value)
        return value

    torch.manual_seed(6124)
    b, j = 4, 29
    default = torch.linspace(-0.2, 0.2, j)
    states = torch.randn(b, 33, 71) * 0.3
    states[..., 2] += 0.8
    states[..., 3:7] /= states[..., 3:7].norm(dim=-1, keepdim=True)
    actions = torch.randn(b, 33, j) * 0.2
    context = torch.rand(b, 33) * 2 - 1
    scale = torch.linspace(0.1, 0.8, j)
    rep = create_representation('gencoord', 71, j, 0.02, 4, default, 'rep_state')
    dynamics = Dynamics(rep, j, 33, scale, default, (16, 12, 8))
    learner = WorldModelLearner(dynamics, rep, WorldModelLearnerCfg())
    learner.update_normalization(states, actions)
    initial = snapshot(dynamics)
    fixture = {'default': default, 'states': states, 'actions': actions, 'context': context,
               'action_scale': scale, 'wm_initial': initial, 'wm': {}}
    for horizon in (1, 8, 32):
        dynamics.load_state_dict(initial)
        dynamics.eval()
        dynamics.zero_grad(set_to_none=True)
        act = actions[:, :horizon+1].clone().requires_grad_(True)
        source = states[:, :horizon+1]
        representations = rep.represent(source)
        aux = rep.extract_aux_state(source)
        pred, pred_aux = learner._rollout(representations[:, 0], act, aux[:, 0], context)
        loss, groups = learner._compute_losses(pred, pred_aux, representations[:, 1:], aux[:, 1:])
        loss.backward()
        fixture['wm'][str(horizon)] = detach({
            'states': representations, 'predicted': pred, 'aux': pred_aux,
            'reconstructed': rep.reconstruct_full_state(pred_aux, pred),
            'loss': loss, 'groups': groups, 'grads': grads(dynamics), 'action_grad': act.grad,
        })
    dynamics.load_state_dict(initial)
    dynamics.eval()
    source = states[:, :8]  # The official horizon=8 contains 8 rows and scores 7 predictions.
    act = actions[:, :8].clone().requires_grad_(True)
    pred, aux = learner._rollout(rep.represent(source)[:, 0], act, source[:, 0, :7], context)
    loss, _ = learner._compute_losses(pred, aux, rep.represent(source)[:, 1:], source[:, 1:, :7])
    norm = _optimize_step(learner.optimizers, loss, list(dynamics.parameters()), max_norm=1.)
    fixture['wm_update'] = detach({'loss': loss, 'norm': norm, 'parameters': snapshot(dynamics),
                                   'grads': grads(dynamics), 'optimizers': [o.state_dict() for o in learner.optimizers]})

    # Strided goal windows include the first/last frames of two adjacent clips.
    motion = SimpleNamespace(motion_ends=torch.tensor([4, 60]),
        joint_pos=torch.randn(60, j), joint_vel=torch.randn(60, j),
        root_pos_w=torch.randn(60, 3), root_quat_w=torch.randn(60, 4))
    motion.root_quat_w /= motion.root_quat_w.norm(dim=-1, keepdim=True)
    times = torch.tensor([0, 3, 4, 59])
    wm_goal, actor_goal = rep.compute_wm_goal(motion, times, states[:, 0, :7])
    fixture['goal'] = detach({'motion': vars(motion), 'times': times,
                             'indices': goal_window_indices(motion, times),
                             'world_model': wm_goal, 'actor': actor_goal})
    fixture['proprio'] = rep.compute_wm_obs(rep.represent(states[:, 0]), states[:, 0, :7],
                                           rep.joint_state(rep.represent(states[:, 0])), context)

    # Independent synthetic policy update, including non-anchor latent BC.
    specs = [TaskSpec('motion', 640, lambda *a: None), TaskSpec('velocity', 3, lambda *a: None)]
    starts = TensorDict({'state': rep.represent(states[:, 0]), 'goal_motion': actor_goal.detach(),
                        'goal_velocity': torch.randn(b, 3), 'context': context}, batch_size=[b])
    fixture['policy_starts'] = starts.to_dict()
    fixture['policy'] = {}
    for horizon in (1, 8, 32):
        actor = MultiEncoderAE(68, 29, specs, {'motion': 0.5, 'velocity': 0.5}, 32, (16, 12, 8), .1, 1)
        env = DifferentiableProbeEnv(starts)
        obs = env.reset(starts)
        actor.update_normalization(obs)
        critic = build_mlp(68+640+33, (12, 8), 1)
        cfg = ActorCriticPolicyLearnerCfg(horizon=horizon, batch_size=b)
        pl = ActorCriticPolicyLearner(actor, critic, env, cfg)
        entry = {'actor': snapshot(actor), 'critic': snapshot(critic), 'dynamics': snapshot(env.dynamics),
                 'action': actor(obs), 'rng': torch.get_rng_state()}
        entry['metrics'] = pl.update()
        entry.update(actor_after=snapshot(actor), critic_after=snapshot(critic),
                     actor_grads=grads(actor), critic_grads=grads(critic),
                     actor_optimizers=[o.state_dict() for o in pl.actor_optimizers],
                     critic_optimizers=[o.state_dict() for o in pl.critic_optimizers],
                     dynamics_grad_none=all(p.grad is None for p in env.dynamics.parameters()))
        fixture['policy'][str(horizon)] = detach(entry)

    # Ring wrap, termination and time-out history boundaries.
    schema = {'state': FieldSpec(2), 'terminated': FieldSpec(dtype=torch.bool), 'truncated': FieldSpec(dtype=torch.bool)}
    buffer = ReplayBuffer(schema, buffer_size=12, num_envs=3)
    rows = []
    for t in range(19):
        row = {'state': torch.stack([torch.arange(3).float(), torch.full((3,), float(t))], -1),
               'terminated': torch.tensor([t in (2, 11), t in (3, 14), False]),
               'truncated': torch.tensor([False, t == 7, t in (0, 8)])}
        buffer.insert(TensorDict(row, batch_size=[3]))
        rows.append(row)
    fixture['replay'] = {'rows': rows, 'samples': {}}
    for length in (1, 3, 8):
        entry = {'rng': torch.get_rng_state()}
        entry['sample'] = buffer.sample(length, 12).to_dict()
        window, padded = buffer.sample_causal(length, 12)
        entry.update(causal=window.to_dict(), padded=padded)
        fixture['replay']['samples'][str(length)] = entry

    bodies = 14
    q = torch.randn(b, bodies, 4)
    q /= q.norm(dim=-1, keepdim=True)
    p = torch.randn(b, bodies, 3)*.1
    reward_inputs = dict(aux_state=states[:, 0, :7], pred_pos=p, pred_rot_mat=matrix_from_quat(q),
        pred_lin_vel=torch.randn(b, bodies, 3)*.2, pred_ang_vel=torch.randn(b, bodies, 3)*.2,
        joint_state=torch.randn(b, 58), action=actions[:, 0], last_action=actions[:, 1],
        target_pos=p+torch.randn_like(p)*.01, target_quat=q,
        target_lin_vel=torch.randn(b, bodies, 3)*.2, target_ang_vel=torch.randn(b, bodies, 3)*.2,
        soft_joint_pos_limits=torch.tensor([[-1., 1.]]).repeat(29, 1), dt=.02)
    fixture['reward'] = {'inputs': reward_inputs, 'value': tracking_reward(**reward_inputs)}

    if any(n == 'gobot' or n.startswith('gobot.') for n in sys.modules):
        raise RuntimeError('Gobot was imported into the reference process')
    imported = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.startswith(('supermimic.', 'mjlab.', 'rsl_rl.')) and Path(path).suffix == '.py':
            imported[name] = sha256(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detach(fixture), output)
    report.update(fixture_sha256=sha256(output), fixture_bytes=output.stat().st_size,
                  gobot_imported=False, compile=False, tf32=False, numerical_sources=imported,
                  exporter_sha256=sha256(__file__), probe_sha256=sha256(Path(__file__).with_name('probes.py')),
                  verifier_sha256=sha256(Path(__file__).with_name('reference.py')), lock_sha256=sha256(LOCK))
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'fixture': str(output), 'bytes': report['fixture_bytes'], 'gobot_imported': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mjlab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.reference, args.mjlab, args.output)
