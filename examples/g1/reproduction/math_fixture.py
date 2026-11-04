"""Generate numerical expectations in a separate process from untouched reference files.

Package initializers that create environments are bypassed. Numerical modules and
mjlab's rotation primitives are executed verbatim; Gobot is never imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types


def numerical_packages(reference, mjlab):
    roots = {'supermimic': Path(reference)/'src/supermimic', 'mjlab': Path(mjlab)/'src/mjlab'}
    # Avoid environment/plugin initialization, without replacing any numerical function.
    for name in ('supermimic', 'supermimic.envs', 'supermimic.envs.mdp',
                 'mjlab', 'mjlab.utils', 'mjlab.utils.lab_api'):
        top, *rest = name.split('.')
        package = types.ModuleType(name)
        package.__path__ = [str(roots[top].joinpath(*rest))]
        if name == 'mjlab':
            package.MJLAB_SRC_PATH = roots[top]
        sys.modules[name] = package


def generate(reference, mjlab, output):
    numerical_packages(reference, mjlab)
    import torch
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame, analytic_observation, flatten_history_term_major
    from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
    from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
    from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1701)
    n, k = 6, 8
    states = torch.randn(k+1, n, 71)
    states[..., 3:7] /= states[..., 3:7].norm(dim=-1, keepdim=True)
    actions = torch.randn(k, n, 29)*0.1
    context = torch.randn(n, 33)*.005
    context[:, 0] = .75
    done = (torch.rand(k, n) > .8).float()
    reference_pos = torch.randn(k, n, 3)
    reference_quat = torch.randn(k, n, 4)
    reference_quat /= reference_quat.norm(dim=-1, keepdim=True)
    refs = [ReferenceFrame(None, None, None, None, reference_pos[i], reference_quat[i]) for i in range(k)]
    model = WorldModel(WorldModelCfg(hidden=(32, 32), term_hidden=(8, 8)), AnalyticHeads, 'cpu')
    # Use a nonzero dynamics head to exercise gradients through every layer.
    with torch.no_grad():
        model.dynamics.mlp.net[-1].weight.normal_(std=.002)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    action_probe = actions.clone().requires_grad_(True)
    loss, metrics = model.window_loss(states, action_probe, context, done, refs)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss.backward()
    grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    action_gradient = action_probe.grad.clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()
    model_after = {name: value.detach().clone() for name, value in model.state_dict().items()}
    pcfg = PolicyCfg(encoder_hidden=(32, 16), decoder_hidden=(32, 16))
    policy = Policy(pcfg)
    critic = Critic(CriticCfg(hidden=(16, 8)), pcfg=pcfg)
    history = states[:10] if len(states) >= 10 else states[:1].repeat(10, 1, 1)
    previous_actions = torch.randn(10, n, 29)
    default = torch.randn(29)*.1
    observations = analytic_observation(history, previous_actions, context, default)
    flat = flatten_history_term_major(observations)
    goals = torch.randn(n, 680)
    canonical = model.spec.canonical(states[0])
    eroot = torch.randn(n, 4)
    xi = model.spec.normalize_xi(context)
    robot_quat = torch.randn(n, 14, 4)
    robot_quat /= robot_quat.norm(dim=-1, keepdim=True)
    robot_pos = torch.randn(n, 14, 3)*.2
    robot_lin = torch.randn(n, 14, 3)*.2
    robot_ang = torch.randn(n, 14, 3)*.2
    ref_quat = quat_mul(quat_from_rotvec(torch.randn(n, 14, 3)*.03), robot_quat)
    reward_input = dict(
        robot_body_pos_w=robot_pos, robot_body_quat_w=robot_quat,
        robot_body_lin_vel_w=robot_lin, robot_body_ang_vel_w=robot_ang,
        ref_body_pos_w=robot_pos+torch.randn(n, 14, 3)*.02, ref_body_quat_w=ref_quat,
        ref_body_lin_vel_w=robot_lin+torch.randn(n, 14, 3)*.05,
        ref_body_ang_vel_w=robot_ang+torch.randn(n, 14, 3)*.1,
        anchor_index=7, action=actions[0], prev_action=previous_actions[0],
        joint_pos=states[0, :, 7:36], soft_limits=torch.tensor([[-1., 1.]]).repeat(29, 1),
    )
    payload = dict(states=states, actions=actions, context=context, done=done,
                   ref_pos=reference_pos, ref_quat=reference_quat, wm_initial=initial,
                   wm_loss=loss.detach(), wm_metrics=metrics, wm_grads=grads,
                   action_gradient=action_gradient, wm_after=model_after,
                   history=history, previous_actions=previous_actions, default=default,
                   observations=observations, flat_history=flat, goals=goals,
                   canonical=canonical, eroot=eroot, normalized_context=xi,
                   policy=policy.state_dict(), critic=critic.state_dict(),
                   policy_output=policy(flat, goals).detach(),
                   critic_output=critic(canonical, goals, xi, eroot).detach(),
                   reward_input=reward_input,
                   reward_terms=tracking_reward_terms(**reward_input, cfg=TrackingRewardCfg()))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    imported = {}
    for name, module in list(sys.modules.items()):
        source = getattr(module, '__file__', None)
        if source and name.startswith(('supermimic.', 'mjlab.')):
            path = Path(source)
            if path.suffix == '.py':
                imported[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'protocol': 'reproduced_math_fixture_v1', 'torch': torch.__version__,
                'mujoco': importlib.metadata.version('mujoco'), 'numerical_sources': imported,
                'gobot_imported': 'gobot' in sys.modules,
                'scope': 'CPU algebra only; not physics or full-training parity',
                'fixture_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mjlab', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.reference, args.mjlab, args.output)
