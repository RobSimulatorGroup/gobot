"""Independent real-rollout validation before unfreezing a transferred actor."""
from __future__ import annotations
import math
from dataclasses import asdict
import torch

from gobot.rl.experiment_identity import canonical_digest
from gobot.rl.supermimic import networks
from gobot.rl.supermimic.geometry import unflatten_state
from .evaluate import load_policy
from .training import read_checkpoint


def _world_model(payload, env):
    args=payload['args']
    model=networks.WorldModel(hidden=tuple(args.get('test_hidden_dims') or args['wm_hidden']),
        num_joints=env.supermimic_mdp.num_joints,context_dim=env.supermimic_mdp.context_dim).to(env.device)
    model.load_state_dict(payload['wm'],strict=True)
    return model.eval()


def real_return_targets(rewards, terminated, done, next_values, *, gamma, horizon=32):
    """Truncated real returns with pre-reset bootstrap and no episode crossing."""
    if rewards.shape!=terminated.shape or rewards.shape!=done.shape or rewards.shape!=next_values.shape:
        raise ValueError('return target arrays must share [time, environment] dimensions')
    if rewards.ndim!=2 or rewards.shape[0]<horizon or horizon<1:
        raise ValueError('not enough real transitions for the return horizon')
    starts=range(0,rewards.shape[0]-horizon+1,horizon)
    targets=[]
    for start in starts:
        alive=torch.ones_like(done[start]);value=torch.zeros_like(rewards[start])
        for offset in range(horizon):
            index=start+offset
            value+=gamma**offset*torch.where(alive,rewards[index],0.)
            bootstrap=alive & ~terminated[index] & (done[index] | (offset==horizon-1))
            value+=gamma**(offset+1)*torch.where(bootstrap,next_values[index],0.)
            alive &= ~done[index]
        targets.append(value)
    return torch.stack(targets)


@torch.inference_mode()
def validate_adaptation(env, checkpoint, baseline, *, thresholds=None, steps=128, seed=100071, progress=None):
    """Use a separate environment; no validation samples enter training replay.

    Without predeclared thresholds this returns measurements only. Error scaling
    is fixed by the source checkpoint so updating normalization cannot improve a
    validation score by changing its denominator.
    """
    if steps<32:
        raise ValueError('adaptation validation requires at least 32 real steps')
    candidate,digest=read_checkpoint(checkpoint)
    source,source_digest=read_checkpoint(baseline)
    metadata=candidate.get('gobot_training',{})
    if metadata.get('phase')!='adaptation' or not candidate.get('actor_frozen'):
        raise ValueError('validate a full adaptation checkpoint with its actor frozen')
    if not metadata.get('lineage') or metadata['lineage'][-1]['checkpoint_sha256']!=source_digest:
        raise ValueError('baseline does not match the adaptation checkpoint lineage')
    if metadata['contract']['physics']!=env.batch.artifact.digest:
        raise ValueError('validation environment does not match the target physics artifact')
    if metadata['contract']['backend']!=env.batch.backend:
        raise ValueError('adaptation validation must use the target training backend')
    if canonical_digest(metadata['contract']['task']['randomization'])!=canonical_digest(asdict(env.cfg.randomization)):
        raise ValueError('validation randomization differs from the training contract')
    if 'critic' not in candidate:
        raise ValueError('critic validation requires a full checkpoint')
    if thresholds is not None:
        required=('world_model_rmse','critic_relative_rmse','minimum_survival_fraction','minimum_step_return')
        if any(name not in thresholds for name in required):
            raise ValueError('validation thresholds are incomplete')
        bounds=[thresholds['world_model_rmse'].get(str(h)) for h in (1,8,32)]
        values=[*bounds,thresholds['critic_relative_rmse'],thresholds['minimum_survival_fraction'],thresholds['minimum_step_return']]
        if any(value is None or not math.isfinite(value) for value in values) or min(*bounds,thresholds['critic_relative_rmse'])<0:
            raise ValueError('validation thresholds must be finite with nonnegative error bounds')
        if not 0<=thresholds['minimum_survival_fraction']<=1:
            raise ValueError('survival threshold must lie in [0,1]')
    actor_identical=(candidate['pi'].keys()==source['pi'].keys() and
        all(torch.equal(value,source['pi'][name]) for name,value in candidate['pi'].items()))
    policy=load_policy(env,candidate)
    model,old_model=_world_model(candidate,env),_world_model(source,env)
    args=candidate['args'];mdp=env.supermimic_mdp
    critic=networks.Critic(candidate['critic']['norm.mean'].numel(),
        hidden=tuple(args.get('test_hidden_dims') or args['critic_hidden'])).to(env.device)
    critic.load_state_dict(candidate['critic'],strict=True);critic.eval()
    env.reset(seed=seed)
    history=None;transitions=[]
    for index in range(steps):
        current=env.state.info['supermimic_state']
        obs=mdp.obs(unflatten_state(current['s']),current['a_prev'],current['xi'],current['f'])
        shifted=obs[:,None].repeat(1,int(args['hist']),1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
        history=torch.where((current['ep_step']==0)[:,None,None],obs[:,None].expand_as(shifted),shifted)
        action=policy(history,mdp.goal(current['f'],int(args['goal_frames'])))
        transition=env.step(action).info['supermimic']
        transitions.append({key:transition[key].clone() for key in ('s','s1','a','xi','f','r','term','done')})
        if progress:progress(index+1)
    data={key:torch.stack([row[key] for row in transitions]) for key in transitions[0]}
    if not all(bool(torch.isfinite(value).all()) for value in data.values()):
        raise FloatingPointError('validation produced non-finite physical transitions')
    floor=torch.tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*mdp.num_joints+[.05]*mdp.num_joints,device=env.device)
    scale=torch.maximum(old_model.out_norm.std(),floor)
    measurements={}
    for model_name,wm in (('candidate',model),('baseline',old_model)):
        totals={h:torch.zeros((),device=env.device) for h in (1,8,32)}
        counts={h:0 for h in totals}
        for start in range(0,steps-31,32):
            state=unflatten_state(data['s'][start]);alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            for offset in range(32):
                index=start+offset
                state=wm.step(state,data['a'][index],mdp.normalize_context(data['xi'][index]))
                horizon=offset+1
                if horizon in totals:
                    errors=wm.state_error(state,unflatten_state(data['s1'][index]))/scale
                    valid_errors=errors[alive]
                    if not bool(torch.isfinite(valid_errors).all()):
                        raise FloatingPointError('non-finite world-model validation prediction')
                    totals[horizon]+=valid_errors.square().sum()
                    counts[horizon]+=int(alive.sum())*errors.shape[-1]
                alive &= ~data['done'][index]
        measurements[model_name]={str(h):float((totals[h]/max(counts[h],1)).sqrt()) if counts[h] else None for h in totals}
    valid_windows={str(h):counts[h]//scale.numel() for h in counts}
    frames=data['f'].flatten();context=mdp.normalize_context(data['xi'].flatten(0,1))
    state=unflatten_state(data['s'].flatten(0,1));next_state=unflatten_state(data['s1'].flatten(0,1))
    next_frames=torch.minimum(frames+1,mdp.motion.clip_end[mdp.motion.clip_id[frames]])
    values=critic(torch.cat((mdp.critic_features(state,frames,context),mdp.goal(frames,int(args['goal_frames']))),-1))
    next_values=critic(torch.cat((mdp.critic_features(next_state,next_frames,context),mdp.goal(next_frames,int(args['goal_frames']))),-1))
    targets=data['r'].flatten()+float(args['gamma'])*(~data['term'].flatten())*next_values
    if not bool(torch.isfinite(values).all() and torch.isfinite(targets).all()):
        raise FloatingPointError('non-finite critic validation prediction')
    value_scale=mdp.max_step_reward/(1-float(args['gamma']))
    one_step_error=float(((values-targets).square().mean()).sqrt()/value_scale)
    real_targets=real_return_targets(data['r'],data['term'],data['done'],next_values.view(steps,env.num_envs),
        gamma=float(args['gamma']),horizon=32)
    starts=torch.arange(0,steps-31,32,device=env.device)
    critic_error=float((values.view(steps,env.num_envs)[starts]-real_targets).square().mean().sqrt()/value_scale)
    survival=float((~data['term'].any(0)).float().mean())
    step_return=float(data['r'].mean()/mdp.max_step_reward)
    wm_values=measurements['candidate']
    checks={
        'world_model_1_8_32':{'passed':thresholds is not None and all(wm_values[str(h)] is not None and wm_values[str(h)]<=thresholds['world_model_rmse'][str(h)] for h in (1,8,32)),
                             'normalized_rmse':measurements,'valid_windows_per_horizon':valid_windows,
                             'scale_source':'baseline_output_normalizer_with_fixed_floors'},
        'critic':{'passed':thresholds is not None and math.isfinite(critic_error) and critic_error<=thresholds['critic_relative_rmse'],
                  'relative_32_step_return_rmse':critic_error,'relative_one_step_bellman_rmse':one_step_error,
                  'bootstrap':'pre-reset next state; bootstrap timeouts, never cross episode boundaries'},
        'physical_rollout':{'passed':thresholds is not None and actor_identical and survival>=thresholds['minimum_survival_fraction'] and step_return>=thresholds['minimum_step_return'],
                            'actor_and_normalizers_unchanged':actor_identical,'survival_fraction':survival,'mean_step_return':step_return}}
    return {'protocol':'tracking_adaptation_validation_v1','passed':all(row['passed'] for row in checks.values()),
            'metrics_only':thresholds is None,'checkpoint_sha256':digest,'baseline_sha256':source_digest,
            'contract_digest':metadata['contract_digest'],'checks':checks,'thresholds':thresholds,
            'thresholds_digest':canonical_digest(thresholds),'seed':seed,'steps':steps,'num_envs':env.num_envs,
            'backend':env.batch.backend,'deterministic':env.cfg.deterministic,
            'dataset':{'split':env.cfg.split,'clips':mdp.motion.names},'note':'warmup validation, not full motion or locomotion admission'}
