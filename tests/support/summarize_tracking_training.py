"""Summarize paired rank traces without hiding sparse contact/termination failures."""
import argparse
import json
from pathlib import Path
import torch


def max_difference(a, b):
    if torch.is_tensor(a):
        if a.shape != b.shape:
            raise ValueError('trace shapes differ')
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('trace keys differ')
        return max((max_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError('trace lengths differ')
        return max((max_difference(x, y) for x, y in zip(a, b)), default=0.)
    if a == b:
        return 0.
    if isinstance(a, (int,float)) and isinstance(b,(int,float)):
        return abs(a-b)
    raise ValueError('incompatible non-numeric trace value')


def summarize(a, b):
    # The generic Gobot compatibility wrapper supplies an unused `policy`
    # alias. Verify it really aliases actor observations before comparing the
    # explicitly configured actor/critic input groups.
    for trace in (a,b):
        observations=[trace['after_reset']['obs']]
        observations += [u['storage']['observations'] for u in trace['updates']]
        for obs in observations:
            if set(obs)=={'actor','critic','policy'}:
                torch.testing.assert_close(obs['policy'],obs['actor'],atol=0,rtol=0)
                del obs['policy']
    result = {
        'initial_model_max_abs': max_difference(a['after_runner']['models'], b['after_runner']['models']),
        'initial_observation_max_abs': max_difference(a['after_reset']['obs'], b['after_reset']['obs']),
        'updates': [],
    }
    for u,v in zip(a['updates'],b['updates']):
        collision = [int(((s['terms'][:,-1]-t['terms'][:,-1]).abs()>1e-5).sum())
                     for s,t in zip(u['steps'],v['steps'])]
        dones = sum(int((s['dones'].bool()!=t['dones'].bool()).sum()) for s,t in zip(u['steps'],v['steps']))
        fields = ('storage','gradients','post_update','pre_update_rng','post_update_rng','losses')
        maxima = {k:max_difference(u[k],v[k]) for k in fields}
        result['updates'].append({
            'self_collision_mismatched_transitions':sum(collision),
            'first_step_self_collision_mismatches':collision[0],
            'total_transitions':sum(s['rewards'].numel() for s in u['steps']),
            'termination_mismatches':dones,
            'reference_frame_mismatches':sum(int((s['state']['frames']!=t['state']['frames']).sum())
                                              for s,t in zip(u['steps'],v['steps'])),
            'rng_equal':all(max_difference(s['rng'],t['rng'])==0 for s,t in zip(u['steps'],v['steps'])),
            'continuous_reward_max_abs':max(max_difference(s['terms'][:,:-1],t['terms'][:,:-1])
                                            for s,t in zip(u['steps'],v['steps'])),
            'reward_max_abs':max_difference(u['storage']['rewards'],v['storage']['rewards']),
            'reward_rms':float((u['storage']['rewards'].double()-v['storage']['rewards'].double()).square().mean().sqrt()),
            'returns_max_abs':max_difference(u['storage']['returns'],v['storage']['returns']),
            'advantages_max_abs':max_difference(u['storage']['advantages'],v['storage']['advantages']),
            'gradient_max_abs':maxima['gradients'],
            'post_update_max_abs':maxima['post_update'],
            'exact_ppo_update':all(value==0 for value in maxima.values()),
        })
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pairs',nargs='+',required=True,help='directory_a:directory_b')
    p.add_argument('--ranks',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report={}
    for pair in args.pairs:
        a,b=pair.split(':')
        report[pair]=[]
        for rank in range(args.ranks):
            suffix=Path(f'rank{rank}')/'trace.pt' if args.ranks>1 else Path('trace.pt')
            first=torch.load(args.root/a/suffix,map_location='cpu',weights_only=False)
            second=torch.load(args.root/b/suffix,map_location='cpu',weights_only=False)
            report[pair].append({'rank':rank,**summarize(first,second)})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':
    main()
