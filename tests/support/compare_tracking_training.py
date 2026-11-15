"""Find the first numerical/RNG disagreement between instrumented PPO updates."""
import argparse
import json
from pathlib import Path
import torch


def difference(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        if a.shape!=b.shape:return {'shape_a':list(a.shape),'shape_b':list(b.shape)}
        d=(a.to(torch.float64)-b.to(torch.float64)).abs()
        return {'max_abs':float(d.max()) if d.numel() else 0.,'rms':float(d.square().mean().sqrt()) if d.numel() else 0.,'equal':torch.equal(a,b)}
    if isinstance(a,dict) and isinstance(b,dict):return {str(k):difference(a[k],b[k]) for k in a.keys()&b.keys()}
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return [difference(x,y) for x,y in zip(a,b)]
    return {'a':a,'b':b,'equal':a==b}


def worst(value):
    if isinstance(value,dict):
        if 'max_abs' in value:return value['max_abs']
        return max((worst(v) for v in value.values()),default=0.)
    if isinstance(value,list):return max(map(worst,value),default=0.)
    return 0.


def compare(a,b,output,require_exact_ppo=False):
    a=torch.load(a,map_location='cpu',weights_only=False);b=torch.load(b,map_location='cpu',weights_only=False)
    result={}
    for name in ('after_reset','after_runner'):
        result[name]=difference(a[name],b[name])
        print(name,'rng',result[name]['rng'],'obs_max',worst(result[name].get('obs',{})),
              'models_max',worst(result[name].get('models',{})),flush=True)
    result['updates']=[]
    for i,(u,v) in enumerate(zip(a['updates'],b['updates'])):
        d=difference(u,v);result['updates'].append(d)
        print('UPDATE',i,'start_rng',d['start']['rng'], 'start_model_max',worst(d['start']['models']))
        for j,entry in enumerate(d['acts']):
            s=d['steps'][j]
            print('step',j,'rng_equal',entry['rng']['cuda']['equal'],'act',worst(entry['action']),
                  'reward',worst(s['rewards']),'done',worst(s['dones']),'obs',worst(s['obs']),
                  'frames',worst(s['state']['frames']),'next_rng_equal',s['rng']['cuda']['equal'])
        print('storage', {k:worst(val) for k,val in d['storage'].items()})
        print('gradients',[worst(grad['values']) for grad in d['gradients']])
        print('losses',d['losses'],'post_update_max',worst(d['post_update']))
    output.write_text(json.dumps(result,indent=2)+'\n')
    if require_exact_ppo:
        if len(a['updates']) != len(b['updates']) or not result['updates']:
            raise AssertionError('PPO replay update counts differ or are empty')
        def exact(value):
            if isinstance(value, dict):
                if 'equal' in value: return value['equal']
                return all(exact(v) for v in value.values())
            if isinstance(value, list): return all(exact(v) for v in value)
            return True
        fields=('storage', 'gradients', 'post_update', 'losses', 'pre_update_rng', 'post_update_rng')
        if not all(exact(update[field]) for update in result['updates'] for field in fields):
            raise AssertionError('PPO replay diverged; inspect comparison JSON')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-exact-ppo', action='store_true')
    compare(**vars(p.parse_args()))
