"""Compare complete paired tracking evaluations without hiding per-clip failures."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def compare(reference, candidate, *, max_mean_drop=.02):
    if not math.isfinite(max_mean_drop) or max_mean_drop < 0:
        raise ValueError('invalid return tolerance')
    if reference.get('screening') or candidate.get('screening'):
        raise ValueError('screening windows cannot establish full-clip fidelity')
    left,right=reference['per_clip'],candidate['per_clip']
    if not left or set(left)!=set(right):
        raise ValueError('paired fidelity requires the same nonempty clip set')
    rows=[]
    for name,a in left.items():
        b=right[name]
        if min(a['steps'],b['steps']) < 0:
            raise ValueError('negative evaluation duration')
        if a['expected_steps']!=b['expected_steps'] or a['expected_steps']<=0:
            raise ValueError(f'full-clip denominator changed: {name}')
        if not all(math.isfinite(float(row['norm_return'])) for row in (a,b)):
            raise ValueError('non-finite return in comparison')
        a_complete=not a['failed'] and a['steps']>=a['expected_steps']
        b_complete=not b['failed'] and b['steps']>=b['expected_steps']
        rows.append({'clip':name,'reference_return':a['norm_return'],'candidate_return':b['norm_return'],
            'return_delta':b['norm_return']-a['norm_return'],
            'expected_steps':a['expected_steps'],
            'reference_steps':a['steps'],'candidate_steps':b['steps'],
            'reference_complete':a_complete,'candidate_complete':b_complete,
            'new_failure':a_complete and not b_complete,
            'failure_earlier_s':max(0,a['steps']-b['steps'])*.02 if not b_complete else 0.})
    # Recompute from the paired clips; don't trust a stale aggregate header.
    mean=sum(row['return_delta'] for row in rows)/len(rows)
    new=[row['clip'] for row in rows if row['new_failure']]
    return {'schema_version':1,'protocol':'paired_tracking_fidelity_v1',
        'passed':mean>=-max_mean_drop and not new,'max_mean_drop':max_mean_drop,'mean_return_delta':mean,
        'reference_full_clips':sum(row['reference_complete'] for row in rows),
        'candidate_full_clips':sum(row['candidate_complete'] for row in rows),
        'new_failed_clips':new,'worst_return_delta':min(row['return_delta'] for row in rows),'per_clip':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-identity',type=Path,required=True)
    p.add_argument('--candidate-identity',type=Path,required=True)
    a=p.parse_args()
    identities=[json.loads(path.read_text()) for path in (a.reference_identity,a.candidate_identity)]
    def checkpoint_hash(identity):
        value=identity.get('checkpoint_sha256') or (identity.get('parent') or {}).get('sha256')
        if not value:raise ValueError('paired comparison requires checkpoint identity for both runs')
        return value.removeprefix('sha256:')
    modes=[value.get('deterministic',value.get('physics',{}).get('deterministic',False)) for value in identities]
    if modes[0]!=modes[1]:raise ValueError('deterministic execution modes differ')
    if checkpoint_hash(identities[0])!=checkpoint_hash(identities[1]):
        raise ValueError('paired fidelity cannot compare different checkpoints')
    report=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()))
    report['checkpoint_sha256']=checkpoint_hash(identities[0])
    report['candidate_identity']=identities[1].get('digest')
    report['candidate_physics']=identities[1].get('physics',{})
    report.update(reference=str(a.reference.resolve()),candidate=str(a.candidate.resolve()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_clip'},indent=2))


if __name__=='__main__':main()
