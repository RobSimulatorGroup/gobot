"""Validate complete-clip reports and compare matching motion sets without hiding failures."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from .check_context import sha256


def validate(payload):
    rows=payload['clips']
    if not rows:raise ValueError('benchmark contains no clips')
    indexed={}
    for row in rows:
        name=row['name'];frames=row['frames'];steps=row['steps']
        if name in indexed:raise ValueError(f'duplicate benchmark clip: {name}')
        if not isinstance(frames,int) or frames<3 or not isinstance(steps,int) or not 1<=steps<=frames:
            raise ValueError(f'invalid frame/step counts: {name}')
        if not isinstance(row['failed'],bool) or not math.isfinite(row['norm_return']):
            raise ValueError(f'invalid failure/return field: {name}')
        if row['failed']:
            if row['fail_step']!=steps:raise ValueError(f'failure step differs from executed steps: {name}')
        elif steps!=frames or row['fail_step']!=-1:
            raise ValueError(f'nonfailed clip was not completed: {name}')
        indexed[name]=row
    summary={'clips':len(rows),'failed':sum(r['failed'] for r in rows),
             'mean_norm_return':sum(r['norm_return'] for r in rows)/len(rows),
             'step_weighted_norm_return':sum(r['norm_return']*r['steps'] for r in rows)/sum(r['steps'] for r in rows)}
    for key,value in summary.items():
        if not math.isclose(payload['summary'][key],value,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError(f'benchmark summary does not match per-clip results: {key}')
    return indexed,summary


def compare(reference,candidate):
    a,sa=validate(reference);b,sb=validate(candidate)
    if a.keys()!=b.keys():raise ValueError('benchmarks do not cover the same clips')
    rows=[]
    for name in sorted(a):
        x,y=a[name],b[name]
        if x['frames']!=y['frames']:raise ValueError(f'clip frame counts differ: {name}')
        rows.append({'name':name,'frames':x['frames'],'reference_steps':x['steps'],'gobot_steps':y['steps'],
                     'reference_failed':x['failed'],'gobot_failed':y['failed'],
                     'reference_return':x['norm_return'],'gobot_return':y['norm_return'],
                     'return_difference':y['norm_return']-x['norm_return']})
    return {'protocol':'complete_clip_paired_comparison_v1','reference':sa,'gobot':sb,
            'mean_return_difference':sb['mean_norm_return']-sa['mean_norm_return'],
            'weighted_return_difference':sb['step_weighted_norm_return']-sa['step_weighted_norm_return'],
            'failed_count_difference':sb['failed']-sa['failed'],
            'reference_only_failures':[r['name'] for r in rows if r['reference_failed'] and not r['gobot_failed']],
            'gobot_only_failures':[r['name'] for r in rows if r['gobot_failed'] and not r['reference_failed']],
            'max_abs_clip_return_difference':max(abs(r['return_difference']) for r in rows),
            'clips':rows,'fresh_training_reproduction':False,
            'scope':'paired fixed-policy diagnostics; no statistical training-quality claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()))
    report.update(reference_report_sha256=sha256(a.reference),candidate_report_sha256=sha256(a.candidate))
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='clips'},indent=2))

if __name__=='__main__':main()
