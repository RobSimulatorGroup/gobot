"""Summarize independent cluster jobs without treating smoke completion as admission."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root):
    rows=[]
    for path in sorted(Path(root).rglob('experiment.json')):
        folder=path.parent
        experiment=json.loads(path.read_text())
        if 'purpose' not in experiment:continue
        summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        failure=json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else {}
        hardware=json.loads((folder/'hardware.json').read_text()) if (folder/'hardware.json').exists() else {}
        metrics=[]
        if (folder/'metrics.jsonl').exists():
            for line in (folder/'metrics.jsonl').read_text().splitlines():
                try:metrics.append(json.loads(line))
                except json.JSONDecodeError:continue  # live trailing write
        last=[row['iteration_time'] for row in metrics[-3:] if row.get('iteration_time',0)>0]
        algorithm=experiment.get('algorithm',{})
        ledger=json.loads((folder/'gpu_budget.json').read_text()) if (folder/'gpu_budget.json').exists() else {}
        rows.append({'run':str(folder),'purpose':experiment['purpose'],'seed':experiment['seed'],
            'num_envs':experiment['num_envs'],'wm_batch':algorithm.get('wm_batch'),'pi_batch':algorithm.get('pi_batch'),
            'gpu':hardware.get('gpu'),'state':'failed' if failure else status.get('status','unknown'),
            'completed':summary.get('completed',False),'phase':summary.get('phase',status.get('phase')),
            'iteration':summary.get('iteration',status.get('iteration')),
            'elapsed_s':summary.get('elapsed_s',status.get('elapsed_s')),
            'process_gpu_hours':ledger.get('seconds',0)/3600,
            'median_last_updates_s':statistics.median(last) if len(last)==3 else None,
            'peak_torch_allocated_bytes':summary.get('peak_torch_allocated_bytes'),
            'error':failure.get('error'),'policy_admitted':False})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=summarize(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'runs':rows,'note':'process ledgers exclude scheduler overhead; smoke success is not policy admission'},indent=2)+'\n')
    if rows:
        with a.output.with_suffix('.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
