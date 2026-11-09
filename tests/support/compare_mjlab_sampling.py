"""Report sampling equality separately from numerical physics differences.

Run capture_mjlab_sampling.py once per runtime into the same output directory.
Each runtime uses its own Python environment and robot authoring path.
"""
import argparse
import json
from pathlib import Path

import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
left=torch.load(a.directory/'gobot.pt',map_location='cpu',weights_only=True)
right=torch.load(a.directory/'mjlab.pt',map_location='cpu',weights_only=True)
assert left.keys()==right.keys()
exact={'rng','command_time','push_time','heading','standing','heading_env','forward_env','done'}
report={'sampling_equal':True,'phases':{}}
for phase,values in left.items():
    errors={}
    for key,value in values.items():
        other=right[phase][key]
        if value.shape!=other.shape:
            raise ValueError(f'{phase}/{key}: shape mismatch {value.shape} != {other.shape}')
        errors[key]=float((value.double()-other.double()).abs().max())
        if key in exact or phase=='startup':
            report['sampling_equal'] &= torch.equal(value,other)
    report['phases'][phase]=errors
report['note']='Sampling equality does not imply identical collision solutions or full training trajectories.'
(a.directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not report['sampling_equal']:
    raise SystemExit('sampling diverged; inspect the first mismatched phase before training')
