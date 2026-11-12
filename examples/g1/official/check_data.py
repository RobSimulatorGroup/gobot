"""Independently load converted motions with the pinned official MotionLoader."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import types
from .reference import LOCK, verify_checkout, sha256


def check(data,output):
    import mjlab
    import numpy as np
    import torch
    root=Path(mjlab.__file__).resolve().parents[2]
    lock=json.loads(LOCK.read_text())
    verify_checkout(root,lock['mjlab_research']['commit'])
    package=types.ModuleType('mjlab.tasks');package.__path__=[str(root/'src/mjlab/tasks')]
    sys.modules['mjlab.tasks']=package
    from mjlab.tasks.tracking.mdp.commands import MotionLoader
    data=Path(data)
    joints=json.loads((data/'joint_names.json').read_text())
    loader=MotionLoader(str(data),types.SimpleNamespace(joint_names=joints),device='cpu')
    names=('root_pos_w','root_quat_w','joint_pos','root_lin_vel_w','root_ang_vel_w','joint_vel')
    fields={}
    for name in names:
        expected=torch.from_numpy(np.load(data/f'{name}.npy')).float()
        actual=getattr(loader,name)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        fields[name]={'shape':list(actual.shape),'exact':True}
    assert not any(name=='gobot' or name.startswith('gobot.') for name in sys.modules)
    report={'passed':True,'gobot_imported':False,'manifest_sha256':sha256(data/'manifest.json'),
            'reference_commit':lock['mjlab_research']['commit'],'fields':fields,
            'frames':loader.time_step_total,'clips':len(loader.motion_ends),
            'scope':'official loader values exactly match shared packed tensors; no physics or trained-quality claim'}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check(a.data,a.output)
