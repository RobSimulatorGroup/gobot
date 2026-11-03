from types import SimpleNamespace
import json
import torch
from gobot.rl.policy import PolicyManifest
from examples.g1.tracking_video_queue import VideoSchedule


def test_preview_schedule_uses_fixed_deadlines_and_durable_checkpoints(tmp_path):
    manifest=PolicyManifest(task_name='fixture',task_version='v1',observation_spec={'dim':1},action_spec={'dim':1},
        joint_names=('joint',),physics_dt=.005,decimation=4,control={})
    model=torch.nn.Linear(1,1)
    session=SimpleNamespace(iteration=10,pi=model,actor_frozen=True,phase='adaptation',manifest=manifest,
        contract_digest='contract',_checkpoint=lambda _: {'args':{'hist':10}})
    schedule=VideoSchedule(tmp_path,interval_seconds=900,start=0)
    assert schedule.enqueue(session,now=899) is None
    request=schedule.enqueue(session,now=905)
    data=json.loads(request.read_text())
    checkpoint=tmp_path/data['checkpoint']
    assert checkpoint.exists() and data['preview_only']
    assert schedule.next_time==1800
    assert 'wm' not in torch.load(checkpoint,weights_only=True)
    assert schedule.enqueue(session,now=1800) is None
    session.iteration=20
    request=schedule.enqueue(session,now=2800)
    assert json.loads(request.read_text())['iteration']==20
    assert schedule.next_time==3600
    assert schedule.enqueue(session,force=True) is None
