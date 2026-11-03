"""Durable 15-minute preview requests, optionally rendered by a separate worker."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
from pathlib import Path
import time
import torch

from gobot.rl.experiment_identity import file_digest
from gobot.rl.policy import POLICY_MANIFEST_KEY,write_policy_manifest_sidecar
from examples.locomotion.supermimic_reference import write_json


class VideoSchedule:
    def __init__(self, directory, *, interval_seconds=900., start=None):
        if not math.isfinite(interval_seconds) or interval_seconds<=0:
            raise ValueError('preview interval must be finite and positive')
        self.directory=Path(directory)
        self.interval=interval_seconds
        self.next_time=(time.monotonic() if start is None else start)+interval_seconds
        self.last_iteration=None

    def enqueue(self, session, *, now=None, force=False):
        now=time.monotonic() if now is None else now
        if not force and now<self.next_time:return None
        missed=0
        while self.next_time<=now:
            self.next_time+=self.interval;missed+=1
        if session.iteration==self.last_iteration:return None
        requests=self.directory/'video_requests';requests.mkdir(parents=True,exist_ok=True)
        request=requests/f'{session.iteration:08d}.json'
        if request.exists():
            self.last_iteration=session.iteration
            return None
        folder=self.directory/'video_checkpoints';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'policy_{session.iteration:08d}.pt'
        payload={'pi':session.pi.state_dict(),'args':session._checkpoint(False)['args'],
                 'iteration':session.iteration,'actor_frozen':session.actor_frozen,
                 POLICY_MANIFEST_KEY:session.manifest.metadata(),'source_contract':session.contract_digest,
                 'purpose':'preview_only'}
        temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
        write_policy_manifest_sidecar(path,session.manifest)
        write_json(request,{'protocol':'gobot_tracking_preview_v1','iteration':session.iteration,
            'checkpoint':str(path.relative_to(self.directory)),'checkpoint_sha256':file_digest(path),
            'requested_unix':time.time(),'missed_intervals':max(0,missed-1),'seconds':8.,
            'phase':session.phase,'preview_only':True})
        self.last_iteration=session.iteration
        return request


def render_pending(directory):
    from .record_tracking import record,WINDOWS
    import gobot
    directory=Path(directory).resolve();results=directory/'video_results';results.mkdir(parents=True,exist_ok=True)
    try:
        for request in sorted((directory/'video_requests').glob('*.json')):
            result=results/request.name
            if result.exists():continue
            data=json.loads(request.read_text());checkpoint=(directory/data['checkpoint']).resolve()
            if not checkpoint.is_relative_to(directory) or file_digest(checkpoint)!=data['checkpoint_sha256']:
                raise ValueError('preview checkpoint path/hash differs from the queued request')
            output=directory/'videos'/request.stem
            try:
                reports=[record(checkpoint,output/f'{clip}.mp4',clip,start,data['seconds']) for clip,start in WINDOWS.items()]
                write_json(result,{'status':'finished','request':data,'output':str(output),
                                   'clips':[{k:v for k,v in row.items() if k!='trajectory'} for row in reports]})
            except Exception as error:
                write_json(result,{'status':'failed','request':data,'error':str(error)})
                raise
    finally:gobot.render._shutdown_headless_render_context()


def main():
    from examples.locomotion.budget import add_budget_arguments,evaluation_budget
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    add_budget_arguments(parser);args=parser.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    with (args.run_dir/'video_worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        with evaluation_budget(args,stage='tracking_periodic_preview'):
            render_pending(args.run_dir)


if __name__=='__main__':main()
