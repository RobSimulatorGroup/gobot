"""Budgeted real CPU/Warp tracking state, contact and masked-reset comparison."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from gobot.rl.motion_tracking import MotionTrackingCfg, make_motion_tracking_env
from gobot.rl.supermimic.geometry import flatten_state
from gobot.rl.locomotion.budget import TrainingBudget
from .budget import DEFAULT_LEDGER
from .supermimic_reference import RunStatus, write_json


def run(output, *, ledger=DEFAULT_LEDGER, hours=.15):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]/'g1'
    cfg=MotionTrackingCfg(project_path=project,motion_files=(str(project/'assets/motions/tracking/walk1_subject1.csv'),),
        contact_links=('left_ankle_roll_link','right_ankle_roll_link'))
    budget=TrainingBudget(ledger,total_hours=24,run_hours=hours,metadata={'stage':'tracking_cpu_warp_parity'})
    status=RunStatus(output,budget);cpu=gpu=None
    try:
        torch.set_num_threads(4)
        cpu=make_motion_tracking_env(cfg,num_envs=2,backend='mujoco-cpu')
        gpu=make_motion_tracking_env(cfg,num_envs=2,backend='mujoco-warp')
        frames=torch.tensor([200,300])
        cpu.reset_to_frames(frames);gpu.reset_to_frames(frames.cuda())
        action=(cpu.batch.read_state().joint_position-cpu.default_q)/cpu.action_scale
        measurements=[]
        solver_state_preserved=True
        for step in range(8):
            if budget.remaining<10:raise TimeoutError('parity budget exhausted')
            a=cpu.step(action);b=gpu.step(action.cuda())
            arrays=gpu.batch._provider.arrays
            names=('qpos','qvel','qacc','qacc_warmstart','qfrc_actuator','time','ctrl')
            solver_state={name:arrays[name].clone() for name in names}
            gpu.batch._provider.refresh_kinematics()
            for name,old in solver_state.items():
                torch.testing.assert_close(arrays[name],old,rtol=0,atol=0)
            x,y=a.info['supermimic']['s1'],b.info['supermimic']['s1'].cpu()
            c,d=cpu.batch.read_contacts(),gpu.batch.read_contacts()
            measurements.append({'step':step+1,'state_max_abs':float((x-y).abs().max()),
                'base_position_max_abs':float((x[:,:3]-y[:,:3]).abs().max()),
                'joint_position_max_abs':float((x[:,13:42]-y[:,13:42]).abs().max()),
                'reward_max_abs':float((a.reward-b.reward.cpu()).abs().max()),
                'contact_force_max_abs':float((c['force']-d['force'].cpu()).abs().max()),
                'contact_found_equal':bool(torch.equal(c['found'],d['found'].cpu()))})
            if not torch.equal(a.done,b.done.cpu()):raise AssertionError('done masks differ')
            if step==3:
                for env in (cpu,gpu):
                    before=flatten_state(env.physical_state())[1].clone()
                    env.reset_to_frames(torch.tensor([500],device=env.device),[0])
                    torch.testing.assert_close(flatten_state(env.physical_state())[1],before,atol=2.e-6,rtol=1.e-6)
            status.update(status='checking',iteration=step+1)
        report={'protocol':'tracking_short_horizon_parity_v1','measurements':measurements,
                'max_state_error':max(row['state_max_abs'] for row in measurements),
                'masked_reset_passed':True,'kinematic_refresh_preserves_solver_state':solver_state_preserved,'contact_error_reported_separately':True,
                'passed':max(row['state_max_abs'] for row in measurements)<=1.e-3}
        write_json(output/'parity.json',report)
        status.update(status='finished',passed=report['passed'])
        if not report['passed']:raise AssertionError('short horizon state parity exceeds 1e-3; inspect per-field errors')
        return report
    except BaseException as error:
        status.update(status='failed',error=str(error));raise
    finally:
        if gpu:gpu.close()
        if cpu:cpu.close()
        status.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    parser.add_argument('--hours',type=float,default=.15)
    args=parser.parse_args();run(args.output,ledger=args.ledger,hours=args.hours)
