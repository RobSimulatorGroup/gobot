"""Verify actual per-world GPU parameters and future trajectories after masked reset."""
from pathlib import Path
import argparse
import torch

from gobot.rl.motion_tracking import MotionTrackingCfg,MotionRandomizationCfg,make_motion_tracking_env
from gobot.rl.locomotion.budget import TrainingBudget
from .budget import DEFAULT_LEDGER
from .supermimic_reference import RunStatus,write_json


def run(output,*,hours=.15):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]/'g1'
    cfg=MotionTrackingCfg(project_path=project,motion_files=(str(project/'assets/motions/tracking/walk1_subject1.csv'),),
        eval_mode=False,reset_noise=False,randomization=MotionRandomizationCfg(enabled=True,joint_offset=(.004,.004)))
    budget=TrainingBudget(DEFAULT_LEDGER,total_hours=None,run_hours=hours,metadata={'stage':'tracking_randomization_parity'})
    status=RunStatus(output,budget);cpu=gpu=None
    try:
        torch.set_num_threads(4)
        cpu=make_motion_tracking_env(cfg,num_envs=2,backend='mujoco-cpu',seed=42)
        gpu=make_motion_tracking_env(cfg,num_envs=2,backend='mujoco-warp',seed=42)
        gpu.xi.copy_(cpu.xi.to(gpu.device));gpu._apply_physics_context()
        provider=gpu.batch._provider;prefix=gpu.batch.artifact.robot_prefix(cfg.robot_name)
        torso=provider.resolve_object_ids('body',(prefix+'torso_link',))[0]
        body_ipos=provider.model_array('body_ipos')
        actual=body_ipos[:,torso]-provider.model_constant('body_ipos').to(body_ipos.dtype)[torso]
        torch.testing.assert_close(actual,gpu.xi[:,30:33],atol=1.e-7,rtol=1.e-6)
        for key in ('body_invweight0','dof_invweight0','actuator_acc0'):
            assert provider.model_array(key).shape[0]==gpu.num_envs
        for name,value in gpu.batch.physics_parameter_state()['shape_friction'].items():
            geom=provider.resolve_object_ids('geom',(prefix+name,))[0]
            torch.testing.assert_close(provider.model_array('geom_friction')[:,geom].cpu(),value,rtol=0,atol=0)
        frames=torch.tensor([200,300]);cpu.reset_to_frames(frames);gpu.reset_to_frames(frames.cuda())
        action=(cpu.batch.read_state().joint_position-cpu.default_q-cpu.xi[:,1:30])/cpu.action_scale
        errors=[]
        for step in range(8):
            if budget.remaining<10:raise TimeoutError('randomization probe deadline')
            a=cpu.step(action);b=gpu.step(action.cuda())
            errors.append(float((a.info['supermimic']['s1']-b.info['supermimic']['s1'].cpu()).abs().max()))
            if step==3:
                arrays=provider.arrays
                saved={key:arrays[key][1].clone() for key in ('qpos','qvel','ctrl','qacc_warmstart','time','qacc')}
                gpu.reset_to_frames(torch.tensor([500],device=gpu.device),[0])
                cpu.reset_to_frames(torch.tensor([500]),[0])
                for key,value in saved.items():torch.testing.assert_close(arrays[key][1],value,rtol=0,atol=0)
            status.update(status='checking',iteration=step+1)
        report={'passed':max(errors)<1.e-3,'max_state_error':max(errors),'per_step_errors':errors,
                'actual_parameter_readback_passed':True,'derived_fields_are_per_world':True,
                'masked_reset_preserves_other_solver_state':True}
        write_json(output/'randomization_parity.json',report)
        if not report['passed']:raise AssertionError('randomized CPU/Warp state difference exceeds tolerance')
        status.update(status='finished');return report
    except BaseException as error:status.update(status='failed',error=str(error));raise
    finally:
        if gpu:gpu.close()
        if cpu:cpu.close()
        status.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--hours',type=float,default=.15)
    a=parser.parse_args();run(a.output,hours=a.hours)
