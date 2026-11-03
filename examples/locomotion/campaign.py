"""Run local comparisons with stage admission and one shared GPU-hour budget."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
from .experiment import Experiment
from .train import atomic_json
from .budget import effective_cap_hours


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('examples/locomotion/logs/natural_v1'))
    p.add_argument('--budget-file',type=Path,default=Path('examples/g1/logs/gpu_budget.json'))
    p.add_argument('--additional-hours',type=float,default=6.);p.add_argument('--total-hours',type=float,default=24.)
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--seeds',type=int,nargs='+',default=(42,43))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def consumed():return json.loads(a.budget_file.read_text())['seconds']/3600 if a.budget_file.exists() else 0.
    state_file=a.output/'campaign.json'
    report=json.loads(state_file.read_text()) if state_file.exists() else {'state':'running','initial_gpu_hours':consumed(),'jobs':[]}
    cap=effective_cap_hours(a.budget_file,min(a.total_hours,report['initial_gpu_hours']+a.additional_hours))
    stopped=False;process=None
    def stop(*_):
        nonlocal stopped
        stopped=True
        if process and process.poll() is None:process.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(exp,name,initialize=None):
        nonlocal process
        previous=next((job for job in report['jobs'] if job['name']==name),None)
        directory=a.output/name
        if previous and previous.get('finished') and previous.get('summary',{}).get('stop_reason')!='signal':
            return directory if previous.get('stage_passed',previous.get('admitted')) else None
        if stopped or cap-consumed()<.025:return None
        exp=replace(exp,hours=min(exp.hours,cap-consumed()),budget_hours=cap,num_envs=a.num_envs)
        config=a.output/f'{name}.json';atomic_json(config,exp.metadata())
        job=previous or {'name':name,'config':str(config)}
        if previous is None:report['jobs'].append(job)
        command=[sys.executable,'-B','-m','examples.locomotion.train','--config',str(config),'--log-dir',str(directory),
                 '--budget-file',str(a.budget_file),'--log-stdout',str(a.output/f'{name}.stdout.log')]
        if (directory/'latest.pt').exists():command+=['--resume',str(directory/'latest.pt')]
        elif initialize:command+=['--initialize',str(initialize/'best.pt')]
        process=subprocess.Popen(command)
        report.update(state='running',active=name,active_pid=process.pid,gpu_hours=consumed(),gpu_hour_cap=cap)
        atomic_json(state_file,report)
        code=process.wait();process=None
        summary=directory/'summary.json'
        job.update(exit_code=code,finished=code==0,admitted=False)
        if summary.exists():
            result=json.loads(summary.read_text())
            job.update(summary=result,admitted=result['admitted'],stage_passed=result.get('stage_passed',result['admitted']),finished=code==0 and result['stop_reason']!='signal')
        report.update(gpu_hours=consumed());atomic_json(state_file,report)
        if code and not stopped:raise RuntimeError(f'{name} failed; inspect its status.json before restarting the campaign')
        return directory if job.get('stage_passed',job['admitted']) else None
    try:
        controls={}
        # Replicate the initial comparison before spending the budget on harder stages.
        for seed in a.seeds:
            if stopped or consumed()>=cap:break
            base=Experiment(seed=seed,hours=.5)
            controls[('g1',seed)]=launch(base,f'g1_control_s{seed}')
            launch(replace(base,robot='go1'),f'go1_baseline_s{seed}')
            controls[('go1',seed)]=launch(replace(base,robot='go1',recipe='natural'),f'go1_natural_s{seed}')
        walking={}
        for seed in a.seeds:
            base=Experiment(seed=seed,hours=.5)
            initial=controls.get(('g1',seed))
            if initial:
                common=replace(base,stage='flat_gait',reference_reset_probability=.5)
                launch(common,f'g1_ppo_gait_s{seed}',initial)
                walking[('g1',seed)]=launch(replace(common,recipe='natural'),f'g1_amp_gait_s{seed}',initial)
            initial=controls.get(('go1',seed))
            if initial:
                walking[('go1',seed)]=launch(replace(base,robot='go1',recipe='natural',stage='flat_gait'),f'go1_gait_s{seed}',initial)
        for (robot,seed),initial in walking.items():
            if not initial:continue
            base=Experiment(robot=robot,recipe='natural',seed=seed,hours=.5)
            rough=launch(replace(base,stage='rough'),f'{robot}_rough_s{seed}',initial)
            if rough:launch(replace(base,stage='mixed'),f'{robot}_mixed_s{seed}',rough)
        report['state']='stopped' if stopped else 'completed'
    except BaseException as error:
        report.update(state='failed',error=str(error));raise
    finally:
        report.update(gpu_hours=consumed());report.pop('active',None);report.pop('active_pid',None);atomic_json(state_file,report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
