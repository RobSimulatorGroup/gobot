"""Final held-out mobility/gait gate. Produces an export only after all tests pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from gobot.rl.locomotion.budget import TrainingBudget
from gobot.rl.locomotion.training import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-file',type=Path,required=True);p.add_argument('--budget-hours',type=float,default=24.)
    p.add_argument('--hours',type=float,default=1.);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'checkpoint':str(a.checkpoint),'protocol':'heldout_1001_2002_3003','results':[]}
    budget=TrainingBudget(a.budget_file,total_hours=a.budget_hours,run_hours=a.hours,metadata={'stage':'final_admission'})
    try:
        for seed in (1001,2002,3003):
            for stage,slope,robust in [('rough',0.,False),('flat',0.,False),('flat',5.,False),('flat',0.,True)]:
                output=a.output/f'{stage}_{slope:g}_{seed}_{"robust" if robust else "nominal"}.json'
                command=[sys.executable,'-B','-m','examples.locomotion.assessment','--checkpoint',str(a.checkpoint),
                    '--output',str(output),'--final','--seed',str(seed),'--steps','1000' if slope else '1500','--scenario-stage',stage,'--slope',str(slope),
                    '--budget-file',str(a.budget_file.resolve()),'--budget-parent-pid',str(os.getpid())]
                if robust:command.append('--robust')
                with (a.output/'admission.log').open('a',buffering=1) as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        while process.poll() is None:
                            if budget.remaining<=0:raise TimeoutError('admission budget exhausted')
                            try:process.wait(timeout=min(5.,budget.remaining))
                            except subprocess.TimeoutExpired:pass
                            budget.update(scenario=output.name)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:process.wait(timeout=30)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                if process.returncode:raise RuntimeError(f'admission subprocess failed: {output.name}')
                result=json.loads(output.read_text())
                report['results'].append({'file':str(output),'passed':result['passed'],'score':result['score']})
                atomic_json(a.output/'admission.json',report)
        report['passed']=all(r['passed'] for r in report['results'])
        report['visual_review_required']=True
        if report['passed']:
            subprocess.run([sys.executable,'-B','-m','examples.locomotion.verify_export','--checkpoint',str(a.checkpoint),
                            '--output',str(a.output/'policy.onnx')],check=True,timeout=min(300.,budget.remaining))
    except Exception as error:
        report['error']=str(error);raise
    finally:
        budget.close();atomic_json(a.output/'admission.json',report)
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
