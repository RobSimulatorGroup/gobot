"""Exercise shell GPU allocation handling without submitting or running training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def environment(tmp_path):
    root = tmp_path / 'project with spaces'
    root.mkdir()
    python = root / 'mock python'
    python.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'GOBOT_'))}
    env.update(GOBOT_PROJECT_ROOT=str(root), GOBOT_PYTHON=str(python),
               GOBOT_SCENE_PROJECT=str(root / 'scene project'),
               GOBOT_MOTION_DATA=str(root / 'motion data'),
               GOBOT_MOTION_LIBRARY=str(root / 'motion data/library.pt'))
    return env


@pytest.mark.parametrize('script,default', [('train_tracking.sbatch', 4), ('pipeline_smoke.sbatch', 1)])
@pytest.mark.parametrize('allocation,override', [(None, None), ('1', None), ('4', None), ('8', None), ('8', '4')])
def test_launch_matches_allocation_and_keeps_arguments(environment, script, default, allocation, override):
    if allocation:
        environment['SLURM_GPUS_ON_NODE'] = allocation
    if override:
        environment['GOBOT_NPROC_PER_NODE'] = override
    expected = override or allocation or str(default)
    result = subprocess.run(['bash', str(ROOT / 'examples/locomotion/cluster' / script),
                             '--output', '/tmp/result with spaces'], env=environment,
                            check=True, capture_output=True, text=True)
    args = json.loads(result.stdout)
    assert args[:6] == ['-m', 'torch.distributed.run', '--standalone',
                       f'--nproc-per-node={expected}', '-m', 'gobot.rl.train']
    assert args[args.index('--project') + 1] == environment['GOBOT_SCENE_PROJECT']
    assert args[-2:] == ['--output', '/tmp/result with spaces']


@pytest.mark.parametrize('script', ['train_tracking.sbatch', 'pipeline_smoke.sbatch'])
@pytest.mark.parametrize('extra', [
    {'GOBOT_NPROC_PER_NODE': '0'}, {'GOBOT_NPROC_PER_NODE': 'eight'},
    {'SLURM_GPUS_ON_NODE': '4', 'GOBOT_NPROC_PER_NODE': '8'},
    {'SLURM_JOB_NUM_NODES': '2'},
])
def test_invalid_launch_fails_before_python(environment, script, extra):
    result = subprocess.run(['bash', str(ROOT / 'examples/locomotion/cluster' / script)],
                            env={**environment, **extra}, capture_output=True, text=True)
    assert result.returncode == 2 and result.stderr and not result.stdout
