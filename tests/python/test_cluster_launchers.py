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


@pytest.mark.parametrize('processes,task,seed', [
    (None, 'go1-velocity', '42'),
    ('2', 'g1-velocity', '7'),
])
def test_velocity_launch_trains_one_shared_policy(environment, processes, task, seed):
    environment.update(GOBOT_REPO=environment['GOBOT_PROJECT_ROOT'],
                       GOBOT_OUTPUT='/tmp/velocity runs', GOBOT_TASK=task,
                       GOBOT_SEED=seed)
    if processes:
        environment['GOBOT_NPROC_PER_NODE'] = processes
    result = subprocess.run(
        ['bash', str(ROOT / 'examples/locomotion/cluster/train_velocity.sbatch'),
         '--terrain-reference', '/tmp/shared terrain.json'],
        env=environment, capture_output=True, text=True, check=True,
    )
    args = json.loads(result.stdout)
    assert args[:8] == ['-B', '-m', 'torch.distributed.run', '--standalone',
                       f'--nproc-per-node={processes or "8"}', '-m', 'gobot.rl.train', task]
    assert args[args.index('--seed') + 1] == seed
    assert args[args.index('--output') + 1] == '/tmp/velocity runs'
    assert args[args.index('--budget-file') + 1] == '/tmp/velocity runs/budget.json'
    assert args[-2:] == ['--terrain-reference', '/tmp/shared terrain.json']
