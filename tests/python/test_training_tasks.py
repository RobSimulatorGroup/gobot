"""The training entry registers one reference task with independent configs."""
from pathlib import Path
from gobot.rl import tasks
from gobot.rl.train import main
TASK='Gobot-Tracking-Flat-G1-SuperMimic-v0'


def test_task_factories():
    cfg=tasks.load_env_cfg(TASK,project=Path('/tmp/project'))
    play=tasks.load_env_cfg(TASK,play=True)
    assert cfg.env.domain_randomization and cfg.env.reset_noise
    assert not play.env.domain_randomization and not play.env.obs_noise and not play.env.reset_noise
    cfg.wm_trainer.lr=.2
    assert tasks.load_runner_cfg(TASK).wm_trainer.lr==2e-4


def test_cli_list(capsys):
    main(['--list'])
    assert TASK in capsys.readouterr().out
