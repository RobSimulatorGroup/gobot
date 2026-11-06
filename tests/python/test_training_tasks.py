"""Registered workflow defaults are independent; registration stays optional."""
import pytest
from gobot.rl import tasks
from gobot.rl.train import main

TASK = "g1-tracking"


def test_task_factories():
    spec = tasks.load_training_spec(TASK)
    cfg = spec.config_factory()
    cfg.wm_trainer.lr = .2
    assert spec.config_factory().wm_trainer.lr == 2e-4
    assert spec.supports_distributed


def test_cli_list(capsys):
    main(["--list"])
    assert TASK in capsys.readouterr().out


def test_training_requires_an_explicit_workflow(capsys):
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert "provide a task ID or module:factory" in capsys.readouterr().err
