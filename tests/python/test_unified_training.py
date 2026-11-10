"""Training selection, overrides and early failures without launching long jobs."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from gobot.rl import tasks
from gobot.rl.train import main, resolve_spec
from gobot.rl.training import TrainingSpec, launch
from examples.locomotion.presets import training_spec

GO1 = "go1-velocity"
G1 = "g1-velocity"
AMP = "g1-amp"
SM = "g1-tracking"
ROOT = Path(__file__).resolve().parents[2]


def test_builtin_workflows_and_independent_configs(capsys):
    main(["--list"])
    output = capsys.readouterr().out
    for name in (GO1, G1, AMP, SM):
        assert name in output
        assert tasks.load_training_spec(name).supported_backends == ("mujoco-cpu", "mujoco-warp")
    spec = tasks.load_training_spec(GO1)
    first, second = spec.config_factory(), spec.config_factory()
    first.experiment.learning_rate = 1.0
    first.env["rewards"] = {"foot_slip": -0.1}
    assert second.experiment.learning_rate == 1.e-3
    assert not second.env


@pytest.mark.parametrize("task", [GO1, G1, AMP])
@pytest.mark.parametrize("backend,device", [("mujoco-cpu", "cpu"), ("mujoco-warp", "cuda:0")])
def test_backend_switch_and_dry_run_never_creates_output(tmp_path, task, backend, device):
    output = tmp_path / "must-not-exist"
    result = main([task, "--backend", backend, "--num-envs", "4", "--iterations", "2",
                   "--output", str(output), "--no-eval", "--dry-run"])
    assert result["config"]["backend"] == backend
    assert result["config"]["device"] == device
    assert result["config"]["experiment"]["num_envs"] == 4
    assert result["step_unit"] == "ppo_updates"
    assert not output.exists()


def test_reward_and_agent_overrides_are_applied_to_actual_runner(monkeypatch, tmp_path):
    from gobot.rl.locomotion import training
    captured = {}
    def run(exp, directory, **kwargs):
        captured.update(exp=exp, directory=directory, **kwargs)
        return {"completed_updates": 2}
    monkeypatch.setattr(training, "run", run)
    main([GO1, "--backend", "mujoco-cpu", "--iterations", "2", "--no-eval",
          "--output", str(tmp_path / "run"), "--set", "env.rewards.foot_slip=-0.1",
          "--set", "agent.algorithm.learning_rate=0.0002"])
    assert captured["task_cfg"].rewards.foot_slip == -0.1
    assert captured["agent_cfg"]["algorithm"]["learning_rate"] == 0.0002
    assert captured["agent_cfg"]["algorithm"]["class_name"] == "rsl_rl.algorithms.PPO"
    assert captured["exp"].iterations == 2
    assert captured["backend"] == "mujoco-cpu"
    assert captured["device"] == "cpu"


def test_builtin_defaults_use_existing_training_and_evaluator(monkeypatch, tmp_path):
    from gobot.rl.locomotion import training
    captured = {}
    monkeypatch.setattr(training, "run", lambda exp, directory, **kw: captured.update(kw))
    main([GO1, "--backend", "mujoco-cpu", "--output", str(tmp_path)])
    assert captured["task_cfg"].robot_name == "go1"
    assert captured["agent_cfg"]["algorithm"]["class_name"] == "rsl_rl.algorithms.PPO"
    assert not captured["no_eval"]


def test_experiment_recipe_and_cli_precedence(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"experiment": {"robot": "g1", "recipe": "amp",
                                "stage": "flat", "num_envs": 16}}))
    result = main([AMP, "--config", str(path), "--backend", "mujoco-cpu", "--num-envs", "4",
                   "--no-eval", "--set", "experiment.num_envs=2", "--dry-run"])
    assert result["config"]["experiment"]["num_envs"] == 2
    assert result["config"]["experiment"]["recipe"] == "amp"


def test_supermimic_keeps_schedule_and_dispatch(monkeypatch, tmp_path):
    from gobot.rl.motion_tracking import runner
    captured = {}
    def run(cfg):
        captured.update(cfg=cfg)
        return {"iteration": 45}
    monkeypatch.setattr(runner, "run", run)
    main([SM, "--config", str(ROOT / "examples/g1/train/reproduced_smoke.json"),
          "--backend", "mujoco-cpu", "--num-envs", "2", "--stop-iteration", "45",
          "--output", str(tmp_path / "sm"), "--no-compile", "--no-capture-graphs"])
    cfg = captured["cfg"]
    assert cfg.schedule.total_iterations == 320000
    assert cfg.stop_iteration == 45
    assert cfg.env.num_envs == 2
    assert cfg.device == "cpu"
    assert cfg.wm_trainer.compile is None
    assert cfg.policy_trainer.compile is None
    assert not cfg.capture_graphs
    assert cfg.log_root / cfg.run_name == tmp_path / "sm"


@pytest.mark.parametrize("args,message", [
    (["--backend", "newton"], "unsupported backend"),
    (["--backend", "mujoco-warp", "--device", "cpu"], "requires a CUDA"),
    (["--num-envs", "0"], "positive"),
    (["--set", "experiment.iteratons=2"], "unknown configuration"),
    (["--set", "env.rewards.misspelled=-1", "--no-eval"], "unknown configuration"),
    (["--set", "env.rewards.foot_slip=-0.1"], "independent evaluator"),
    (["--set", "agent.max_iterations=3"], "unsupported agent setting"),
    (["--set", "agent.algorithm.learnig_rate=0.1"], "unknown configuration field"),
    (["--set", "env.kp=[1]", "--no-eval"], "env.robot"),
    (["--resume", "a.pt", "--initialize", "b.pt"], "choose --resume"),
    (["--num-envs", "2", "--set", "experiment.num_envs=true"], "integer"),
])
def test_invalid_settings_fail_before_simulation(args, message, monkeypatch, capsys, tmp_path):
    from gobot.rl.locomotion import training
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration reached the runner")
    monkeypatch.setattr(training, "run", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        main([GO1, "--output", str(output), *args])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_distributed_amp_rejected_before_run(monkeypatch, capsys):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        main([AMP, "--dry-run"])
    assert "does not support synchronized" in capsys.readouterr().err


def test_unregistered_python_factory_and_direct_launch(monkeypatch, tmp_path):
    module = ModuleType("gobot_test_custom_training")
    captured = []
    @dataclass
    class Config:
        backend: str = "mujoco-cpu"
        device: str | None = None
        num_envs: int = 2
        seed: int = 7
        output: Path = tmp_path
        resume: Path | None = None
    module.create = lambda: TrainingSpec(Config, lambda cfg: captured.append(cfg), supported_backends=("mujoco-cpu",))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    before = tasks.list_tasks()
    main([module.__name__ + ":create", "--num-envs", "3", "--seed", "8"])
    assert captured[0].num_envs == 3 and captured[0].seed == 8
    assert captured[0].device == "cpu"
    assert before == tasks.list_tasks()
    launch(module.create())
    assert captured[1].num_envs == 2


def test_task_registration_extension_remains_supported(monkeypatch, tmp_path):
    from gobot.rl.tasks import registry
    monkeypatch.setattr(registry, "_TASKS", dict(registry._TASKS))
    original = registry.get_task(GO1)
    registry.register_task("My-Go1", replace(original, training_factory=lambda: training_spec(robot="go1", recipe="gait")))
    result = main(["My-Go1", "--backend", "mujoco-cpu", "--dry-run"])
    assert result["config"]["experiment"]["recipe"] == "gait"


def test_custom_factory_return_type_error(monkeypatch, capsys):
    module = ModuleType("gobot_test_invalid_training")
    module.create = lambda: {}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(SystemExit):
        main([module.__name__ + ":create", "--dry-run"])
    assert "must return" in capsys.readouterr().err


def test_reward_override_does_not_mutate_registered_defaults():
    spec = resolve_spec(GO1)
    config = spec.config_factory()
    original = config.resolve_env().rewards.foot_slip
    config.env = {"rewards": {"foot_slip": -99.}}
    assert config.resolve_env().rewards.foot_slip == -99.
    assert spec.config_factory().resolve_env().rewards.foot_slip == original


@pytest.mark.parametrize("task", [GO1, SM])
def test_backend_override_infers_device_after_dotted_settings(task):
    result = main([task, "--set", 'backend="mujoco-cpu"', "--dry-run"])
    assert result["config"]["device"] == "cpu"
    explicit = main([task, "--backend", "mujoco-cpu", "--device", "cuda:1", "--dry-run"])
    assert explicit["config"]["device"] == "cuda:1"


def test_standard_ppo_has_no_stage_gate_or_forced_action_clipping():
    result = main([GO1, "--terrain", "rough", "--set", "env.action_clip=null",
                   "--no-eval", "--dry-run"])
    assert result["config"]["env"]["action_clip"] is None
    assert result["config"]["experiment"]["stage"] == "rough"


def test_python_factory_works_outside_the_checkout(tmp_path):
    import os
    import subprocess
    import sys
    source = """
from gobot.rl.training.projects import example_module
cfg = example_module('locomotion.presets').training_spec(robot='go1').config_factory()
assert cfg.resolve_env().robot_name == 'go1'
assert cfg.resolve_agent()['algorithm']['class_name'] == 'rsl_rl.algorithms.PPO'
"""
    # Keep only explicit dependencies/package paths; the checkout cwd is absent.
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p and Path(p).resolve() != ROOT)
    flags = ['-S'] if sys.flags.no_site else []
    result = subprocess.run([sys.executable, *flags, '-c', source], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("task", [GO1, G1, "go1-gait"])
def test_distributed_ppo_keeps_per_rank_environment_count(monkeypatch, task):
    monkeypatch.setenv("WORLD_SIZE", "8")
    result = main([task, "--num-envs", "1024", "--dry-run"])
    assert result["config"]["experiment"]["num_envs"] == 1024
    assert tasks.load_training_spec(task).supports_distributed


def test_custom_amp_cannot_bypass_distributed_check(monkeypatch, capsys):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        main([G1, "--set", 'experiment.recipe="amp"', "--dry-run"])
    assert "AMP discriminator synchronization" in capsys.readouterr().err
