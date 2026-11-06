"""Standard velocity PPO recipes, with explicit AMP and gait extensions.

Baseline: local mjlab dcdf6ce519abfd2ea45e9503ae6fb650b75bb869 (Apache-2.0).
Scenes and robot parameters are still owned and compiled by Gobot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

MJLAB_REFERENCE_REVISION = "dcdf6ce519abfd2ea45e9503ae6fb650b75bb869"


@dataclass
class Experiment:
    robot: str = "go1"
    recipe: str = "baseline"
    stage: str = "rough"
    seed: int = 42
    num_envs: int = 1024
    iterations: int = 10000
    hours: float = .75
    budget_hours: float = 24.
    learning_rate: float = 1.e-3
    entropy_coef: float = .01
    save_interval: int = 50
    eval_interval: int = 500
    eval_envs: int = 64
    eval_steps: int = 600
    reference_reset_probability: float = 0.
    initial_action_std: float = 1.
    motion_dataset: str = "examples/g1/assets/motions/locomotion.npz"
    style_weight: float = 2.
    eval_seed: int = 71

    def __post_init__(self):
        if self.robot not in {"g1", "go1"}:
            raise ValueError("unknown robot")
        if self.recipe not in {"baseline", "amp", "gait"} or self.stage not in {"flat", "rough", "mixed"}:
            raise ValueError("unknown training recipe or terrain")
        if self.recipe == "amp" and self.robot != "g1" or self.recipe == "gait" and self.robot != "go1":
            raise ValueError("AMP requires a G1 motion prior; the phase-gait preset is for Go1")
        if self.recipe == "baseline" and self.stage == "mixed":
            raise ValueError("standard velocity PPO uses flat or rough terrain; mixed is an explicit gait/AMP extension")
        for name in ("num_envs", "iterations", "save_interval", "eval_interval", "eval_envs", "eval_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("hours", "budget_hours", "learning_rate", "initial_action_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError("entropy_coef must be finite and nonnegative")
        if not 0 <= self.reference_reset_probability <= 1:
            raise ValueError("invalid reference reset probability")
        if self.reference_reset_probability and self.recipe != "amp":
            raise ValueError("reference-state reset belongs to the AMP recipe")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self):
        return asdict(self)


def task_config(experiment: Experiment, *, play=False):
    from gobot.rl.locomotion.config import NaturalGaitCfg
    from gobot.rl.locomotion.command import VelocityCommandStage
    from gobot.rl.training.projects import example_module, examples_root

    if experiment.robot == "g1":
        module = example_module("g1.config")
        cfg = module.g1_amp_cfg(stage=experiment.stage, play=play) if experiment.recipe == "amp" else module.g1_velocity_cfg(play=play)
    else:
        cfg = example_module("go1.train.go1_velocity_cfg").go1_velocity_cfg(
            project_path=examples_root() / "go1", play=play)

    if experiment.recipe == "baseline":
        cfg.name = f"{experiment.robot}_{experiment.stage}_velocity"
        cfg.robot_resource_path = f"res://{experiment.robot}.jscn"
        cfg.task_version = "mjlab_velocity"
        cfg.capture_transitions = True  # Optional diagnostics; standard PPO consumes only normal observations.
        cfg.post_step_sensing = False  # Reward/termination precede the post-step forward, as in the reference.
        cfg.action_clip = None
        cfg.curriculum_mode = "command_displacement"
        cfg.max_init_terrain_level = 5
        cfg.command_curriculum = (
            VelocityCommandStage(step=0),
            VelocityCommandStage(step=120000, lin_vel_x=(-1.5, 2.0), ang_vel_z=(-.7, .7)),
            VelocityCommandStage(step=240000, lin_vel_x=(-2., 3.)),
        )
        cfg.run_command_curriculum = ()
        if experiment.robot == "g1":
            cfg.scene_path = "res://g1_velocity_scene.jscn"
        if experiment.stage == "flat":
            cfg.scene_path = f"res://{experiment.robot}_flat_scene.jscn"
            cfg.observations.include_height_scan = False
            cfg.rewards.upright_reference = "world"
            cfg.terrain_curriculum = False
            cfg.terrain_out_of_bounds = False
            cfg.njmax = 300
            cfg.mujoco_solver_settings = {**cfg.mujoco_solver_settings, "convex_collision_iterations": 50}
            if experiment.robot == "go1":
                cfg.rewards.self_collisions = cfg.rewards.shank_collision = cfg.rewards.trunk_head_collision = 0.
                cfg.illegal_contact.enabled = False
                cfg.illegal_contact.terminate_on_thigh = False
                cfg.fall_angle = math.radians(70.)
        if play:
            cfg.command_curriculum = ()
        return cfg

    # Extensions are selected explicitly and do not overwrite the PPO baseline.
    cfg.robot_resource_path = f"res://{experiment.robot}.jscn"
    cfg.capture_transitions = True
    cfg.post_step_sensing = True
    cfg.action_clip = 5.
    if experiment.recipe == "gait":
        cfg.name = "go1_gait_velocity"
        cfg.task_version = "go1_gait"
        cfg.scene_path = "res://go1_flat_scene.jscn" if experiment.stage == "flat" else "res://go1_practical_scene.jscn"
        cfg.gait = NaturalGaitCfg(enabled=True, weight=.5)
        cfg.command_curriculum = cfg.run_command_curriculum = ()
        cfg.command.ranges.lin_vel_x = (-.3, 3.)
        cfg.command.rel_run_envs = .3
        cfg.command.run_velocity_x = (2., 3.)
        cfg.min_base_clearance = .12
        cfg.illegal_contact.terminate_on_trunk_contact = True
    if experiment.stage == "flat":
        cfg.terrain_curriculum = cfg.terrain_out_of_bounds = False
    if play:
        cfg.command.source = "external"
    return cfg
