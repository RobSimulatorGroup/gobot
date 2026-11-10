"""Shipped robot presets; the shared trainer accepts any typed task factory."""
from dataclasses import dataclass
import math
import sys
from gobot.rl.locomotion.experiment import RunConfig
from gobot.rl.training.projects import examples_root

MJLAB_REFERENCE_REVISION = "dcdf6ce519abfd2ea45e9503ae6fb650b75bb869"


@dataclass
class Experiment(RunConfig):
    robot: str = "go1"

    def __post_init__(self):
        if self.robot not in {"g1", "go1"}:
            raise ValueError("unknown built-in robot; supply a custom task factory")
        if self.recipe == "amp" and self.robot != "g1" or self.recipe == "gait" and self.robot != "go1":
            raise ValueError("the shipped AMP preset is for G1 and the gait preset is for Go1")
        if self.recipe == "amp" and not self.motion_dataset:
            self.motion_dataset = str(examples_root() / "g1/assets/motions/locomotion.npz")
        super().__post_init__()


def task_config(experiment: Experiment, *, play=False):
    from gobot.rl.locomotion.command import VelocityCommandStage
    from gobot.rl.training.projects import example_module, examples_root

    if experiment.robot == "g1":
        module = example_module("g1.config")
        cfg = module.g1_amp_cfg(stage=experiment.stage, play=play) if experiment.recipe == "amp" else module.g1_velocity_cfg(play=play)
    elif experiment.robot == "go1":
        cfg = example_module("go1.train.go1_velocity_cfg").go1_velocity_cfg(
            project_path=examples_root() / "go1", play=play)
    else:
        raise ValueError(f"unknown built-in robot {experiment.robot!r}; supply your own task factory")

    if experiment.recipe == "baseline":
        cfg.name = f"{experiment.robot}_{experiment.stage}_velocity"
        cfg.robot_resource_path = f"res://{experiment.robot}.jscn"
        cfg.capture_transitions = False
        cfg.action_clip = None
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
    cfg.action_clip = 5.
    if experiment.recipe == "gait":
        cfg.name = "go1_gait_velocity"
        cfg.scene_path = "res://go1_flat_scene.jscn" if experiment.stage == "flat" else "res://go1_practical_scene.jscn"
        cfg.gait = example_module("go1.train.go1_velocity_cfg").go1_gait_cfg()
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


def training_spec(*, robot="go1", recipe="baseline"):
    from gobot.rl.training.locomotion import training_spec as make_spec
    return make_spec(
        task_factory=f"{__name__}:task_config", experiment=Experiment(robot=robot, recipe=recipe),
        name=f"{robot}_{recipe}", evaluation_command=(
            sys.executable, "-B", "-c",
            "from gobot.rl.training.projects import example_module; example_module('locomotion.assessment').main()"))
