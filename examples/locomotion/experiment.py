"""Versioned recipes for comparable single-GPU locomotion experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Experiment:
    schema_version: int = 2
    robot: str = 'g1'
    recipe: str = 'baseline'
    stage: str = 'flat_control'
    seed: int = 42
    num_envs: int = 1024
    iterations: int = 10000
    hours: float = .75
    budget_hours: float = 24.
    learning_rate: float = 3e-4
    entropy_coef: float = .003
    save_interval: int = 100
    eval_interval: int = 500
    eval_envs: int = 64
    eval_steps: int = 600
    reference_reset_probability: float = 0.
    normalization_warmup_steps: int = 64
    guarded_updates: bool = True
    initial_action_std: float = .5
    stop_on_gate: bool = False
    motion_dataset: str = 'examples/g1/assets/motions/locomotion.npz'
    style_weight: float = 2.
    eval_seed: int = 71
    patience: int = 3

    def __post_init__(self):
        if self.schema_version not in {1,2} or self.robot not in {'g1','go1'}:
            raise ValueError('unsupported experiment schema or robot')
        if self.recipe not in {'baseline','natural'} or self.stage not in {'flat_control','flat_gait','rough','mixed'}:
            raise ValueError('unknown training recipe or stage')
        if min(self.num_envs,self.iterations,self.save_interval,self.eval_interval,self.eval_envs,self.eval_steps) < 1:
            raise ValueError('experiment counts must be positive')
        if self.hours <= 0 or self.budget_hours <= 0 or not 0 <= self.reference_reset_probability <= 1:
            raise ValueError('invalid training budget or reference reset probability')
        if self.schema_version == 1 and self.recipe == 'natural' and self.robot == 'g1' and self.stage == 'flat_control':
            raise ValueError('G1 AMP starts at flat_gait after flat_control admission')

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self):
        return asdict(self)


def task_config(experiment: Experiment, *, play=False):
    from gobot.rl.locomotion.config import NaturalGaitCfg
    stage = 'flat' if experiment.stage.startswith('flat') else experiment.stage
    if experiment.robot == 'g1':
        from examples.g1.config import g1_amp_cfg
        cfg = g1_amp_cfg(stage=stage, play=play)
    else:
        from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
        cfg = go1_velocity_cfg(project_path=Path(__file__).parents[1]/'go1', play=play)
        cfg.name = 'go1_natural_velocity'
        cfg.task_version = 'go1_natural_velocity_v1'
        cfg.robot_resource_path = 'res://go1.jscn'
        cfg.scene_path = 'res://go1_flat_scene.jscn' if stage == 'flat' else 'res://go1_practical_scene.jscn'
        cfg.capture_transitions = True
        cfg.post_step_sensing = True
        cfg.gait = NaturalGaitCfg(enabled=True,weight=.5 if experiment.recipe=='natural' else 0.,active_threshold=.05 if experiment.schema_version==2 else .1)
        cfg.action_clip = 5.
        cfg.curriculum_mode = 'command_progress'
        cfg.min_base_clearance = .12
        cfg.illegal_contact.terminate_on_trunk_contact = True
        cfg.rewards.bound_gait = cfg.rewards.run_progress = 0.
        cfg.domain_randomization.foot_friction_slide_range = (.4, 1.4)
        cfg.push_interval_range_s = (5.,10.)
        cfg.push_velocity_ranges = {'x':(-.3,.3),'y':(-.3,.3),'z':(0.,0.),'roll':(0.,0.),'pitch':(0.,0.),'yaw':(0.,0.)}
    # All expansion is stage-gated, not tied to a number of PPO updates.
    cfg.command_curriculum = cfg.run_command_curriculum = ()
    cfg.command.ranges.lin_vel_x = (-.3, 1.2)
    cfg.command.ranges.lin_vel_y = (-.3,.3)
    cfg.command.ranges.ang_vel_z = (-.6,.6)
    cfg.command.rel_run_envs = 0.
    cfg.command.rel_forward_envs = .6
    cfg.command.rel_standing_envs = .1
    cfg.command.rel_heading_envs = .2
    cfg.max_init_terrain_level = 0
    cfg.max_terrain_level = None
    if stage == 'flat':
        cfg.max_terrain_level = 0
        cfg.domain_randomization.enabled = cfg.push_enabled = False
    if experiment.stage == 'flat_control':
        cfg.command.ranges.lin_vel_x = (-.1,1.)
        cfg.command.ranges.lin_vel_y = (-.15,.15)
        cfg.command.ranges.ang_vel_z = (-.3,.3)
    if experiment.stage in {'flat_gait','mixed'}:
        upper = 2.2 if experiment.robot == 'g1' else 3.
        cfg.command.ranges.lin_vel_x = (-.3,upper)
        cfg.command.rel_run_envs = .3
        cfg.command.run_velocity_x = (1.6,upper) if experiment.robot == 'g1' else (2.,upper)
    if stage == 'mixed':
        cfg.running_terrain_types = (0,3,4)
        cfg.max_running_terrain_level = 1  # authored maximum <=5 degrees at these levels
    # Stable base control reward; AMP and phase terms are the only recipe differences.
    cfg.rewards.track_linear_velocity = 4.
    cfg.rewards.track_angular_velocity = 4.
    cfg.rewards.foot_slip = -.2
    cfg.rewards.air_time = 0.
    if experiment.schema_version == 2:
        cfg.task_version = 'g1_amp_velocity_v4' if experiment.robot=='g1' else 'go1_natural_velocity_v2'
        cfg.command.mode_probabilities = (.7,.1,.1,.05,.05,0.) if experiment.robot=='g1' and experiment.stage=='flat_control' else (.3,.2,.1,.2,.2,0.)
        cfg.command.heading_command = False
        cfg.command.minimum_forward_velocity = .4 if experiment.stage=='flat_control' and experiment.robot=='g1' else .3
        if experiment.stage=='flat_control' and experiment.robot=='g1':cfg.command.ranges.lin_vel_x=(-.1,.8)
        if experiment.stage in {'flat_gait','mixed'}:cfg.command.mode_probabilities=(.2,.15,.1,.15,.15,.25)
        cfg.command.source = 'external' if play else 'sampled'
        cfg.rewards.adaptive_tracking = True
        cfg.rewards.command_progress = 2.
        cfg.rewards.body_lin_vel_z = -.5
        cfg.rewards.track_linear_velocity = 6. if experiment.robot=='g1' else 4.
        cfg.rewards.track_angular_velocity = 2.
        if experiment.robot=='g1':
            cfg.rewards.moving_pose_scale = .25
            cfg.rewards.action_rate_l2 = -.02 if experiment.stage=='flat_control' else -.05
        cfg.observations.actor_noise_ranges = dict(cfg.observations.actor_noise_ranges)
        cfg.observations.actor_noise_ranges['base_lin_vel'] = (-.1,.1)
    return cfg
