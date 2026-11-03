"""Explicit G1 reference tracking task configuration."""
from dataclasses import dataclass, field
from pathlib import Path
import math

@dataclass(frozen=True)
class MotionRandomizationCfg:
    enabled: bool = False
    sliding_friction: tuple[float, float] = (.3, 1.2)
    torso_link: str = 'torso_link'
    com_offset_min: tuple[float, float, float] = (-.025, -.05, -.05)
    com_offset_max: tuple[float, float, float] = (.025, .05, .05)
    joint_offset: tuple[float, float] = (-.01, .01)

    def __post_init__(self):
        if len(self.sliding_friction)!=2 or len(self.joint_offset)!=2 or len(self.com_offset_min)!=3 or len(self.com_offset_max)!=3:
            raise ValueError('invalid motion randomization dimensions')
        ranges = [self.sliding_friction,self.joint_offset,*zip(self.com_offset_min,self.com_offset_max)]
        if any(not all(math.isfinite(x) for x in pair) or pair[0]>pair[1] for pair in ranges):
            raise ValueError('randomization ranges must be finite and ordered')
        if self.sliding_friction[0]<0 or not self.torso_link:
            raise ValueError('invalid friction or center-of-mass link')


@dataclass(frozen=True)
class MotionTrackingCfg:
    project_path: str | Path
    motion_files: tuple[str, ...]
    robot_profile: str = 'assets/supermimic/robot_profile.json'
    scene_path: str = 'res://g1_29dof_flat_scene.jscn'
    robot_name: str = 'g1_29dof'
    base_link: str = 'pelvis'
    physics_dt: float = .005
    decimation: int = 4
    goal_frames: int = 10
    source_fps: float = 30.
    max_episode_steps: int = 500
    gamma: float = .99
    eval_mode: bool = True
    reset_noise: bool = False
    autoreset: bool = True
    split: str = 'coverage40'
    task_version: str = 'g1_tracking_v1'
    contact_links: tuple[str, ...] = ()
    deterministic: bool = False
    randomization: MotionRandomizationCfg = field(default_factory=MotionRandomizationCfg)

    def __post_init__(self):
        if not self.motion_files:
            raise ValueError('motion tracking requires at least one clip')
        if not math.isclose(self.physics_dt*self.decimation, .02, abs_tol=1.e-9):
            raise ValueError('reference tracking requires a 50 Hz control clock')
        if self.goal_frames < 1 or self.max_episode_steps < 1 or not 0 < self.gamma < 1:
            raise ValueError('invalid tracking horizon or discount')
        if self.eval_mode and self.reset_noise:
            raise ValueError('evaluation must disable reset noise')
        if self.eval_mode and self.randomization.enabled:
            raise ValueError('evaluation must disable physics randomization')
        if self.split not in ('coverage40', 'train', 'validation', 'holdout'):
            raise ValueError('unknown motion split')
