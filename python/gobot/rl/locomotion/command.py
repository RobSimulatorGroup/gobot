"""Velocity command configuration shared by locomotion backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class UniformVelocityCommandRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    heading: tuple[float, float] | None = (-math.pi, math.pi)


@dataclass
class UniformVelocityCommandCfg:
    # Forward, turn, stand, reverse, strafe, run. None selects legacy sampling.
    mode_probabilities: tuple[float, ...] | None = None
    source: str = "sampled"
    minimum_forward_velocity: float = .3
    minimum_moving_velocity: float = .1
    resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rel_standing_envs: float = 0.1
    rel_heading_envs: float = 0.3
    rel_world_envs: float = 0.0
    rel_forward_envs: float = 0.2
    rel_run_envs: float = 0.0
    run_velocity_x: tuple[float, float] = (1.5, 2.5)
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    zero_small_xy_threshold: float = 0.0
    ranges: UniformVelocityCommandRanges = field(default_factory=UniformVelocityCommandRanges)

    def __post_init__(self):
        if self.source not in {"sampled", "external"}:
            raise ValueError("command source must be sampled or external")
        if self.mode_probabilities is not None:
            p = self.mode_probabilities
            if len(p) != 6 or any(not math.isfinite(v) or v < 0 for v in p) or not math.isclose(sum(p),1.,abs_tol=1e-6):
                raise ValueError("six command probabilities must sum to one")
        if min(self.minimum_forward_velocity,self.minimum_moving_velocity) <= 0:
            raise ValueError("minimum command speeds must be positive")


def sample_categorical_commands(cfg, count, uniform, device):
    """Vectorized named-intent sampling; no overlapping heading/world masks."""
    import torch
    command = torch.zeros(count,3,device=device)
    if cfg.source == "external":
        return command, torch.full((count,),7,dtype=torch.long,device=device)
    probabilities = torch.tensor(cfg.mode_probabilities,device=device)
    modes = torch.searchsorted(probabilities.cumsum(0),uniform((count,),0.,1.)).clamp_max(5)
    r=cfg.ranges;minimum=cfg.minimum_moving_velocity
    forward=uniform((count,),cfg.minimum_forward_velocity,max(cfg.minimum_forward_velocity,r.lin_vel_x[1]))
    reverse=-uniform((count,),minimum,max(minimum,-r.lin_vel_x[0]))
    turn=uniform((count,),minimum,max(minimum,*map(abs,r.ang_vel_z)))
    side=uniform((count,),minimum,max(minimum,*map(abs,r.lin_vel_y)))
    turn *= torch.where(uniform((count,),0.,1.)<.5,-1.,1.)
    side *= torch.where(uniform((count,),0.,1.)<.5,-1.,1.)
    run=uniform((count,),*cfg.run_velocity_x)
    command[:,0]=torch.where(modes==0,forward,torch.where(modes==3,reverse,torch.where(modes==5,run,0.)))
    command[:,1]=torch.where(modes==4,side,0.)
    command[:,2]=torch.where(modes==1,turn,0.)
    return command,modes


@dataclass
class VelocityCommandStage:
    step: int
    lin_vel_x: tuple[float, float] | None = None
    lin_vel_y: tuple[float, float] | None = None
    ang_vel_z: tuple[float, float] | None = None
    run_velocity_x: tuple[float, float] | None = None


__all__ = [
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VelocityCommandStage",
]
