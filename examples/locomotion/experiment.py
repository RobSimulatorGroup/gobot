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
