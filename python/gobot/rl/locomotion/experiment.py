"""Robot-independent settings for bounded locomotion training runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

@dataclass
class RunConfig:
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
    motion_dataset: str | None = None
    style_weight: float = 2.
    eval_seed: int = 71

    def __post_init__(self):
        if self.recipe not in {"baseline", "amp", "gait"} or self.stage not in {"flat", "rough", "mixed"}:
            raise ValueError("unknown training recipe or terrain")
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
        if self.recipe == "amp" and not self.motion_dataset:
            raise ValueError("AMP requires an explicit motion_dataset")
        if self.reference_reset_probability and self.recipe != "amp":
            raise ValueError("reference-state reset belongs to the AMP recipe")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self):
        return asdict(self)
