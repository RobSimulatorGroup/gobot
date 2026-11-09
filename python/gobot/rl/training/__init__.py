"""Headless training workflows, usable with or without task registration."""
from .workflow import TrainingSpec, config_dict, launch, update_config

__all__ = ["TrainingSpec", "config_dict", "launch", "update_config"]
