"""BeyondMimic motion tracking on Gobot scene-owned articulations.

Task equations adapted from mujocolab/mjlab dcdf6ce519abfd2ea45e9503ae6fb650b75bb869.
The task does not import mjlab or own a separate MuJoCo model.
"""

from .api import train_motion
from .config import TrackingConfig
from .env import TrackingEnv

__all__ = ["TrackingConfig", "TrackingEnv", "train_motion"]
