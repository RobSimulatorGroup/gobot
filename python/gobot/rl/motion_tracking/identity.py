"""Content identity of the source modules that define a tracking experiment."""
from importlib.util import find_spec
from pathlib import Path
from .convert import digest


def source_identity():
    modules = ['gobot.rl.supermimic.'+name for name in (
        'config','compiled','distributed','dynamics','geometry','heads','motion_convert','motion_library',
        'objective','policy','policy_trainer','replay','rewards','rollout','rotation_primitives','rotations',
        'sampler','schedule','state_spec','wm_trainer','world_model')]
    modules += ['gobot.rl.motion_tracking.'+name for name in ('env','kinematics','runner','training','convert','identity')]
    modules += ['gobot.rl.articulation','gobot.rl.kinematics','gobot.sim.providers.base','gobot.sim.providers.mujoco_warp']
    return {name:digest(Path(find_spec(name).origin)) for name in modules}
