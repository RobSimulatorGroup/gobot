# Shared velocity task tools

`train`, `evaluate`, and `export` provide one workflow for `--robot go1` and
`--robot g1`. See [G1](../g1/README.md) for commands and preparation, and
[Go1](../go1/README.md) for its existing policies and running preset.

Engine-facing Python code can construct its own typed preset:

```python
import numpy as np
from gobot.rl.locomotion import make_velocity_env
from examples.g1.config import g1_velocity_cfg

cfg = g1_velocity_cfg()
env = make_velocity_env(cfg, backend="mujoco-cpu", num_envs=64, seed=42)
try:
    obs, info = env.reset(seed=42)
    actions = np.zeros((64, 29), dtype=np.float32)
    result = env.step(actions)  # (64, 29); returns BatchEnvState
finally:
    env.close()
```

`LocomotionRobotCfg` describes named policy joints, held joints, drives, feet,
and sensor references. `VelocityTaskCfg` supplies task terms and schedules.
`terrain_grid_shape`, `terrain_type_names`, `terrain_origins`, `terrain_assignments`,
`set_terrain_assignments`, `set_command`, and `locomotion_state` expose the
inspection/evaluation boundary without requiring access to environment internals.
Use `set_terrain_assignments(levels, types, fixed=True)` to keep evaluation or
playback resets in the selected cells. NumPy CPU arrays and Torch CUDA tensors
retain their native backend.
