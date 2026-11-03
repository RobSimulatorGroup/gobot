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

## Full SuperMimic reference (2026-10-28)

The separate `supermimic_reference.py` runner tests Uni-SuperMimic's complete
world-model/BPTT pipeline with pinned physics and all 40 motion clips. It is not
AMP. ModelScope asset preparation is provided by `reference_assets.py`; all assets
are checksummed, including the existing 18 clips after newline normalization.
See [reference configuration, validation, live status and limits](../../doc/supermimic_reference_status.md).
The local full-network benchmark and checkpoint resume passed; the bounded formal
reference run uses 512 environments and automatically evaluates full-clip coverage.
A tracking checkpoint still requires separate velocity/terrain admission.
`supermimic_video.py --watch --run-dir <run>` records fixed walking/running/sprinting
windows every 15 minutes and on completion. The current video and next recording
are listed in `<run>/videos/latest.json` and `watch_status.json`.

## Gobot SuperMimic tracking

The scene-authored port, randomized training, checkpoint recovery, frozen-actor
forks and independent validation are available through `examples.g1.train_tracking`.
Full-network smoke training has run in Gobot; full-library migration admission
is still pending. See the [G1 tracking workflow](../g1/README.md),
[measured port status](../../doc/supermimic_port_status.md), and
[cluster job instructions](cluster/README.md).
