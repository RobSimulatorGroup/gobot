# Panda trajectory optimization

Open the project from the repository root and click **Play** in the top toolbar.

```sh
.venv/bin/gobot_editor --path examples/panda_trajopt
```

The green and purple markers specify the start and goal for the Panda hand.
Once the CPU planner finishes, the viewport plays both motions on the same
clock at quarter speed, holding briefly at each end.

| Viewport | Meaning |
| --- | --- |
| Translucent pink Panda and pink curve | Initial joint trajectory and its hand path. The hand cuts through the orange wall. |
| Solid Panda and cyan curve | Optimized joint trajectory and its hand path. The hand rises above the wall. |

Use **Pause** to inspect the two poses at one instant. **Stop** restores the
authored scene. Move the `Wall`, `Start` or `Goal` node and press **Play** to
solve again. An infeasible plan is reported in the Console.

Each play compiles a Gobot physics snapshot and calls `Planner.plan_poses` on
a worker process. The solver uses 50 waypoints at 0.02 s spacing and checks
collision clearance against three 4.5 cm spheres on the base, elbow and hand.
The visualization computes forward kinematics for both the solver's initial
waypoints and its accepted result. The Null physics backend provides
kinematic playback, and Stop also cancels an in-flight solve.

The Panda joint frames and limits are adapted from the Apache-2.0 Genesis
`panda_bullet/panda.urdf` at revision `ffd119c30118ebb9745fc009349efb2112a8cb76`.
The primitive link visuals are authored in `panda.jscn`. The wall task is based
on [PyRoki's TrajOpt example](https://github.com/chungmin99/pyroki/blob/388e43e1fc0d0ee382968d3dd72970fd62a0450c/examples/07_trajopt.py).
