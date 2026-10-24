# Single-pole swing-up

Swing one pole from its hanging pose to upright with iLQR, then hold it upright
with LQR. The model comes from `cartpole.jscn`. Planning uses MuJoCo 3.17's official
`mujoco.batch.Batch`; batches evaluate candidate trajectories of the same single
pole. There is no policy training or pretrained checkpoint.

From the repository root after `uv sync`:

```bash
uv run gobot_editor --path examples/cartpole
uv run python -m examples.cartpole.cartpole_swingup --workers 4 --seed 0
```

In the editor, press Play. The hanging pose remains visible while the background
solver reports iteration costs in the Console. Once the plan is ready, the robot
swings up and balances. Pause, Stop and Reset use the normal editor controls.
Stopping cancels the solver; playback changes only the runtime scene clone.

The command-line entry point runs without a window or node scripts, solves the
same problem, then verifies four seconds of swing-up and ten seconds of balance
through Gobot's CPU world. It exits with an error if planning or balance fails.
`--headless` is accepted explicitly; `--iterations` sets the solver's iteration
limit (default 300). `--workers 0` selects automatic native thread counts.

The state is `[cart position, pole angle, cart velocity, pole angular velocity]`.
Zero angle is upright; the initial pole angle is pi. Physics runs at 240 Hz,
with six physics substeps per control update and 160 control knots (four seconds).
Actions are normalized to `[-1, 1]` and mapped to a maximum cart force of 15 N.
The hinge is passive. Scene motor limits, planning and playback share this
contract in `cartpole_task.py`.

The optimizer follows the Apache-2.0 mjbatch `cartpole_swingup.py` example's
finite-difference dynamics, regularized backward pass and parallel line search,
adapted to one pole and Gobot scene ownership. Only the animation and solver logs
are displayed; trajectory overlays are not part of this example.
