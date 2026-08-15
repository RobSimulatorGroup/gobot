# Dual-FR3 Rope Twisting

Two facing Franka Research 3 robots friction-grasp square fixtures connected by
three elastic strands. Both wrists rotate while the arms repeatedly pull apart
and relax. Editor Play defaults to a stiff `12 N m` speed-drive showcase, so the
requested wrist rate does not gradually fall as rope reaction grows. The
headless experiment keeps a hard `0.125 N m` limit; with the checked-in
coarse-rope parameters, its expected physical stall window is 15 to 25 relative
wrist turns.

This is a two-way co-simulation task rather than two unrelated simulations in
one scene:

1. MuJoCo Warp closes four finger joints onto two free 6-DoF fixtures. There is
   no weld between a gripper and fixture; the grasp is carried by box-on-pad
   contact with Coulomb friction.
2. The fixture poses drive two libuipc affine proxies. Six
   `DeformableAttachment3D` nodes attach one end section of each tetrahedral
   strand to the corresponding proxy.
3. libuipc advances elasticity, large deformation, self/inter-strand contact,
   friction, and CCD, then exports the exact force and torque on each fixture.
4. The default Newton proxy coupling rolls both solvers back to the same
   microstep start, performs two Aitken-relaxed interface iterations, and
   commits one physical microstep. The resulting wrench passes through the
   friction grasps and opposes both robot wrists.

The robots use the same positive local `fr3_joint7` velocity command. Because
their bases face each other, this counter-rotates the fixtures about the common
world rope axis. The first six joints move both tools about `8.3 mm` outward
and back on every breathing cycle, so the view shows axial stretch, relaxation,
bending, and torsion together.

In the finite-torque experiment, the controller does not stop at a scripted
angle. It declares a physical stall only after both libuipc reaction torques
exceed 75% of the drive limit, both wrist actuators remain near their torque
limit, and both measured wrist speeds remain below `0.08 rad/s` for 150
consecutive fixed steps. The velocity request remains active during the stall
hold, making the lack of motion an actuator/rope result rather than an authored
pause.

Gravity compensation removes only the two robots' known gravity bias. It does
not cancel fixture contact or libuipc reaction wrenches. Wrist passive damping
is reduced for this task so most of the finite drive torque is available to
twist the rope instead of being consumed by the imported arm's nominal damping.

## Run In The Editor

Regenerate the checked-in scene after editing its parameters:

```bash
uv run python examples/dual_arm_rope_twist/build_scene.py
```

Open `examples/dual_arm_rope_twist/project.gobot`, enter Play Mode, and keep the
whole workcell in view. The status line reports phase, relative wrist turns,
measured strand winding, wrist speeds, actuator efforts, fixture slip, mount
error, rope reaction torque, and the active drive mode. Press `P` to restart the
physical cycle.

Editor Play defaults to `newton_proxy` with two Aitken-relaxed coupling
iterations. The compatibility/performance baseline remains available with
`GOBOT_ROPE_TWIST_INTEGRATION_SCHEME=sequential_split`.

The Physics panel's `Contact force arrows` toggle also controls the runtime
force overlay. Magenta arrows show libuipc contact forces on rope vertices, green
arrows show MuJoCo pad/fixture contact forces, and the existing amber/cyan
arrows show the two wrist reaction torques. The panel's force scale and maximum
length settings apply to both contact-force sources. MuJoCo's contact-frame
force components are transformed through the reported normal and tangent basis
before the green arrows are drawn in world coordinates. Nonzero rope-contact
arrows use a 15 mm minimum display length so weak IPC forces remain visible;
their directions and labels still use the unscaled world-space force in newtons.
Every rope vertex above the 0.001 N display threshold is drawn without a
strongest-contact count limit.

The default `showcase` mode still computes and applies MuJoCo/libuipc reaction
forces; it only raises the wrist torque cap to the stock FR3 joint limit. To run
the slower, torque-limited stall experiment in the editor instead:

```bash
GOBOT_ROPE_TWIST_DRIVE_MODE=finite-torque \
  uv run gobot_editor --path examples/dual_arm_rope_twist
```

Play Mode searches local build directories for a compatible native module. To
force one explicitly:

```bash
export GOBOT_LIBUIPC_SOLVER_MODULE="$PWD/build/<matching-build>/python/gobot/libgobot_libuipc_solver.so"
```

## Run Headless

Run batched GPU trials with the same scene and controller:

```bash
uv run python examples/dual_arm_rope_twist/rope_twist_batch.py \
  --num-envs 2 --environments-per-shard 1
```

Use `--drive-mode showcase` for the same `12 N m` constant-speed mode used by
Editor Play. The default remains `finite-torque` so automated runs still measure
a physical stall. Batch runs also default to `newton_proxy` with two coupling
iterations; use `--integration-scheme sequential_split` for the legacy
single-pass baseline.

The default upper bound is a 40,000-step twist followed by a hold, but the run
exits early after every environment has physically stalled. The JSON result
contains actual wrist turns and speeds, actuator efforts, strand winding,
fixture contact forces, grasp slip, attachment error, and both requested and
executed step counts.

Useful ablations:

```bash
# Remove the gripper/fixture friction. The free fixtures should no longer stay
# registered in the jaws under load.
uv run python examples/dual_arm_rope_twist/rope_twist_batch.py \
  --grip-friction-scale 0

# Keep libuipc motion but remove its reaction wrench from MuJoCo. A physical
# torque stall is intentionally disabled in this one-way ablation.
uv run python examples/dual_arm_rope_twist/rope_twist_batch.py \
  --coupling-feedback-scale 0
```

For this demonstration, each wrist joint limit is widened to
`[-200 pi, 200 pi]`, or 100 turns in either direction. This task-specific limit
allows continuous rotation; it is not the stock FR3 safety limit.

## Scope

The fixture, pad, and rope parameters are chosen for a stable, inspectable
demonstration and are not calibrated to a commercial rope or gripper. libuipc's
implicit contact solve makes this accuracy-oriented example substantially more
expensive than rigid-only MuJoCo. The important contract is structural: robot
contact moves free fixtures, fixtures deform the rope, and the resulting soft
reaction changes the robot motion until finite actuator torque is insufficient.
