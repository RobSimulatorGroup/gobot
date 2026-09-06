# Two-hand Mailer Acceptance: MuJoCo Warp + libuipc

## Scope and ownership

This is a task-level headless acceptance runner, not a new engine backend or
editor playback implementation. `conveyor_mujoco_ipc.py` compiles a private
selection of `conveyor_packages.jscn` through `CompiledMuJoCoIpcArtifact` and
reuses `MuJoCoIpcProvider` / `SolverCoupledProxy`. No second robot or bag asset
is authored in a backend-specific format. Offline fingertip fitting uses the
MJCF runtime artifact compiled from that selection, only for kinematics.

The selection keeps both LEAP hands, the blue film and tetrahedral fill, the
worktable, and the stationary outfeed conveyor. It contains 44 driven joints,
1,820 shell vertices, 780 fill vertices, and 36 explicitly coupled links.
All 34 hand collision links return exact native contact wrenches to MuJoCo;
the two stationary supports use one-way pose transfer. MuJoCo owns rigid-rigid
contacts; IPC owns soft-rigid and soft-soft contacts. Shell self-contact stays
enabled. No attachments or externally prescribed package motion are allowed.

Only joint position targets are submitted during the trial. There is no
conveyor traction or additional package force controller. Source mesh,
thickness, density, elastic coefficients, and bending parameters are retained;
the provider's constitutive/contact laws are not identical to SuperDex's.
The JSON records the compiled material data, solver settings, source-scene
hash, controller/measurement hashes, and artifact digest.

## Reproduction

Run from the repository root with the installed CUDA/libuipc-enabled package:

```bash
uv run --no-sync python examples/conveyor_packages/conveyor_mujoco_ipc.py \
  --report build/benchmarks/conveyor-ipc-acceptance/contact.json
```

Defaults are one environment, 2 ms fixed steps, two rollback-based coupling
iterations per step, strict solver convergence, 48 Newton iterations and
16 line-search trials. The full blue sequence is 3,020 motion ticks, plus up to
1,000 waiting ticks with the default contact controller. A failed gate
produces `passed=false` and exit code 1 even if `error=null` and the solvers
completed every tick. Short runs cannot pass. `--compile-only` does not advance
physics and is explicitly labelled `compiled_not_simulated`.

The default candidate fits a family of bounded joint poses with opposed,
parallel fingertip pads from a 50 mm opening to a 2.5 mm nominal collision-box
gap. The thumb is aligned before descending. All three regular fingers share
parallel joint targets. Wrist translations place the pinch centers at measured
crests on the settled upper sheet; wrist rotation is not added.

During descent and acquisition, material-patch tracking is limited to 50 mm/s
and a 60 mm radius around the initial measured center. A loaded hand stops
tracking. The controller closes each hand until thumb and all three regular
fingers have opposing reactions above 0.15 N. Loads above 20 N cause closure
to relax. Both hands must hold continuously for 100 ms with proxy error below
1 mm before the lift trajectory can advance. A lost pinch for 20 ms halts
carrying. Acquisition timeout is recorded as `task_failure`, separately from
solver `error`. Extra waiting ticks cannot count as sequence completion.

`--controller open-loop` retains the original ungated trajectory, and adding
`--pinch-gap 0.0205` checks its original SuperDex gap. Grasp fitting/tracking
only changes joint commands; it never attaches vertices or moves the package.
Use `--snapshot-dir PATH` for compressed NPZ state snapshots at trace boundaries.

## Acceptance gates

| Measurement | Requirement |
| --- | --- |
| Execution | Complete all 3,020 ticks, finite state, no solver error |
| Feedback | Native exact contact-wrench capability, two-way hand mappings |
| Pinch | Both hands: thumb and each of index/middle/ring distal links carry at least 0.01 N; thumb opposes the sum of the other three forces |
| Lift | All package nodes at least 20 mm above the support plane while bilateral pinch persists for 100 ms |
| Flip | Authored upper-sheet area normal turns 180 degrees, within 10 degrees |
| Release | Final 100 ms without hand contact, speed at every node at most 0.05 m/s, flipped, within 10 mm of support |
| Proxy tracking | Collision-surface displacement upper bound at most 1 mm |
| Table penetration | No package vertex more than 1 mm below the support plane |
| Precontact drift | Shell vertex centroid moves at most 0.5 mm horizontally before first hand contact, including initial settling |
| Wrist | Palm orientation changes at most 10 degrees from its initial orientation |

The measurements deliberately distinguish a press from a pinch and a lift/drop
from a flip. Reaction vectors are distal-link net contact forces, not individual
contact pairs. This conservative gate can reject a weak grasp whose opposing
components are obscured by other loads; inspect the stored vectors before
changing the controller or thresholds.

Proxy error is a bound on target/proxy surface displacement using authored
link geometry radii, including rotation and affine shear. It is **not measured
hand/bag penetration**. The table test is vertex-based and assumes the support
plane, rather than testing the bounded table/conveyor triangles. The face-normal
test follows labelled material, not a PCA sign, but is not a certificate that a
highly folded bag has turned as a whole. Passing these gates would still need
geometric intersection checks and visual confirmation before demo admission.

## Performance interpretation

The runner records synchronized `provider.step()` latency, including coupling
iterations, separately from command submission and per-tick acceptance readback.
Open-loop targets are uploaded once; contact-controlled targets are submitted
each step. Per-phase median/p95 values include all
successful ticks, without warmup exclusion; total trial wall time includes
the measurements. Native diagnostic timings in phase samples describe the
sampled step, not an average over the phase.

GPU memory is whole-device used memory sampled at trace boundaries. It includes
other processes and can miss intra-step peaks. It is not per-environment VRAM
consumption and must not be divided into 12 GB to infer a batch capacity.

These are single-environment diagnostic runs, not an apples-to-apples benchmark
against the full 6,292-node SuperDex workcell. Current composite capabilities
report no full-step CUDA Graph and only full-batch reset. Carton flipping,
independent reset and batch throughput are not admitted by this test.

## Pre-friction-fix result: 2026-09-06

The original open-loop full-cycle run on RTX 4070 **failed physical acceptance**, with
`error=null` and all 3,020 steps completed. The local machine-readable record is
`build/benchmarks/conveyor-ipc-acceptance/blue-full-x2.json`.
It used MuJoCo 3.10.0, mujoco-warp 3.10.0.2, Warp 1.15.0 and Torch 2.11.0+cu128.

| Result | Observed |
| --- | --- |
| First hand contact | Tick 841 |
| Bilateral opposed pinch | Never detected |
| Sustained airborne pinch | 0 seconds |
| Final authored-face rotation | 1.449 degrees |
| Precontact shell-centroid drift | 23.740 mm, including settling |
| Maximum proxy surface displacement bound | 0.572 mm |
| Maximum palm rotation | 0.139 degrees |
| Table vertex penetration | 0 mm; limited measurement, not a penetration certificate |
| Synchronized step median / p95 | 93.73 / 292.78 ms |
| Grip-phase step median / p95 | 202.98 / 530.84 ms |
| Trial wall time for 6.04 simulated seconds | 364.47 seconds |
| Sampled whole-device memory peak | Approximately 3.47 GiB, including other processes |

At tick 900 both thumbs and all regular distal links had nonzero contact force,
but every distal net reaction had a positive vertical component and neither
hand passed the opposition test. By tick 1050 all fingertip reactions were zero
and the bag remained on the table. This is consistent with pressing the bag
down and losing contact when lifting, not successful thumb/finger clamping.

The tick-900 coupling diagnostic spent 419.77 ms in IPC advancement across the
two coupling iterations, versus 0.88 ms in rigid advancement. The final IPC
iterate reported 10 Newton iterations and 475 total PCG iterations. This sample
locates the dominant cost in IPC for that contact state; it is not a measured
phase-wide breakdown or evidence that all states have the same bottleneck.
CPU contract tests/configuration briefly overlapped the run, so timings remain
diagnostic and are not a controlled comparative benchmark.

The original 20.5 mm gap also failed in an earlier full-cycle diagnostic.
Reducing it to 2.5 mm made thumb contact possible but did not establish a grasp.
No acceptance thresholds were relaxed to turn these failures into passes.

These historical trials predate the friction-history fix below. Their timings
must not be treated as a baseline with working friction, and the historical
JSON files are kept unchanged. The subsequent `contact-v1.json` and
`contact-v2.json` acquisition trials also failed, stopping at their grasp timeout
without lifting; neither constituted a successful pinch.

## Coupling friction regression and fix

`UploadAffineTargetsAndTwists()` calls the SDK's velocity-only affine accessor
each step. That accessor incorrectly requested a vertex-attribute refresh,
whose geometry-reset path discards all lagged friction candidates. As a result,
coupled scenes lost friction history on every tick, including stationary
table/soft-body contact. Changing the number of coupling iterations did not
avoid it. The host staging path also unnecessarily wrote unchanged transforms.

Velocity-only writes now preserve contact history in both paths. Transform
writes still invalidate it and re-synchronize persistent joint coordinates.
Both the SDK and Gobot native module must be rebuilt, not just the Python code.

`test_real_libuipc_velocity_staging_preserves_friction` applies horizontal
gravity of 1 m/s^2 to an initially falling soft block over a stationary support.
After 120 steps of 5 ms, the broken device path displaced the block 180.909 mm,
with almost no tangential support reaction. With the fix, displacement is
2.420 mm (including initial falling motion) and the last 20 ticks' mean
support reaction is 5.400 N. Device and host staging, each with and without
checkpoint rewind, all pass the same gates. The machine-readable regression
record is `build/benchmarks/conveyor-ipc-acceptance/friction-regression.xml`.

## Contact-controlled rerun after the fix

The RTX 4070 run in `contact-friction-fixed.json` used the default contact
pose family, a +6 mm crest height offset, two coupling iterations and a
500-tick acquisition wait budget. No acceptance threshold was relaxed.
It **still failed the full task**, but advanced beyond the earlier empty grasp:

| Result | Observed |
| --- | --- |
| First hand contact | Step 724 |
| First bilateral opposed load | Step 1001 |
| Sustained acquisition confirmation | 100 ms; lift starts at step 1210 |
| Stop | Step 1242: right-hand grasp lost during lift |
| Sustained airborne pinch | 0 seconds |
| Final authored-face rotation | 9.452 degrees, not a flip |
| Precontact shell-centroid drift | 0.603 mm, still above the 0.5 mm gate |
| Maximum proxy displacement bound | 0.252 mm |
| Maximum palm rotation | 0.042 degrees |
| Solver error | None; physical task failure reported separately |

This ran 1,242 physical steps but only advanced to motion tick 992, so the
sequence-completion gate remains false. The controller stopped instead of
continuing the empty lift/flip trajectory. The immediate manipulation issue is
maintaining finger preload while the load transfers from the table to the
hands; table-supported opposing reactions alone did not establish an airborne
grasp. The small remaining precontact drift also needs its own validation.

Median/p95 synchronized step latency was 108.92/625.65 ms; grip-phase latency
was 152.56/702.29 ms. These remain diagnostic, not throughput acceptance:
the run ends during lift, friction is now active, and CPU contract tests briefly
overlapped its early phases. Do not compare them as a speedup/slowdown against
the earlier full trajectory with broken friction history.

After a successful single-bag grasp, still verify actual hand/bag intersections
and whole-package turnover, then run the carton and batch-throughput trials.
Neither the friction regression nor successful safety gating admits the demo
as a working flip controller.
