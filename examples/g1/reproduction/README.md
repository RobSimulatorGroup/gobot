# Pinned SuperMimic reproduction tools

These are independent diagnostic tools, not a second training framework. The
algorithm/runtime migration is tracked in
[the alignment plan](../../../doc/supermimic_reproduced_alignment_plan.md) and
[the progress record](../../../doc/supermimic_reproduced_progress.md).

`reference.lock.json` pins the two source commits and the SHA256/size of all three
published release assets. Checkouts, environments, exported meshes, large weights
and reports live under ignored `examples/g1/logs/reproduced_alignment/`.

The source's original environment uses MuJoCo 3.16. Gobot requires native mjBatch
from 3.17, so original-result validation and same-version 3.17 comparisons are
separate experiments. The current completed checks use Torch 2.11, MuJoCo 3.17,
MuJoCo Warp 3.17 and Warp 1.17. They do not establish reproduction on the original
Torch 2.14 / Warp 1.18 release stack.

All commands below run from the repository root. `REFERENCE_PYTHON` means the
independent interpreter with the pinned sources importable; `GOBOT_PYTHON` means
an interpreter using the rebuilt Gobot package. Set these to the environments
for the specific original or paired experiment, without mixing native MuJoCo
versions in one process.

```bash
# Freeze verified source archives, release identity and reconstructed eval lists.
python -m examples.g1.reproduction.reference \
  --reference REFERENCE_CHECKOUT --mjlab MJLAB_CHECKOUT \
  --release RELEASE_DIRECTORY --output BASELINE_DIRECTORY

# The reference process does not import Gobot.
"$REFERENCE_PYTHON" -m examples.g1.reproduction.export_model --output REFERENCE_EXPORT

# Author and then reload .jscn before comparing compiled physical parameters.
"$GOBOT_PYTHON" -m examples.g1.prepare_tracking \
  --reference-export REFERENCE_EXPORT --project-path AUTHORED_PROJECT

"$GOBOT_PYTHON" -m examples.g1.reproduction.check_kinematics \
  --project AUTHORED_PROJECT --output FK_REPORT.json

"$GOBOT_PYTHON" -m examples.g1.reproduction.check_controls \
  --project AUTHORED_PROJECT --output CPU_REPORT.json

"$GOBOT_PYTHON" -m examples.g1.reproduction.check_controls \
  --project AUTHORED_PROJECT --backend mujoco-warp \
  --reference-backend mujoco-warp --output WARP_REPORT.json
```

The export invokes the real reference environment configuration and mjlab Scene
builder, but does not initialize unrelated registered tasks or a simulator.
Namespace normalization is checked against the in-memory model before the
export is accepted. Runtime randomization is not baked into the exported XML.

`check_controls` defaults to an independent CPU MuJoCo oracle. The optional Warp
oracle uses `mujoco_warp` directly on the reference export, not Gobot's provider.
The diagnostic uses two environments, 32 control steps, deterministic Warp, and
capacities 128/500. These are recorded and are not the published training recipe's
35/250 capacity settings. CPU-versus-Warp and Warp-versus-Warp reports must not be
substituted for each other.

```bash
# Independent numerical fixture; package initializers are bypassed, not math code.
"$REFERENCE_PYTHON" -m examples.g1.reproduction.math_fixture \
  --reference REFERENCE_CHECKOUT --mjlab MJLAB_CHECKOUT --output math_reference.pt

# Fixed-input, full-size released actor comparison in separate processes.
"$REFERENCE_PYTHON" -m examples.g1.reproduction.check_release_policy reference \
  --reference REFERENCE_CHECKOUT --mjlab MJLAB_CHECKOUT \
  --checkpoint RELEASE_CHECKPOINT --output actor_reference.pt
"$GOBOT_PYTHON" -m examples.g1.reproduction.check_release_policy candidate \
  --checkpoint RELEASE_CHECKPOINT --fixture actor_reference.pt --output actor_parity.json
```

The checked-in small numerical fixture includes its content digest, exact imported
source hashes and software versions. It covers observations, fixed-scale actor /
critic, eight reward terms, 8-step world-model loss, gradients and one Adam update.
It is not a real-motion rollout or a complete BPTT training-loop test.

The published `config.json` belongs to the resumed 92k segment. Learning rates
are already scaled to 0.0004; `max_hours=0.8` is not the full experiment duration.
The released checkpoint records 100000 control iterations and 39968 updates.
The published report contains 129943 clips; applying the pinned name filter yields
129491 clips and 145 failures. Neither list proves the original training library's
bytes or membership; the original training manifest is still unavailable.

Additional training checks now exercise the active implementation:

```bash
# Independent replay + full WM, BPTT and critic/Adam update fixture.
"$REFERENCE_PYTHON" -m examples.g1.reproduction.training_fixture \
  --reference REFERENCE_CHECKOUT --mjlab MJLAB_CHECKOUT --output training_reference.pt

# Same-stack closed-loop traces, with optional --compact for fp16 motion storage.
"$REFERENCE_PYTHON" -m examples.g1.reproduction.check_task --mode reference \
  --project AUTHORED_PROJECT --data WALK.npz --checkpoint RELEASE_CHECKPOINT --output reference_task.pt
"$GOBOT_PYTHON" -m examples.g1.reproduction.check_task --mode gobot \
  --project AUTHORED_PROJECT --data WALK.npz --checkpoint RELEASE_CHECKPOINT --output gobot_task.pt

# Untouched reference complete-clip benchmark (teacher evaluation, not fresh training).
"$REFERENCE_PYTHON" -m examples.g1.reproduction.reference_benchmark \
  --data WALK.npz --checkpoint RELEASE_CHECKPOINT --output reference_benchmark.json
```

The training fixture uses a deterministic analytic rig to isolate the optimizer and
replay math. The task trace and complete-clip benchmark use actual G1 physics. Do
not substitute one for the other. The original release environment has now been
installed, while the same-stack traces/compiled learner tests use Torch 2.11,
MuJoCo 3.17 and Warp 1.17. Frozen source hashes and runtime versions accompany
reference outputs.

The complete single-world walk triggered a capacity overflow at the reference
recipe's 35/250 budget. The larger-capacity diagnostic uses `--nconmax 128
--njmax 500` on both `reference_benchmark` and `examples.g1.evaluate_tracking`.
The failure remains recorded; larger-capacity results do not certify the formal
4096-environment/rank capacity settings. The paired 13066-step walk finished with
no falls on either side; returns were 0.6029567138 (reference) and 0.5995487229
(Gobot). This is one released-actor clip, not a from-scratch reproduction.

Further diagnostics added on 2026-11-04:

```bash
# Same randomized physical context and executed actions (4 lanes, 24 controls).
"$REFERENCE_PYTHON" -m examples.g1.reproduction.check_context reference \
  --project AUTHORED_PROJECT --data WALK.npz --checkpoint RELEASE_CHECKPOINT --output context.pt
"$GOBOT_PYTHON" -m examples.g1.reproduction.check_context candidate \
  --project AUTHORED_PROJECT --data WALK.npz --fixture context.pt --output context_comparison.json

# Actual G1 FK and recorded transitions, full networks, H=32, one learning update.
"$REFERENCE_PYTHON" -m examples.g1.reproduction.check_update reference \
  --reference REFERENCE_CHECKOUT --mjlab MJLAB_CHECKOUT --project AUTHORED_PROJECT \
  --data WALK.npz --transitions context.pt --output real_update.pt
"$GOBOT_PYTHON" -m examples.g1.reproduction.check_update candidate \
  --project AUTHORED_PROJECT --data WALK.npz --transitions context.pt \
  --fixture real_update.pt --output real_update_comparison.json

# Add --compact --episode-length .2 to check_task for repeated auto-resets.
# Validate complete-clip counts and preserve every disagreement before comparing.
python -m examples.g1.reproduction.compare_benchmarks \
  --reference REFERENCE_REPORT.json --candidate GOBOT_REPORT.json --output comparison.json
```

The randomized-context check injects reference context/action noise into Gobot;
it does not prove identical random-number consumption. The real-update check uses
full WM/policy/critic sizes and a small batch of four, with an eight-step WM loss,
32-step imagination and 5×4 critic updates. It does not prove convergence.

All 40 LAFAN1 clips have now been converted (441140 frames), with original CSV,
NPZ and packed-library digests recorded. The fixed BONES teacher evaluated on
these same bytes, with 40 lanes and diagnostic 128/500 capacity, produced:
reference mean return 0.6235608319 / 21 failures; Gobot 0.6266288506 / 19 failures.
Six clips disagree on failure status. These are cross-dataset teacher results,
not the LAFAN1-from-scratch experiments or the BONES published benchmark.
