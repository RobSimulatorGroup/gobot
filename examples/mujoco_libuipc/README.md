# MuJoCo Warp + libuipc GPU batch

This example runs a batch of independent soft-body press environments on one
CUDA device. MuJoCo Warp owns the prismatic press articulation and rigid
dynamics. libuipc owns FEM deformation and every contact pair involving the
soft block. `MuJoCoIpcCoupler` exchanges rigid poses and reaction wrenches once
per fixed step.

## Editor Play Mode

Open the project and press Play in the top toolbar:

```bash
uv run gobot_editor --path examples/mujoco_libuipc
```

The attached `mujoco_libuipc_play.py` starts four GPU environments through an
external `ProviderPlaySession`. It compiles the authored scene once, then adds
three display-only runtime copies after compilation. The viewport shows all
four environments in a 2x2 grid with different press depths and deformation.
The copies do not run scripts, enter the physics artifact, or modify the
checked-in scene. The Physics panel should report `Provider: MuJoCo+libuipc`,
`Device: cuda:0`, and `Environments: 4`; the backend selector is disabled while
this session owns simulation.

The script discovers the usual in-tree solver build automatically. Override it
when necessary:

```bash
GOBOT_LIBUIPC_SOLVER_MODULE=/absolute/path/libgobot_libuipc_solver.so \
  uv run gobot_editor --path examples/mujoco_libuipc
```

Press `P` or the Physics panel Reset button to restart the compression cycle.
The first launch may pause while Warp compiles CUDA kernels; progress is shown
in the Console, and subsequent launches reuse the cache.

## Headless Batch

The authored source scene is `soft_press_batch.jscn`. Regenerate it with:

```bash
uv run python examples/mujoco_libuipc/build_scene.py
```

Its two explicit `PhysicsCoupling` nodes make the ground a `OneWay` pose proxy
and the press head a `TwoWay` proxy. Collision layers and masks still select
which libuipc deformables may contact those proxies.

Run the default four-environment batch without the editor:

```bash
uv run python examples/mujoco_libuipc/mujoco_libuipc_batch.py
```

The default trajectory uses 128 coupled steps so the press reaches the soft
block. Environments receive different press depths, so the final JSON reports
a range of joint positions, soft-block compression, and reaction forces. It
also checks stable CUDA storage and demonstrates deterministic full-batch
reset.

Run a short topology/throughput smoke test with four 64-environment shards:

```bash
uv run python examples/mujoco_libuipc/mujoco_libuipc_batch.py \
  --num-envs 256 --environments-per-shard 64 --steps 24
```

Record median results at the admission capacities with warmup and repeated
runs:

```bash
uv run python benchmark/mujoco_libuipc_batch_benchmark.py \
  --module-path "$PWD/build/libuipc-novcpkg/python/gobot/libgobot_libuipc_solver.so"
```

The default benchmark preserves the sequential IPC baseline at 4, 64, and 256
environments. Pass a prior JSON report as `--baseline-json`; the
256-environment run then fails if throughput regresses by more than 10 percent.
To record the Newton/AL-IPC correctness matrix, run:

```bash
uv run python benchmark/mujoco_libuipc_batch_benchmark.py \
  --environment-counts 1 64 \
  --integration-scheme newton_proxy \
  --coupling-iterations 1 2 4 \
  --contact-constitutions ipc al-ipc
```

Results include median environment-steps/s, per-shard IPC latency, interface
residual, Aitken coefficient, penetration and wrench proxies, storage
stability, feedback source, reset scope, and graph state. The A/B report only
marks AL-IPC eligible after a 15-percent median IPC step speedup and all listed
physical bounds pass; standard IPC remains the default.

For a non-installed solver module, provide its exact path:

```bash
uv run python examples/mujoco_libuipc/mujoco_libuipc_batch.py \
  --module-path "$PWD/build/libuipc-novcpkg/python/gobot/libgobot_libuipc_solver.so"
```

Useful options:

- `--num-envs`: fixed environment capacity.
- `--environments-per-shard`: isolated libuipc subscenes per native world.
- `--steps`: coupled fixed steps.
- `--warmup-steps`: unmeasured warmup steps before timing.
- `--repeats`: number of timed runs used for the median.
- `--press-depth`: final prismatic target in meters, limited to `0.17`.
- `--fixed-dt`: shared MuJoCo/libuipc timestep.
- `--integration-scheme`: `sequential_split` (default) or rollback-based
  `newton_proxy`.
- `--coupling-iterations`: interface iterations per Newton microstep.
- `--relaxation-mode`: fixed or bounded Aitken interface relaxation.
- `--contact-constitution`: standard `ipc` or experimental `al-ipc`.
- `--no-mujoco-graph`: disable MuJoCo Warp CUDA graph capture for debugging.
- `--rebuild-scene`: regenerate the `.jscn` before running.

The native batch C ABI is v2, while the compiled IPC/composite artifact is
schema v4 with schema-v3 read compatibility. The composite provider requires
fixed topology and full-batch reset. Newton mode additionally requires equal
rigid/IPC microsteps and uses a single-slot native checkpoint for rollback; any
number of interface iterations still commits one physical microstep. A partial
reset is rejected because shard-local history cannot yet be restored
selectively. The composite provider reports `graph_capture=false` because
libuipc shards are stepped outside capture; its MuJoCo Warp subsolver can still
replay a captured graph. libuipc currently stages the small affine pose/twist
table D2H/H2D once per shard and step. The Play display adds batched CPU
readback only in its viewport callback; the headless batch path does not
perform that display synchronization.
