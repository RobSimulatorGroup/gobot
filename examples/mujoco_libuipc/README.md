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
external `ProviderPlaySession`. The viewport displays environment 3, which has
the deepest press command, while all four environments continue stepping. The
Physics panel should report `Provider: MuJoCo+libuipc`, `Device: cuda:0`, and
`Environments: 4`; the backend selector is disabled while this session owns
simulation.

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

Run the default four-environment batch without the editor:

```bash
uv run python examples/mujoco_libuipc/mujoco_libuipc_batch.py
```

The default trajectory uses 128 coupled steps so the press reaches the soft
block. Environments receive different press depths, so the final JSON reports
a range of joint positions, soft-block compression, and reaction forces. It
also checks stable CUDA storage and demonstrates the v1 full-batch reset.

Run a short topology/throughput smoke test with four 64-environment shards:

```bash
uv run python examples/mujoco_libuipc/mujoco_libuipc_batch.py \
  --num-envs 256 --environments-per-shard 64 --steps 24
```

For a non-installed solver module, provide its exact path:

```bash
uv run python examples/mujoco_libuipc/mujoco_libuipc_batch.py \
  --module-path "$PWD/build/libuipc-novcpkg/python/gobot/libgobot_libuipc_solver.so"
```

Useful options:

- `--num-envs`: fixed environment capacity.
- `--environments-per-shard`: isolated libuipc subscenes per native world.
- `--steps`: coupled fixed steps.
- `--press-depth`: final prismatic target in meters, limited to `0.17`.
- `--fixed-dt`: shared MuJoCo/libuipc timestep.
- `--no-mujoco-graph`: disable MuJoCo Warp CUDA graph capture for debugging.
- `--rebuild-scene`: regenerate the `.jscn` before running.

Batch v1 requires a full reset and fixed topology. The composite provider does
not claim graph capture; its MuJoCo Warp subsolver can still replay captured
graphs. libuipc currently stages the small affine-target table through host
memory once per shard and step.
