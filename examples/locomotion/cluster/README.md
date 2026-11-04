# SuperMimic-Reproduced G1 jobs

`train_tracking.sbatch` launches **four synchronous GPU ranks** for one seed.
Each rank owns 4096 environments and its replay; model gradients are averaged.
Different seeds are separate jobs. The formal recipe uses full networks and the
published e8 batch settings, with 100000 control steps on a 320000-step LR schedule.

Build/install Gobot for the cluster's Python and native MuJoCo 3.17 stack first.
Prepare the scene with `examples.g1.prepare_tracking` and the motion library with
`gobot.rl.motion_tracking.convert`; see [the complete workflow](../../../doc/gobot_training.md).

```bash
export GOBOT_PROJECT_ROOT=/work/gobot
export GOBOT_SCENE_PROJECT=/data/g1_project
export GOBOT_MOTION_LIBRARY=/data/bones/library.pt
export GOBOT_MOTION_DATA=/data/lafan1_npz
export GOBOT_RUN_ROOT=/work/results
sbatch examples/locomotion/cluster/pipeline_smoke.sbatch
GOBOT_SEED=0 sbatch examples/locomotion/cluster/train_tracking.sbatch
```

The scripts reserve one node with four GPUs, 32 CPU cores and 128 GB host RAM.
Use a compatible cluster partition and sufficient GPU memory; no formal setting
is reduced to fit the development RTX 4070. These are launch templates, not proof
that a cluster job has run. Original BONES training data identity and paired
multi-seed evaluation remain prerequisites for a reproduction claim.
