# G1 SuperMimic reference experiment — 2026-10-28

The full **world-model + BPTT** implementation has completed its first local test separately from
Gobot's PPO/AMP recipes. This experiment uses the local Uni-SuperMimic implementation,
not IsaacLab SuperMimic, and is not a claim that a deployable multi-terrain policy
has been trained.


## Completed result — October 28, 16:16:57 CST

The reference run finished all 12,000 iterations and its full-library evaluation
normally. Training took 1 h 52 min; initialization, training and evaluation took
1 h 58 min altogether. Both training and periodic-video services exited successfully.

| Motion group | Full clips completed | Mean normalized return |
|---|---:|---:|
| Walking | 12/12 | 0.8502 |
| Running | 4/4 | 0.8046 |
| Sprinting | 2/2 | 0.8116 |
| Dance | 6/8 | 0.7071 |
| Fall and get up | 3/6 | 0.5117 |
| Jumps | 2/3 | 0.7602 |
| Fight/sports | 4/5 | 0.6443 |
| Whole library | 33/40 | 0.7318 |

Frame-weighted return is 0.7425; total frame coverage is 91.98%. All 18 walking,
running and sprinting clips completed from their first frames, totaling 73.44
minutes of simulated motion. This exceeds what the eight-second video previews
alone establish. The planned return >=0.82 and 40/40 full-clip target has **not**
been met. The termination is a reference-relative pelvis height/tilt error threshold;
a failure flag does not always mean the robot physically fell.

The seven failed clips terminate at: dance1_subject1 107.22 s,
dance2_subject4 60.90 s, fallAndGetUp1_subject1 9.12 s,
fallAndGetUp1_subject5 57.24 s, fallAndGetUp2_subject3 157.74 s,
fightAndSports1_subject1 12.58 s, jumps1_subject2 234.34 s.

The final artifacts are under the run directory: `evaluation.json`,
`results_summary.json`, `ckpt_012000.pt`, full-resume `last.pt`, and
`videos/gait_windows/iteration_012000/comparison.mp4`. Final visual inspection shows
walking follows the reference reasonably closely; running/sprinting retain visible
arm, orientation and path differences. No foot-slip metric was collected by this
reference evaluator, so the result is not a no-slip claim. The run uses the full
training library, MuJoCo 3.11 and independent Uni-SuperMimic; it has not been ported
to Gobot and does not establish held-out generalization or terrain competence.

The final visual audit additionally replayed two failures continuously from the
first frame, then cropped video around the event. `audit_final/comparison.mp4`
contains a successful dance preview, a failed dance and a failed fall/get-up clip.
The independent dance replay terminates at 61.10 s (canonical full evaluation:
60.90 s), with tilt-component error 0.8079 versus threshold 0.8. Fall/get-up
terminates at 9.10 s (canonical: 9.12 s), with height error 0.2550 m versus the
0.25 m threshold. These repeat checks confirm the failure category while showing
small replay timing differences; canonical scores remain those in `evaluation.json`.
The visual audit does not change the full-library completion counts.

The accepted implementation plan remains: first port this motion tracker faithfully
to Gobot and verify physics/observation equivalence; then use verified walking/running
teachers for student distillation and PPO/AMP task/terrain adaptation. The failed
motion groups need targeted improvement and full-clip re-evaluation before admission.

## Data and implementation

- Reference: `Motphys/Uni-SuperMimic`, commit
  `faf7818e8abac3ec6abb8f00e6e904d3577f800b`.
- Original checkout remains unchanged. The experiment copy and isolated environment
  are under `examples/locomotion/logs/supermimic_reference/`.
- ModelScope: `AI-ModelScope/LAFAN1_Retargeting_Dataset`, pinned revision
  `cd6548b049528d19012d92bc36d23cb27dec0f85`.
- Download/validation tool: `examples/locomotion/reference_assets.py`.
- Validated 76 assets: 40 G1 motion CSVs, 264,705 source frames at 30 Hz, 36 columns,
  35 meshes and one URDF. All downloaded/reused files match the mirror SHA256.
- The existing 18 walking/running/sprinting CSVs match after CRLF-to-LF conversion;
  their numeric contents are unchanged. The URDF also matches the Hugging Face copy.
- Upstream asset validation confirms 29 natural-order hinges, pelvis free joint,
  14 tracked frames, joint limits, PD gains, armature, offsets and scene parameters.

Pinned reference physics: MuJoCo 3.11.0, MuJoCo Warp 3.11.0, Warp 1.16.0,
UniLab 1.5.1, UniSim Core 2.1.1. Torch is the existing **2.11.0+cu128**, shared via
an isolated-environment `.pth`; this differs from the reference lock. Gobot's
MuJoCo 3.17 environment is not imported into the reference process.

UniLab 1.5.1 identifies a robot using any `robots/g1` path segment, including an
external project's path, and starts downloading its own unrelated asset directory.
The bounded runner skips that fallback only for the verified local reference asset
tree. Learning equations, robot physics, motion data and reward terms are unchanged.

## Validation already completed

| Check | Result |
|---|---|
| Upstream default tests | 83 passed, 2 skipped, 6 slow tests deselected |
| Asset validation | All 40 clips, expected frame count and robot numerics passed |
| GPU full network smoke | 64 environments, 6 updates, world model and policy backward, checkpoint and evaluation passed |
| GPU compiled benchmark | 512 environments/batches, 12 updates, full networks, horizon 32; latest iteration 0.526 s |
| Benchmark memory | Peak Torch allocated 3.20 GiB; reserved 3.54 GiB (not total driver usage) |
| Full checkpoint resume | Restored world model, actor, critic and optimizers; update 12 → 13 passed |
| Evaluation coverage regression | A non-falling clip truncated early is not counted as a full clip |
| Gobot regressions | 15 native tests, CPU/Warp external-command reset and low-speed clock checks passed; CPU PPO/AMP resume and 3 PPO guard tests passed |

Short smoke checkpoints failed 40/40 full motion clips. They validate the execution
path, not gait quality. No smoke checkpoint is admitted or installed for playback.

## Formal local run

Started **2026-10-28 14:19:05 CST**:

- Service: `codex-supermimic-g1-20261028.service`.
- Run: `examples/locomotion/logs/supermimic_reference/g1_reference_s42_512/`.
- Seed 42; 512 simulation environments, world-model batch 512, policy batch 512.
- Full world model `[2048, 2048, 2048]`, policy `[2048, 1024, 512, 256]`,
  critic `[256, 128]`, history 10, horizon 32, mixed precision bf16, compilation on.
- 500 world-model-only iterations, then joint world-model/policy/critic training;
  target 12,000 iterations.
- Up to **3 GPU hours**, including initialization and end evaluation. Reserve 20
  minutes for evaluation if the training reaches the wall-time limit.
- Full `last.pt` and numbered policy/model checkpoints every 250 iterations;
  additional full saves after 10 minutes and on controlled stop.
- Heartbeat and GPU budget updates continue during long compilation. A watchdog
  ends an unresponsive process at the budget deadline, preserving earlier checkpoints.
- Live truth: `status.json`, `train_log.csv`, `console.log`, checkpoint files and
  the systemd service. This document's startup record is not proof of current liveness.

The reference default has 4,096 environments, world-model batch 2,048 and policy
batch 4,096. This local run retains the network/horizon but changes sample counts;
its results must not be presented as an exact default-scale reproduction. Cluster
runs should record and compare those differences explicitly.

The ledger is `examples/g1/logs/gpu_budget.json`. This experiment window began at
24,258.961823352 seconds (~6.7386 GPU hours). The additional authorization is 8 GPU
hours, with a cumulative cap of min(24, 6.7386 + 8) hours. The frozen window is
`examples/locomotion/logs/natural_v2/budget_window.json`; restarting a process does
not reset it. Setup failures and benchmark time are charged too.

To inspect:

```bash
systemctl --user status codex-supermimic-g1-20261028.service
cat examples/locomotion/logs/supermimic_reference/g1_reference_s42_512/status.json
tail examples/locomotion/logs/supermimic_reference/g1_reference_s42_512/train_log.csv
```

The bounded entry point is `examples/locomotion/supermimic_reference.py`, run with
`examples/locomotion/logs/supermimic_reference/.venv/bin/python`. Resume requires
`--resume <run>/last.pt` and the same `--run-dir <run>`, source, assets and semantic
configuration; `--iterations` is the new total target. Upstream resumes optimizer
state but refills replay and resets physical environments, so continuation is not
bit-for-bit replay of an uninterrupted run. Stop with the service's normal SIGTERM
path so a full checkpoint can be saved. The transient service is independent of
this conversation but does not automatically resume after a computer reboot.

## Interpretation and next admission

After training, deterministic evaluation uses one environment per clip, first-frame
resets and no observation/action noise. `evaluation.json` contains normalized return,
falls, actual/expected steps, full-clip coverage and separate walk/run/sprint groups.
All 40 clips are training-library clips in this reference experiment; unlike the
Gobot AMP split, this is **not held-out generalization**.

Full motion tracking is not velocity-command or terrain admission. The next decision
must use actual full-clip tracking results and video, then separately test a command
interface, held-out motion/commands and rough terrain. No default playback policy has
been replaced. The corrected Gobot PPO/AMP branch remains experimental: CPU optimizer,
resume and KL rollback tests passed, but it has not begun a new long training run;
stage-capability gating and episode-collapse recovery still need completion before
launching its next campaign.


## Periodic gait videos (added October 28)

`codex-supermimic-video-20261028.service` runs the independent
`examples/locomotion/supermimic_video.py` watcher. It records the latest atomically
saved checkpoint immediately, then every 15 minutes after each recording finishes;
when training ends it records the final checkpoint and exits. It does not restart
or modify training. Like the training service, it is transient across reboot.

Each recording includes three fixed 8-second active-motion windows:
`walk1_subject1` at 4 s, `run1_subject2` at 32 s, and `sprint1_subject2` at 8 s.
These reference-state resets deliberately skip the source files' standing preludes;
they are previews rather than full-clip evaluation. The solid robot is the learned
policy; the blue ghost is the reference in world coordinates, without pelvis
realignment. Failure ends the clip and retains the terminal physical state, not the
automatically reset state. Two video regression tests check checkpoint selection,
fixed starts and inclusion of the failure state.

Outputs live under `<run>/videos/gait_windows/iteration_NNNNNN/`: three MP4s,
`comparison.mp4`, `preview.jpg` and a metadata manifest. `<run>/videos/latest.json`
points to the latest group; `watch_status.json` records the next recording time.
The first active-motion group at iteration 2000 completed all three 8-second previews
without a termination. The H.264 montage is 960 × 540, 25 fps and 24.12 seconds.
This is not a full-clip success or terrain admission. Generation took about 28 seconds
on the same GPU concurrently with training; its overlapping wall time is already
counted by the training ledger. A standalone final recording acquires its own budget.
The earlier `videos/iteration_001750/` group starts at source time zero and includes
the original preparation poses; use the `gait_windows` groups for future comparisons.
