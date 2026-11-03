# G1 locomotion experiment status — 2026-10-28

No checkpoint has passed full multi-terrain or natural-gait admission yet.

October 28 update: the independent Uni-SuperMimic world-model/BPTT reference
finished 12,000 iterations and full evaluation normally at 16:16:57 CST. Walking,
running and sprinting completed 18/18 clips; the full library completed 33/40 with
mean normalized return 0.7318. Dance completed 6/8 and fall/get-up 3/6. This is a
reference motion-tracking result, not Gobot PPO/AMP or multi-terrain admission.
See [the reference experiment record](supermimic_reference_status.md) for complete
scores, failure locations, final videos and transfer limitations. The October 26
PPO campaign history below remains useful diagnosis.

## Natural locomotion implementation and local campaign

The new `examples/locomotion/` workflow separates G1 PPO/AMP and Go1 baseline/natural
walk-trot-gallop experiments. It fixes G1 world-upright rewards, direction-aware
curriculum, post-step Go1 sensing, final-state bootstrap, observation buffer reuse,
inference-mode clock resets and bounded log standard deviation. Candidate selection
now includes yaw and cross-track error; evaluation runs in an independent process.

G1 references exclude subject4 (validation) and subject5 (final holdout) from training.
AMP matches speed and gait, uses one discriminator update and a 500-update style
warmup. Go1 uses soft contact timing, not animal mocap. No new default playback actor
has been published.

The local `natural_v1` campaign started at approximately 14:38 CST on October 26,
2026 under `gobot-natural-locomotion-v1.service`. It starts with 1,024 G1 environments,
then schedules independent Go1 comparisons and seeds 42/43. At launch the cumulative
ledger contained 3.7375 GPU-hours. This campaign is capped at 6 additional hours
(9.7375 cumulative), below the existing 24-hour local ceiling. Stages require passed
validation before advancing. Read live status rather than interpreting this static
report as evidence that a process is still running:

```bash
uv run --no-sync python -m examples.locomotion.status examples/locomotion/logs/natural_v1
```

Verified before launch: existing 8 AMP/motion tests; new direction, gait-classification,
conditional sampling, yaw ranking and GPU-hour tests; CPU/GPU observation/reset/manifest
integration for both robots; real small PPO and AMP optimizer/checkpoint/resume/evaluation
runs on both backends; Go1 regression; and G1 CPU/Warp short-horizon parity (maximum
base-position difference 1.19e-7 m). Smoke policies were correctly rejected by evaluation.
These checks establish pipeline correctness, not gait quality or terrain admission.

The first new G1 control screening at 500 updates passed standing only: forward
0.8 m/s had 0.795 m/s tracking error. At 1,000 updates forward error was 0.816 m/s;
forward, reverse, strafe and yaw admission were still zero. The policy is primarily
standing. The 500-update eight-second renderer video completed without a fall,
but that is not a walking success. G1 AMP and terrain promotion remain gated on
actual control admission. Check the live JSON/logs for subsequent evaluations.

The portable bundle is `examples/locomotion/logs/natural-training-bundle.tar.gz`.
Its archive contents were checked against SHA-256 for every included file. The new
Go1 clock-bearing ONNX actor also passed 128 real CPU observations against Torch
(maximum absolute action difference 1.49e-7); this is inference equivalence only.

## Earlier experiment record


## Findings and fixes

- The authored G1 collision proxies lacked mjlab's four link exclusions.
  The resulting default-pose contacts generated approximately 831 N at pelvis/hip
  and 1,304 N at elbow/wrist pairs in a CPU diagnostic. `Robot3D` now stores named
  exclusion pairs; MJCF import, scene packing, and the MuJoCo compiler preserve them.
- Orientation-only termination allowed upright sitting with a pelvis height near
  0.13 m. The AMP task now requires at least 0.45 m terrain-relative clearance and
  rejects prohibited hip/thigh/torso/arm support.
- Checkpoint inference omitted the RSL observation normalizer's epsilon. The
  reconstructed actor now uses the same normalization. A 128-observation rollout
  export check measured maximum Torch/ONNX action error 4.77e-7.
- The initial discriminator saturated before useful locomotion developed. Its
  feature scaling now includes expert and policy samples. AMP still needs an
  independently evaluated improvement over the PPO control; no natural-gait
  claim is inferred from discriminator loss or reward.

## Reboot recovery and measured progress

The reboot interrupted `ppo_control_v1` at logged iteration 1422. Its complete
atomic checkpoint was iteration 1249 (1,250 updates). The pre-reboot ledger
recorded 7,458.41 seconds. Resume intentionally repeats unsaved updates, without
charging the offline interval.

`ppo_recovery_v2` resumes that checkpoint with a stronger planar tracking reward,
a fixed 3e-4 learning rate and reset exploration standard deviation 0.5. It was
launched under the user service `gobot-g1-ppo-recovery.service`, with a three-hour
job cap and shared 24-hour budget. Checkpoints are saved every 250 updates and a
fixed-command screening is performed every 500 updates.

| Completed updates | Command | Mean forward speed | Planar error | Yaw error | Survival | Screening admission |
| --- | --- | --- | --- | --- | --- | --- |
| 1,500 | 0.8 m/s | 0.597 m/s | 0.208 m/s | 0.678 rad/s | 1.0 | 0.0 |
| 2,000 | 0.8 m/s | 0.742 m/s | 0.083 m/s | 0.247 rad/s | 1.0 | 0.0 |
| 2,500 | 0.8 m/s | 0.702 m/s | 0.107 m/s | 0.300 rad/s | 1.0 | 0.0 |

These are deterministic policy screenings starting near cell centers in the
flat training scene, not final held-out terrain admission routes. At update
2000, planar tracking passes its threshold but yaw tracking does not; slip and
cadence still need improvement. Videos and JSON trajectories are recorded in
`examples/g1/logs/ppo_recovery_v2/`. Final admission also requires per-cell
traversal, held-out terrain seeds, lateral/reverse/turn/stop commands, continuous
running, and reference gait checks.

## Verification completed

- CMake built the changed engine/bindings and relevant tests. All 11 MJCF tests,
  4 robot-node tests and 51 simulation-server tests passed before the reboot.
- After the reboot: eight motion/history/termination/normalizer/gait tests,
  budget locking/offline recovery tests, GPU history integration, native G1
  integration, policy-manifest checks and the Go1 velocity regression passed.
- G1 CPU/Warp reset and four-step reward/state parity passed; maximum measured
  base position difference was 1.19e-7 m and joint position difference 1.51e-8 rad.
- The headless Gobot video path renders the same scene-compiled robot and exits
  cleanly after releasing its rendering context. It stops recording at a failure
  rather than showing automatic reset as recovery.

Use `python -m examples.g1.training_status` for current run progress. Logs and
weights are local experiment artifacts; published policy admission remains pending.
