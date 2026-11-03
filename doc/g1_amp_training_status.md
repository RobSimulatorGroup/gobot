# G1 locomotion experiment status — 2026-10-26

No checkpoint has passed full multi-terrain or natural-gait admission yet.
The current experiment is a PPO control used to establish reliable locomotion
before reintroducing the AMP motion reward.

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
