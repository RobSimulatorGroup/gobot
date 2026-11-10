"""Replay physics from a trusted, local version-2 locomotion failure snapshot.

Example: .venv/bin/python tests/support/replay_locomotion_failure.py failure.pt \
    --backend cpu --integrator implicit --output /tmp/replay.json

The saved model, randomization and pre-step state are restored for each failed
world, then the saved control is held for the requested number of physics steps.
This does not run a policy, rewards, pushes or episode resets. Exported model
precision and GPU contact ordering can prevent bitwise reproduction. Warp uses
automatic contact/constraint capacities, which are reported in the result.
Only load trusted snapshots: torch.load uses pickle for their metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch


STATE_FIELDS = ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")


def restore(snapshot, index, integrator):
    model = mujoco.MjModel.from_xml_string(snapshot["artifact"]["content"])
    for name, values in snapshot["model_fields"].items():
        target = getattr(model, name)
        value = values[index].numpy()
        if target.shape != value.shape:
            raise ValueError(f"model field {name} shape mismatch: {value.shape} != {target.shape}")
        target[:] = value
    if integrator != "saved":
        model.opt.integrator = {
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }[integrator]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    for name, values in snapshot["pre_step_state"].items():
        if name == "time":
            data.time = float(values[index])
            continue
        target = getattr(data, name)
        value = values[index].numpy()
        if target.shape != value.shape:
            raise ValueError(f"state field {name} shape mismatch: {value.shape} != {target.shape}")
        target[:] = value
    data.ctrl[:] = snapshot["state"]["ctrl"][index].numpy()
    return model, data


def statistics(arrays):
    finite = all(bool(np.isfinite(value).all()) for value in arrays.values())
    # JSON null reports a non-finite maximum; it never substitutes simulator state.
    def maximum(name):
        value = float(np.abs(arrays[name]).max(initial=0.0))
        return value if np.isfinite(value) else None

    return {"finite": finite, "qvel_max": maximum("qvel"), "qacc_max": maximum("qacc")}


def cpu_steps(model, data, steps):
    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)
        warnings = {
            mujoco.mjtWarning(i).name: {"count": int(w.number), "last_info": int(w.lastinfo)}
            for i, w in enumerate(data.warning) if w.number
        }
        yield {"step": step, **statistics({k: getattr(data, k) for k in STATE_FIELDS}),
               "contacts": int(data.ncon), "warnings": warnings}


def warp_steps(model, data, steps, capacities):
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import collision_convex
    from mujoco_warp._src.types import OverflowType

    # Match the provider's EPA scratch allocation before compiling kernels.
    collision_convex.MJ_MAX_EPAHORIZON = max(collision_convex.MJ_MAX_EPAHORIZON, 64)
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        warp_model = mjw.put_model(model)
        warp_data = mjw.put_data(model, data, nworld=1)
        capacities.update(nconmax=int(warp_data.naconmax), njmax=int(warp_data.njmax),
                          njmax_nnz=int(warp_data.njmax_nnz),
                          epa_horizon=int(collision_convex.MJ_MAX_EPAHORIZON))
        print("CAPACITIES", json.dumps(capacities), flush=True)
        for step in range(1, steps + 1):
            mjw.step(warp_model, warp_data)
            mask = int(warp_data.overflow.numpy()[0])
            warnings = [flag.name for flag in OverflowType if mask & flag.value]
            yield {"step": step,
                   **statistics({k: getattr(warp_data, k).numpy() for k in STATE_FIELDS}),
                   "contacts": int(warp_data.nacon.numpy()[0]),
                   "overflow": mask, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path, help="trusted local version-2 failure .pt file")
    parser.add_argument("--backend", choices=("cpu", "warp"), default="cpu")
    parser.add_argument("--integrator", choices=("saved", "implicit", "implicitfast"), default="saved")
    parser.add_argument("--steps", type=int, help="physics steps; defaults to the saved decimation")
    parser.add_argument("--output", type=Path, help="write the complete JSON report to a new file")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if snapshot.get("version") != 2:
        parser.error("expected a version-2 snapshot with pre_step_state")
    steps = snapshot["decimation"] if args.steps is None else args.steps
    if steps < 1:
        parser.error("steps must be positive")
    if snapshot.get("phase") == "post_reset_forward":
        parser.error("post-reset failures require reset replay; this tool only replays physics")
    report = {"snapshot": str(args.snapshot.resolve()), "backend": args.backend,
              "integrator": args.integrator, "steps": steps, "phase": snapshot.get("phase"),
              "mujoco_version": mujoco.__version__, "worlds": []}
    for index, env_id in enumerate(snapshot["env_ids"].tolist()):
        model, data = restore(snapshot, index, args.integrator)
        result = {"env_id": env_id, "physics_dt": float(model.opt.timestep),
                  "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
                  "initial_qvel_max": float(np.abs(data.qvel).max(initial=0.0)),
                  "capacities": {}, "trace": []}
        iterator = (cpu_steps(model, data, steps) if args.backend == "cpu" else
                    warp_steps(model, data, steps, result["capacities"]))
        for row in iterator:
            result["trace"].append(row)
            print("STEP", json.dumps({"env_id": env_id, **row}, allow_nan=False), flush=True)
        report["worlds"].append(result)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print("SUMMARY", json.dumps(report, allow_nan=False), flush=True)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
