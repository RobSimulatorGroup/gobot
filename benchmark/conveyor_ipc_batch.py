"""Isolated-process 1/2/4/8 LEAP + mailer contact-stage throughput baseline."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/conveyor_packages"))
import conveyor_mujoco_ipc as trial


def process_memory_rows(output):
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("unexpected nvidia-smi process memory columns")
        pid, memory = int(row[0]), int(row[1])
        if pid <= 0 or memory < 0:
            raise ValueError("invalid process memory accounting")
        rows.append((pid, memory * 1024 ** 2, row[2].strip().removeprefix("GPU-").lower()))
    return rows


class ProcessMemorySampler:
    """Sample NVML's per-process accounting through nvidia-smi, not Torch's allocator."""
    def __init__(self, pid, gpu_uuid, interval=1.):
        self.pid, self.gpu_uuid, self.interval = pid, gpu_uuid.removeprefix("GPU-").lower(), interval
        self.peak = self.samples = 0
        self.other_pids = set()
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                                     "--format=csv,noheader,nounits"], check=True,
                                    capture_output=True, text=True, timeout=5)
            values = process_memory_rows(result.stdout)
            own = [memory for pid, memory, uuid in values if pid == self.pid and uuid == self.gpu_uuid]
            if own:
                self.samples += 1
                self.peak = max(self.peak, max(own))
            self.other_pids.update(pid for pid, _, uuid in values if pid != self.pid and uuid == self.gpu_uuid)
        except Exception as error:
            self.error = str(error)

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return {"sampled_process_peak_bytes": self.peak if self.samples else None,
                "samples": self.samples, "interval_seconds": self.interval,
                "pid": self.pid, "gpu_uuid": self.gpu_uuid,
                "other_compute_pids": sorted(self.other_pids),
                "error": self.error or (None if self.samples else "no matching GPU process samples"),
                "scope": "NVML compute-process memory, includes Warp/IPC/Torch/context; sampled, not an exact allocation peak."}


def timing(values):
    return {"samples": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def run_worker(args):
    if (args.num_envs < 1 or args.steps < 1 or args.warmup_steps < 0
            or not np.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0.):
        raise ValueError("invalid batch, sample count, warmup, or wall-time limit")
    count = args.num_envs
    report = {"schema_version": 1, "num_envs": count, "error": None, "benchmark_completed": False,
              "workload": "two_LEAP_blue_mailer_contact_acquisition", "grasp_acceptance_passed": False,
              "warmup_steps": args.warmup_steps, "requested_sample_steps": args.steps,
              "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
              "fixed_dt": trial.FIXED_DT, "source_sha256": {
                  path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                      Path(__file__), ROOT / "examples/conveyor_packages/conveyor_pinch_controller.py",
                      ROOT / "examples/conveyor_packages/conveyor_mujoco_ipc.py")}}
    context = provider = sampler = None
    provider_ms, total_ms = [], []
    completed_steps = 0
    with tempfile.TemporaryDirectory(prefix="gobot-conveyor-batch-") as workspace:
        try:
            context, artifact, triangles = trial.load_trial(args.scene)
            poses, fits = trial.contact_pinch_poses(artifact, .0025)
            normals = np.asarray([fit["fingertip_normals_world"] for fit in fits[-1]])
            report["artifact_digest"] = artifact.digest
            properties = torch.cuda.get_device_properties(0)
            report["gpu"] = properties.name
            report["device_total_bytes"] = properties.total_memory
            uuid = str(properties.uuid)
            sampler = ProcessMemorySampler(os.getpid(), uuid)
            with sampler:
                build_start = time.perf_counter()
                provider = trial.MuJoCoIpcProvider(artifact, config=trial.MuJoCoIpcConfig(
                    num_envs=count, environments_per_shard=count, coupling_iterations=2),
                    libuipc_config=trial.LibuipcBatchConfig(solver=trial.LibuipcConfig(
                        fixed_time_step=trial.FIXED_DT, workspace=workspace, module_path=args.module_path,
                        contact_activation_distance=.0008, friction_coefficient=1., kinematic_strength=100.),
                        environments_per_shard=count, newton_max_iterations=48,
                        line_search_max_iterations=16, strict_convergence=True),
                    mujoco_options={"nconmax": 512, "njmax": 2048, "overflow_check_interval": 1})
                provider.synchronize()
                report["build_seconds"] = time.perf_counter() - build_start
                report["solver_config"] = asdict(provider.ipc_solver.config)
                report["capabilities"] = asdict(provider.capabilities)
                arrays = provider.arrays
                views = [provider.create_robot_view(robot_name=name, base_link=base, joint_names=joints,
                                                   link_names=trial.LEAP_CONTACT_LINK_NAMES)
                         for name, base, joints in zip(trial.LEAP_ROBOT_NAMES, trial.HAND_BASE_LINK_NAMES,
                                                       trial.HAND_JOINT_NAMES_BY_SIDE, strict=True)]
                body = next(body for body in provider.ipc_solver.deformable_bodies
                            if str(body["path"]).endswith("/" + trial.PACKAGE_NAMES[0]))
                offset, size = int(body["element_offset"]), int(body["element_count"])
                shells = arrays["ipc_positions"][:, offset:offset + size].cpu().numpy().copy()
                settings = trial.PinchControlSettings(maximum_wait_steps=1000)
                controllers = [trial.ContactPinchController(poses, shell, triangles, trial.BLUE_SEGMENTS,
                               trial.FIXED_DT, settings, fingertip_normals=normals) for shell in shells]
                hand_ids = [m.ipc_body_index for m in artifact.coupled_bodies if m.robot_name in trial.LEAP_ROBOT_NAMES]
                tip_ids = [[next(m.ipc_body_index for m in artifact.coupled_bodies
                                 if m.robot_name == name and m.link_name == tip) for tip in trial.TIP_NAMES]
                           for name in trial.LEAP_ROBOT_NAMES]
                radii_map = trial.collision_link_radii(artifact)
                radii = np.asarray([radii_map[m.ipc_path] for m in artifact.coupled_bodies
                                    if m.robot_name in trial.LEAP_ROBOT_NAMES])
                command = torch.empty((count, 2, 22), device="cuda:0", dtype=arrays["ctrl"].dtype)
                contact_steps = np.zeros(count, dtype=int)
                pinch_steps = np.zeros(count, dtype=int)
                maximum_proxy_error = 0.
                started = time.perf_counter()
                for tick in range(args.warmup_steps + args.steps):
                    if time.perf_counter() - started > args.max_wall_seconds:
                        raise TimeoutError("batch wall-time limit reached")
                    start = time.perf_counter()
                    commands = np.stack([controller.command(shell) for controller, shell in zip(controllers, shells)])
                    command.copy_(torch.from_numpy(commands))
                    for side, view in enumerate(views):
                        view.set_position_targets(command[:, side])
                    solve_start = time.perf_counter()
                    provider.step()
                    provider.synchronize()
                    completed_steps += 1
                    solve_ms = 1000. * (time.perf_counter() - solve_start)
                    shells = arrays["ipc_positions"][:, offset:offset + size].cpu().numpy()
                    forces = arrays["ipc_affine_contact_wrenches"].cpu().numpy()
                    targets = arrays["ipc_affine_targets"].cpu().numpy()
                    proxies = arrays["ipc_affine_transforms"].cpu().numpy()
                    if not all(bool(torch.isfinite(arrays[key]).all()) for key in ("qpos", "qvel", "ipc_positions", "ipc_velocities")):
                        raise RuntimeError("non-finite batch state")
                    for environment, controller in enumerate(controllers):
                        error = trial.proxy_surface_error(targets[environment, hand_ids],
                                                         proxies[environment, hand_ids], radii)
                        controller.observe(forces[environment, tip_ids, :3], error)
                        if controller.failure:
                            raise RuntimeError(f"environment {environment}: {controller.failure}")
                        if tick >= args.warmup_steps:
                            maximum_proxy_error = max(maximum_proxy_error, error)
                            contact_steps[environment] += np.linalg.norm(forces[environment, hand_ids, :3], axis=1).max() > .01
                            pinch_steps[environment] += all(controller.last_pinches)
                    if tick >= args.warmup_steps:
                        provider_ms.append(solve_ms)
                        total_ms.append(1000. * (time.perf_counter() - start))
                    if (tick + 1) % 100 == 0:
                        print(json.dumps({"num_envs": count, "step": tick + 1, "sample_steps": len(total_ms)}), flush=True)
                report.update(benchmark_completed=True, provider_step_ms=timing(provider_ms),
                              end_to_end_step_ms=timing(total_ms),
                              environment_steps_per_second=count * len(total_ms) * 1000. / sum(total_ms),
                              provider_environment_steps_per_second=count * len(provider_ms) * 1000. / sum(provider_ms),
                              contact_step_fraction_per_environment=(contact_steps / args.steps).tolist(),
                              bilateral_pinch_fraction_per_environment=(pinch_steps / args.steps).tolist(),
                              maximum_proxy_error_meters=maximum_proxy_error,
                              all_environments_have_contact=bool(np.all(contact_steps > 0)),
                              controllers=[controller.diagnostics() for controller in controllers],
                              final_diagnostics=trial._jsonable(provider.diagnostics))
                report["benchmark_valid"] = bool(np.all(contact_steps == args.steps) and maximum_proxy_error <= .001)
        except Exception as error:
            report["error"] = str(error)
            report["benchmark_valid"] = False
            if provider_ms:
                report["partial_provider_step_ms"] = timing(provider_ms)
                report["partial_end_to_end_step_ms"] = timing(total_ms)
        finally:
            report["completed_physical_steps"] = completed_steps
            report["completed_sample_steps"] = len(provider_ms)
            if provider is not None:
                try:
                    report["final_diagnostics"] = trial._jsonable(provider.diagnostics)
                except Exception as error:
                    report["diagnostics_error"] = str(error)
            if sampler is not None:
                report["gpu_memory"] = sampler.result()
                report["memory_measurement_valid"] = bool(sampler.samples and sampler.peak and sampler.error is None)
                report["benchmark_valid"] = report.get("benchmark_valid", False) and report["memory_measurement_valid"]
            if provider is not None:
                provider.close()
            if context is not None:
                context.clear_scene()
    return trial._jsonable(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--scene", type=Path, default=trial.HERE / trial.SCENE_NAME)
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-wall-seconds", type=float, default=1200.)
    parser.add_argument("--module-path", default="")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.num_envs is not None:
        report = run_worker(args)
        okay = report.get("benchmark_valid", False)
    else:
        report = {"schema_version": 1, "runs": [], "scope": "Contact-stage scaling, not completed grasp throughput."}
        for count in args.counts:
            path = args.report.with_name(args.report.stem + f"-{count}.json")
            command = [sys.executable, str(Path(__file__)), "--num-envs", str(count),
                       "--warmup-steps", str(args.warmup_steps), "--steps", str(args.steps),
                       "--max-wall-seconds", str(args.max_wall_seconds), "--scene", str(args.scene),
                       "--report", str(path), "--module-path", args.module_path]
            with path.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            row = json.loads(path.read_text()) if path.exists() else {"num_envs": count, "error": "worker produced no report"}
            row["returncode"] = result.returncode
            report["runs"].append(row)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"num_envs": count, "returncode": result.returncode,
                              "error": row.get("error"), "report": str(path)}), flush=True)
        okay = all(row.get("benchmark_valid", False) and row["returncode"] == 0 for row in report["runs"])
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
