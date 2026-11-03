"""Bounded Uni-SuperMimic world-model/BPTT experiment with full-clip evaluation.

Run using the isolated reference interpreter. This module intentionally never imports
Gobot: its native MuJoCo build is a different version from this reference experiment.
"""
from __future__ import annotations

import argparse
import gc
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "examples/locomotion/logs/supermimic_reference/Uni-SuperMimic"
DEFAULT_LEDGER = ROOT / "examples/g1/logs/gpu_budget.json"
WINDOW = ROOT / "examples/locomotion/logs/natural_v2/budget_window.json"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hash(reference: Path) -> dict[str, str]:
    return {str(path.relative_to(reference)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((reference / "src").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml", ".xml")}


class ControlledStop(Exception):
    pass


class RunStatus:
    def __init__(self, run_dir: Path, budget):
        self.run_dir = run_dir
        self.budget = budget
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.cancelled = False
        self.state = {"status": "initializing", "pid": os.getpid(), "host": socket.gethostname(),
                      "started_unix": time.time(), "algorithm": "SuperMimic world-model/BPTT"}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            self.state.update(heartbeat_unix=time.time(), elapsed_s=self.budget.elapsed,
                              remaining_s=self.budget.remaining)
            self.budget.update(status=self.state["status"], iteration=self.state.get("iteration", 0))
            write_json(self.run_dir / "status.json", self.state)

    def _heartbeat(self):
        while not self.done.wait(15):
            self.update()
            if self.budget.remaining <= 0:
                self.update(status="budget_exhausted", stop_reason="hard_deadline", checkpoint_note="use most recent saved checkpoint")
                self.budget.close(status="budget_exhausted")
                os._exit(124)

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.update()
        self.budget.close(status=self.state["status"], iteration=self.state.get("iteration", 0))


def budget_class():
    spec = importlib.util.spec_from_file_location(
        "reference_training_budget", ROOT / "python/gobot/rl/locomotion/budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainingBudget


def evaluation_summary(result: dict, motion) -> dict:
    expected = {name: int(length) - 1 for name, length in zip(motion.names, motion.clip_len.tolist())}
    for name, row in result["per_clip"].items():
        row["expected_steps"] = expected[name]
        row["coverage"] = min(1.0, row["steps"] / expected[name])
        row["completed_full_clip"] = not row["failed"] and row["steps"] >= expected[name]
    result["n_full_clips"] = sum(row["completed_full_clip"] for row in result["per_clip"].values())
    result["frame_coverage"] = sum(min(row["steps"], expected[name])
                                   for name, row in result["per_clip"].items()) / sum(expected.values())
    result["gait_groups"] = {}
    for group in ("walk", "run", "sprint"):
        rows = [row for name, row in result["per_clip"].items() if Path(name).name.startswith(group)]
        if rows:
            result["gait_groups"][group] = {
                "clips": len(rows), "norm_return": sum(r["norm_return"] for r in rows) / len(rows),
                "failed": sum(r["failed"] for r in rows),
                "full_clips": sum(r["completed_full_clip"] for r in rows),
            }
    result["admitted_velocity_terrain_policy"] = False
    result["protocol"] = "training-library tracking, first-frame resets, full clips, no observation/action noise"
    return result


def run(args) -> None:
    reference, run_dir = args.reference.resolve(), args.run_dir.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None and args.resume.resolve().parent != run_dir:
        raise ValueError("Upstream resume continues in the checkpoint directory; --run-dir must match")
    if (run_dir / "manifest.json").exists() and args.resume is None:
        raise ValueError("Run already exists; use a new directory or explicit --resume")
    window = json.loads(WINDOW.read_text())
    total_hours = min(24.0, window["baseline_seconds"] / 3600 + window["additional_gpu_hours"])
    budget = budget_class()(args.ledger.resolve(), total_hours=total_hours, run_hours=args.hours,
                            metadata={"robot": "g1", "stage": "supermimic_reference",
                                      "log_dir": str(run_dir), "seed": args.seed})
    status = RunStatus(run_dir, budget)
    holder = {}
    eval_env = None
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: setattr(status, "cancelled", True))
    try:
        os.chdir(reference)
        faulthandler.enable()
        faulthandler.dump_traceback_later(180, repeat=True)
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        import unisupermimic
        from unisupermimic.scripts import train_supermimic as upstream
        from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
        from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import evaluate, make_eval_env

        torch.set_num_threads(args.cpu_threads)
        config_dir = Path(unisupermimic.__file__).parent / "conf/supermimic"
        overrides = [
            f"algo.num_envs={args.num_envs}", f"algo.wm_batch={args.batch_size}",
            f"algo.pi_batch={args.batch_size}", f"algo.seed={args.seed}",
            f"algo.iterations={args.iterations}", f"algo.wm_only_iters={args.wm_only_iters}",
            f"algo.compile={str(args.compile).lower()}", "algo.ckpt_every=250", "algo.log_every=10",
            f"training.log_dir={run_dir}",
        ]
        if args.resume is not None:
            overrides += ["algo.resume=true", f"algo.load_run={args.resume.resolve()}"]
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=overrides)
        if len(cfg.env.commands.motion.motion_files) != 40:
            raise ValueError("Reference run requires all 40 motion clips")
        semantic_cfg = OmegaConf.to_container(cfg, resolve=True)
        semantic_cfg["training"].pop("log_dir", None)
        for key in ("iterations", "load_run", "checkpoint", "resume", "ckpt_every", "log_every"):
            semantic_cfg["algo"].pop(key, None)
        assets = json.loads((reference / "reference-assets.json").read_text())
        asset_hashes = {r["remote"]: r["sha256"] for r in assets["files"]}
        for record in assets["files"]:
            local = reference / record["local"]
            if hashlib.sha256(local.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Pinned asset changed: {local}")
        # UniLab 1.5.1 identifies robots by any robots/g1 path segment, even
        # outside its package, and otherwise downloads unrelated package assets.
        # Skip that fallback only for this already verified reference asset tree.
        from unilab.assets import hub
        from unilab.base import backend_factory
        original_resolver = hub.ensure_robot_assets_for_paths
        asset_root = reference / "src/unisupermimic/assets"

        def resolve_pinned_assets(paths):
            other = []
            for path in paths:
                if path and Path(path).resolve().is_relative_to(asset_root):
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                else:
                    other.append(path)
            original_resolver(other)

        hub.ensure_robot_assets_for_paths = resolve_pinned_assets
        backend_factory.ensure_robot_assets_for_paths = resolve_pinned_assets
        contract = {"config": semantic_cfg, "source_hashes": source_hash(reference), "assets": asset_hashes}
        signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if args.resume is not None:
            previous = json.loads((args.resume.resolve().parent / "manifest.json").read_text())
            if previous["signature"] != signature:
                raise ValueError("Resume configuration, source, or assets changed")
        manifest = {
            "algorithm": "SuperMimic world-model/BPTT (Uni-SuperMimic)",
            "source_revision": "faf7818e8abac3ec6abb8f00e6e904d3577f800b", "signature": signature,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "mujoco", "mujoco-warp", "warp-lang", "unilab", "unisim-core")},
            "gpu": torch.cuda.get_device_name(0), "overrides": overrides,
            "reference_deviations": [
                "local environment/batch counts recorded in overrides; full networks and horizon=32 retained",
                "Torch 2.11.0+cu128 shared with local environment instead of reference lock Torch 2.14.1",
                "external bounded runner adds checkpoint/heartbeat/evaluation, no change to learning equations",
                "skip UniLab unrelated package asset download for checksum-verified external reference asset paths",
            ],
            "source_hashes": contract["source_hashes"], "assets": assets,
            "resume": str(args.resume) if args.resume else None,
            "resume_note": "upstream restores networks and optimizers; replay buffer/environment are refilled",
            "budget": {"run_hours": args.hours, "total_cap_hours": total_hours, "window": window},
        }
        sessions = run_dir / "sessions"
        sessions.mkdir(exist_ok=True)
        write_json(sessions / f"{int(time.time())}_{os.getpid()}.json", manifest)
        write_json(run_dir / "manifest.json", manifest)
        OmegaConf.save(cfg, run_dir / "requested_config.yaml")
        eval_reserve = args.eval_minutes * 60 if args.evaluate else 0

        class BoundedLearner(SuperMimicLearner):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                holder["learner"] = self
                self._last_full_save = time.monotonic()
                self._max_iteration_seconds = 0.0
                self._last_collect = None
                self.stop_reason = None

            def collect(self, n_steps):
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(180, repeat=True)
                now = time.monotonic()
                if self._last_collect is not None:
                    self._max_iteration_seconds = max(self._max_iteration_seconds, now - self._last_collect)
                self._last_collect = now
                status.update(status="training", iteration=self.iteration, sim_steps=self.sim_steps,
                              phase="world_model" if self.iteration < self.cfg.wm_only_iters else "joint",
                              peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                if status.cancelled or budget.remaining <= eval_reserve + max(30, self._max_iteration_seconds * 1.2):
                    self.stop_reason = "signal" if status.cancelled else "time_budget"
                    raise ControlledStop(self.stop_reason)
                if self.iteration and now - self._last_full_save >= 600:
                    self.save()
                return super().collect(n_steps)

            def wm_update(self):
                values = super().wm_update()
                if not all(bool(torch.isfinite(value).all()) for value in values):
                    raise FloatingPointError("World model update produced nonfinite metrics")
                return values

            def policy_update(self):
                values = super().policy_update()
                if not all(bool(torch.isfinite(value).all()) for value in values):
                    raise FloatingPointError("Policy update produced nonfinite metrics")
                return values

            def save(self, path=None, include_optimizers=True):
                result = super().save(path, include_optimizers)
                if include_optimizers:
                    self._last_full_save = time.monotonic()
                status.update(iteration=self.iteration, last_checkpoint=str(result),
                              full_checkpoint=str(self.log_dir / "last.pt"),
                              peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                return result

            def train(self, resume=None):
                try:
                    return super().train(resume)
                except ControlledStop:
                    self.save()
                    return {"completed": False, "iteration": self.iteration, "stop_reason": self.stop_reason}
                finally:
                    status.update(iteration=self.iteration, sim_steps=self.sim_steps)

        upstream.SuperMimicLearner = BoundedLearner
        status.update(status="building_environment")
        upstream.main(cfg)
        learner = holder.pop("learner")
        final_iteration = learner.iteration
        reason = learner.stop_reason or "iteration_limit"
        del learner
        gc.collect()
        torch.cuda.empty_cache()
        status.update(status="trained", iteration=final_iteration, stop_reason=reason)
        if args.evaluate and not status.cancelled and budget.remaining > 30:
            status.update(status="evaluating")
            eval_env = make_eval_env(run_dir, backend="mjwarp")
            faulthandler.cancel_dump_traceback_later()
            original_step = eval_env.step

            def bounded_eval_step(action):
                if status.cancelled or budget.remaining < 10:
                    raise ControlledStop("evaluation_time_budget")
                return original_step(action)

            eval_env.step = bounded_eval_step
            checkpoint = torch.load(run_dir / "last.pt", map_location=eval_env.device, weights_only=True)
            result = evaluation_summary(evaluate(eval_env, checkpoint), eval_env.supermimic_mdp.motion)
            result.update(iteration=final_iteration, evaluated_unix=time.time())
            write_json(run_dir / "evaluation.json", result)
            status.update(evaluation={key: result[key] for key in
                                     ("norm_return", "n_failed", "n_full_clips", "frame_coverage", "gait_groups")})
        status.update(status="stopped" if status.cancelled else "finished", stop_reason=reason)
    except ControlledStop as exc:
        status.update(status="stopped", stop_reason=str(exc))
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if eval_env is not None:
                eval_env.close()
                eval_env.backend.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
            status.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--iterations", type=int, default=12000)
    p.add_argument("--wm-only-iters", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=6)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-minutes", type=float, default=10)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if not math.isfinite(args.hours) or not 0 < args.hours <= 8:
        p.error("--hours must be positive and at most the authorized 8 hours")
    if min(args.num_envs, args.batch_size, args.iterations, args.cpu_threads) <= 0:
        p.error("training sizes must be positive")
    if args.eval_minutes < 0 or (args.evaluate and args.eval_minutes * 60 + 30 >= args.hours * 3600):
        p.error("evaluation reserve must leave training time")
    run(args)


if __name__ == "__main__":
    main()
