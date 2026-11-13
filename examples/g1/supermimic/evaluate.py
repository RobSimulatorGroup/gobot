"""Evaluate one or more checkpoints on identical GPU shards and compare results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gobot.rl.supermimic.evaluate import hash_value, merge_reports, summarize
from gobot.rl.supermimic.logging import write_json
from gobot.rl.supermimic.motion import digest


def compare(results, keep=None):
    if not results:
        raise ValueError("no checkpoint results")
    baseline = results[0]
    names = [row["name"] for row in baseline["clips"]]
    if len(names) != len(set(names)) or not names:
        raise ValueError("invalid baseline membership")
    if keep is not None and (not keep or not keep <= set(names)):
        raise ValueError("invalid fixed filtered subset")
    before = {row["name"] for row in baseline["clips"] if row["failed"]}
    comparisons = []
    for result in results:
        if not result["complete"] or result["common"] != baseline["common"]:
            raise ValueError("incomplete evaluation or different evaluation conditions")
        if result["common_sha256"] != hash_value(result["common"]):
            raise ValueError("evaluation fingerprint differs")
        if [row["name"] for row in result["clips"]] != names:
            raise ValueError("clip membership or order differs")
        if any(
            a["frames"] != b["frames"] or a["expected_steps"] != b["expected_steps"]
            for a, b in zip(baseline["clips"], result["clips"], strict=True)
        ):
            raise ValueError("clip lengths or clocks differ")
        if (
            any(row["capped"] for row in result["clips"])
            or summarize(result["clips"]) != result["summary"]
        ):
            raise ValueError("capped evaluation or inconsistent summary")
        failures = {row["name"] for row in result["clips"] if row["failed"]}
        entry = {
            "checkpoint_sha256": result["checkpoint_sha256"],
            "policy": result["policy_adapter"],
            "full": result["summary"],
            "fixed_baseline_failures": sorted(before - failures),
            "new_failures": sorted(failures - before),
            "remaining_baseline_failures": sorted(before & failures),
        }
        if keep is not None:
            entry["filtered"] = summarize(
                [row for row in result["clips"] if row["name"] in keep]
            )
        comparisons.append(entry)
    return {
        "complete": True,
        "common_sha256": baseline["common_sha256"],
        "checkpoints": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    for key in ("project", "library", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--filter-manifest", type=Path, help="optional CSV with name,keep columns"
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--num-envs", type=int, default=512)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPU indices must be unique and nonnegative")
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    names = json.loads((args.library / "motion_names.json").read_text())
    keep = None
    if args.filter_manifest:
        with args.filter_manifest.open() as stream:
            keep = {row["name"] for row in csv.DictReader(stream) if row["keep"] == "1"}
        if not keep or not keep <= set(names):
            raise ValueError("filter manifest is not a subset of this library")
    plan = {
        "checkpoints": [
            {"path": str(path.resolve()), "sha256": digest(path)}
            for path in args.checkpoints
        ],
        "manifest_sha256": digest(args.library / "manifest.json"),
        "gpus": args.gpus,
        "num_envs": args.num_envs,
        "selection_count": len(names),
        "selection_sha256": hash_value(names),
    }
    if args.filter_manifest:
        plan["filter_manifest_sha256"] = digest(args.filter_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    started = time.monotonic()
    results = []
    for index, checkpoint in enumerate(args.checkpoints):
        folder = args.output / f"checkpoint{index}"
        folder.mkdir()
        children = []
        workers = []
        try:
            for shard, gpu in enumerate(args.gpus):
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "gobot.rl.supermimic.evaluate",
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--project",
                    str(args.project.resolve()),
                    "--library",
                    str(args.library.resolve()),
                    "--output",
                    str((folder / f"shard{shard}.json").resolve()),
                    "--device",
                    "cuda:0",
                    "--num-envs",
                    str(args.num_envs),
                    "--shard-count",
                    str(len(args.gpus)),
                    "--shard-index",
                    str(shard),
                ]
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="1",
                )
                for key in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                ):
                    environment.pop(key, None)
                with (folder / f"worker{shard}.log").open("w") as log:
                    child = subprocess.Popen(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                children.append(child)
                workers.append({"gpu": gpu, "pid": child.pid, "command": command})
            write_json(folder / "workers.json", workers)
            last_log = 0.0
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError(f"checkpoint {index}: evaluation worker failed")
                now = time.monotonic()
                if now - last_log >= 30:
                    progress = [
                        json.loads(path.read_text())
                        for path in folder.glob("*.progress.json")
                    ]
                    row = {
                        "state": "evaluating",
                        "checkpoint": index,
                        "completed_clips": sum(
                            p.get("completed_clips", 0) for p in progress
                        ),
                        "total_clips": len(names),
                        "elapsed_seconds": now - started,
                    }
                    write_json(args.output / "status.json", row)
                    print(json.dumps(row), flush=True)
                    last_log = now
                time.sleep(2)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("evaluation worker failed at completion")
            result = merge_reports(
                [folder / f"shard{i}.json" for i in range(len(args.gpus))],
                names,
                folder / "full.json",
            )
            if result["checkpoint_sha256"] != plan["checkpoints"][index]["sha256"]:
                raise ValueError("checkpoint changed after evaluation planning")
            write_json(
                folder / "failures.json",
                [row for row in result["clips"] if row["failed"]],
            )
            results.append(result)
            write_json(args.output / "comparison.json", compare(results, keep))
        except BaseException as error:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            write_json(
                args.output / "status.json",
                {"state": "failed", "error": str(error), "checkpoint": index},
            )
            raise
    write_json(
        args.output / "status.json",
        {"state": "completed", "elapsed_seconds": time.monotonic() - started},
    )
    print(json.dumps(compare(results, keep), indent=2))


if __name__ == "__main__":
    main()
