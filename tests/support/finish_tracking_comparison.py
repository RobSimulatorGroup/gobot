"""Wait for native mjlab completion, then train/evaluate the matched Gobot run.

A single foreground pipeline intended for a tmux session. No cron/scheduler.
Fails closed if the baseline fails or if any comparison command fails.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--mjlab-python", type=Path, default=Path("mjlab/.venv/bin/python"))
    args = p.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    status = args.baseline / "status.json"
    started = time.monotonic()
    while True:
        if status.exists():
            baseline = json.loads(status.read_text())
            if baseline["state"] == "trained":
                break
            if baseline["state"] != "training":
                raise RuntimeError(f"baseline not running successfully: {baseline}")
            write(
                args.report,
                dict(
                    state="waiting_for_mjlab",
                    baseline_iteration=baseline.get("iteration"),
                    wait_seconds=time.monotonic() - started,
                ),
            )
        if time.monotonic() - started > 24 * 3600:
            raise TimeoutError("baseline did not finish within 24 hours")
        time.sleep(30)
    cfg = json.loads((args.baseline / "comparison.json").read_text())
    root = Path(__file__).resolve().parents[2]

    def execute(command, label, environment=None):
        write(args.report, dict(state=label, command=command))
        with args.report.with_name(label + ".log").open("x") as log:
            try:
                subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except BaseException as error:
                write(
                    args.report,
                    dict(
                        state="failed", phase=label, error=str(error), command=command
                    ),
                )
                raise

    all_gpus = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(map(str, range(cfg["world_size"]))),
        OMP_NUM_THREADS="4",
    )
    execute(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={cfg['world_size']}",
            "tests/support/train_gobot_tracking_comparison.py",
            "--baseline",
            str(args.baseline),
            "--project",
            str(args.project),
            "--output",
            str(args.output),
        ],
        "gobot_training",
        all_gpus,
    )
    one_gpu = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4")
    native_eval = args.report.with_name("mjlab-evaluation.json")
    gobot_eval = args.report.with_name("gobot-evaluation.json")
    motion = args.project / "motion.npz"
    execute(
        [
            str(args.mjlab_python),
            "tests/support/evaluate_mjlab_tracking.py",
            "--checkpoint",
            str(args.baseline / "model.pt"),
            "--motion",
            str(motion),
            "--output",
            str(native_eval),
        ],
        "mjlab_evaluation",
        one_gpu,
    )
    execute(
        [
            sys.executable,
            "-m",
            "gobot.rl.tracking.evaluate",
            str(args.output / "model.pt"),
            "--backend",
            "mujoco-warp",
            "--device",
            "cuda:0",
            "--output",
            str(gobot_eval),
        ],
        "gobot_evaluation",
        one_gpu,
    )
    reports = {}
    for name, directory, evaluation in [
        ("mjlab", args.baseline, native_eval),
        ("gobot", args.output, gobot_eval),
    ]:
        data = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        timing = data[min(10, len(data)) :] or data
        import statistics

        reports[name] = dict(
            training=json.loads((directory / "status.json").read_text()),
            evaluation=json.loads(evaluation.read_text()),
            median_collection_s=statistics.median(r["collection_s"] for r in timing),
            median_learning_s=statistics.median(r["learning_s"] for r in timing),
        )
    write(
        args.report,
        dict(
            state="completed",
            reference_revision=cfg["reference_revision"],
            motion_sha256=cfg["motion_sha256"],
            settings={
                k: cfg[k]
                for k in (
                    "world_size",
                    "num_envs_per_rank",
                    "iterations",
                    "rollout_steps",
                    "seed",
                    "allow_tf32",
                    "nconmax",
                    "njmax",
                )
            },
            results=reports,
            limitations=[
                "single training seed",
                "native first-contact slot is nondeterministic",
                "scene compiler FP32 tolerances",
            ],
        ),
    )


if __name__ == "__main__":
    main()
