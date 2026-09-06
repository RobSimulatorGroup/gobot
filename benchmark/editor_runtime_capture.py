"""Run a bounded editor capture in an isolated project copy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from editor_runtime_report import summarize


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, timeout=10).strip()


def gpu_sample(pid: int) -> list[dict]:
    root = ET.fromstring(command_output(["nvidia-smi", "-q", "-x"]))
    samples = []
    for gpu in root.findall("gpu"):
        processes = [
            {"type": process.findtext("type"), "memory": process.findtext("used_memory")}
            for process in gpu.findall("processes/process_info")
            if process.findtext("pid") == str(pid)
        ]
        samples.append({
            "device": gpu.findtext("product_name"),
            "uuid": gpu.findtext("uuid"),
            "driver": root.findtext("driver_version"),
            "total": gpu.findtext("fb_memory_usage/total"),
            "device_used": gpu.findtext("fb_memory_usage/used"),
            "processes": processes,
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--editor", type=Path, default=Path("build/python/gobot/gobot_editor"))
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--scheduling", choices=("worker", "synchronous"), default="worker")
    parser.add_argument("--render-fps", type=int, default=60)
    args = parser.parse_args()
    project = args.project.resolve()
    editor = args.editor.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    capture = output / "frames.json"
    if capture.exists():
        parser.error("Output already contains a capture; choose a new directory")
    repo = Path(__file__).resolve().parents[1]
    metadata = {
        "source_project": str(project),
        "commit": command_output(["git", "rev-parse", "HEAD"], repo),
        "worktree": command_output(["git", "status", "--short"], repo),
        "submodules": command_output(["git", "submodule", "status"], repo),
        "gpu_samples": [],
        "requested_window": [1920, 1080],
    }
    cache = editor.parents[2] / "CMakeCache.txt"
    if cache.exists():
        metadata["cmake"] = [line for line in cache.read_text().splitlines()
                             if line.startswith(("GOB_", "CMAKE_BUILD_TYPE:"))]
    environment = os.environ.copy()
    package_root = str(editor.parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        package_root, str(repo), str(project), *site.getsitepackages(),
        environment.get("PYTHONPATH", "")]))
    with tempfile.TemporaryDirectory(prefix="gobot-editor-benchmark-") as temporary:
        copied_project = Path(temporary) / project.name
        shutil.copytree(project, copied_project, ignore=shutil.ignore_patterns("__pycache__"))
        command = [str(editor), "--path", str(copied_project),
                   "--benchmark-output", str(capture),
                   "--benchmark-frames", str(args.frames),
                   "--benchmark-warmup", str(args.warmup),
                   "--physics-scheduling", args.scheduling,
                   "--render-fps", str(args.render_fps)]
        if args.play:
            command.append("--benchmark-play")
        with (output / "editor.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic() - started > args.timeout:
                        metadata["timed_out"] = True
                        raise TimeoutError("Editor capture exceeded timeout")
                    try:
                        metadata["gpu_samples"].append({
                            "elapsed_s": time.monotonic() - started,
                            "gpus": gpu_sample(process.pid),
                        })
                    except (OSError, subprocess.SubprocessError, ET.ParseError) as error:
                        metadata["gpu_error"] = str(error)
                    time.sleep(0.5)
                metadata["exit_code"] = process.returncode
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                metadata["exit_code"] = process.returncode
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if process.returncode != 0:
        raise RuntimeError(f"Editor exited with {process.returncode}; see {output / 'editor.log'}")
    summary = summarize(json.loads(capture.read_text()), metadata)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
