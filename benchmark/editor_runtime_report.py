"""Summarize editor --benchmark-output samples without mixing cold and warm frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def memory_mib(value: str | None) -> float | None:
    if not value:
        return None
    fields = value.split()
    if len(fields) != 2 or fields[1] not in {"MiB", "GiB"}:
        return None
    try:
        number = float(fields[0])
    except ValueError:
        return None
    return number * (1024 if fields[1] == "GiB" else 1) if math.isfinite(number) and number >= 0 else None


def summarize_gpu(metadata: dict) -> dict:
    devices: dict[str, dict] = {}
    for sample in metadata.get("gpu_samples", []):
        for gpu in sample.get("gpus", []):
            identity = gpu.get("uuid") or gpu["device"]
            record = devices.setdefault(identity, {"device": gpu["device"], "driver": gpu.get("driver"),
                                                   "process_mib": [], "device_mib": []})
            values = [memory_mib(process.get("memory")) for process in gpu.get("processes", [])]
            if values and all(value is not None for value in values):
                record["process_mib"].append(sum(values))
            used = memory_mib(gpu.get("device_used"))
            if used is not None:
                record["device_mib"].append(used)
    return {
        identity: {
            "device": record["device"], "driver": record["driver"],
            "process_samples": len(record["process_mib"]),
            "process_peak_mib": max(record["process_mib"], default=None),
            "device_peak_mib": max(record["device_mib"], default=None),
            "telemetry": "available" if record["process_mib"] else "unavailable",
        }
        for identity, record in devices.items()
    }


def summarize(report: dict, metadata: dict | None = None) -> dict:
    if metadata is not None and (metadata.get("timed_out") or metadata.get("exit_code", -1) != 0):
        raise ValueError("Editor did not exit successfully; capture cannot be used as a baseline")
    if report["schema"] != 1:
        raise ValueError("Unsupported editor benchmark schema")
    samples = report["samples"][report["warmup_frames"] :]
    if not samples or len(samples) != report["requested_frames"]:
        raise ValueError("Incomplete capture: scene loading, Play, or editor exited early")
    if report["faulted"]:
        raise ValueError(f"Simulation faulted: {report['error']}")
    if report["play"] and not report.get("world_ready", True):
        raise ValueError("Physics world did not finish building")
    columns = report["columns"]
    metrics = {}
    for column in ("frame_ms", "physics_dispatch_ms", "process_ms", "draw_ms"):
        values = [row[columns.index(column)] for row in samples]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid {column} samples")
        metrics[column] = {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    elapsed = sum(row[columns.index("frame_ms")] for row in samples[1:]) / 1000
    time_index = columns.index("simulation_time")
    times = [row[time_index] for row in samples]
    simulated = times[-1] - times[0]
    if any(not math.isfinite(value) for value in times) or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Simulation time reset during measured capture")
    physics = {}
    physics_columns = report.get("physics_columns", [])
    physics_samples = [row for row in report.get("physics_samples", [])
                       if report["warmup_frames"] <= row[physics_columns.index("render_frame")] < len(report["samples"])]
    if physics_samples:
        counts = [row[physics_columns.index("dispatch_ticks")] for row in physics_samples]
        physics = {"samples": len(physics_samples), "completed_ticks": sum(counts),
                   "coverage": "all_ticks" if all(count == 1 for count in counts) else "last_tick_per_dispatch"}
        for column in ("step_ms", "solve_ms", "newton_iterations", "line_search_iterations"):
            values = [row[physics_columns.index(column)] for row in physics_samples]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Invalid physics {column} samples")
            physics[column] = ({"p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
                                "p99": percentile(values, 0.99)} if any(values) else None)
    resources = {}
    for column in ("mesh_cache_entries", "texture_cache_entries", "render_cache_bytes"):
        if column in columns:
            values = [row[columns.index(column)] for row in samples]
            resources[column] = {"peak": max(values), "last": values[-1]}
    for column in ("uploaded_bytes_total", "geometry_uploads_total", "index_uploads_total",
                   "image_uploads_total", "upload_ms_total"):
        if column in columns:
            values = [row[columns.index(column)] for row in samples]
            reset = any(current < previous for previous, current in zip(values, values[1:]))
            resources[column] = {"delta": None if reset else values[-1] - values[0], "counter_reset": reset}
    return {
        "commit": report["commit"] or (metadata or {}).get("commit", ""),
        "project": report["project"],
        "play": report["play"],
        "frames": len(samples),
        "scene_ready_ms": report["scene_ready_ms"],
        "first_play_ms": report["first_play_ms"],
        "world_ready_ms": report.get("world_ready_ms"),
        "fixed_dt": report["fixed_dt"],
        "physics_settings": report.get("physics_settings"),
        "scheduling": report.get("scheduling", "synchronous"),
        "window_size": report.get("window_size"),
        "renderer": report.get("renderer"),
        "renderer_mode": report.get("renderer_mode"),
        "metrics": metrics,
        "physics": physics,
        "resources": resources,
        "gpu": summarize_gpu(metadata or {}),
        "real_time_factor": simulated / elapsed if elapsed > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    metadata_path = args.capture.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    print(json.dumps(summarize(json.loads(args.capture.read_text()), metadata), indent=2))


if __name__ == "__main__":
    main()
