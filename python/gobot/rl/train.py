"""Train a built-in task or a Python factory with explicit physics selection."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from gobot.rl import tasks
from gobot.rl.training.workflow import (
    TrainingSpec, config_dict, get_value, launch, set_value, update_config,
    validate_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("task", nargs="?",
                        help="built-in task ID or module:factory returning TrainingSpec")
    parser.add_argument("--list", action="store_true", help="list built-in/imported tasks and supported backends")
    parser.add_argument("--task-module", action="append", default=[], help="import a Python module containing task registrations")
    parser.add_argument("--config", type=Path, help="JSON configuration for the selected training workflow")
    parser.add_argument("--backend", help="explicit physics backend; unsupported combinations are rejected")
    parser.add_argument("--device", help="training device; defaults to cpu or cuda:0 for the selected backend")
    parser.add_argument("--num-envs", type=int, help="environment count; per process except g1-supermimic, whose --batch-scope defaults to global")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", "--log-dir", dest="output", type=Path)
    parser.add_argument("--resume", type=Path, help="resume a compatible training checkpoint")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=JSON",
                        help="override a configuration field, e.g. --set experiment.learning_rate=0.0002")
    parser.add_argument("--dry-run", "--print-config", dest="dry_run", action="store_true",
                        help="validate and print resolved config without creating an environment or output directory")
    return parser


def resolve_spec(target: str) -> TrainingSpec:
    if ":" not in target:
        return tasks.load_training_spec(target)
    module_name, name = target.split(":", 1)
    if not module_name or not name:
        raise ValueError("Python training factory must be written as module:factory")
    factory = getattr(importlib.import_module(module_name), name)
    spec = factory()
    if not isinstance(spec, TrainingSpec):
        raise TypeError(f"{target} must return gobot.rl.training.TrainingSpec")
    return spec


def resolve_config(spec: TrainingSpec, args: argparse.Namespace):
    cfg = spec.config_factory()
    loaded = {}
    if args.config is not None:
        loaded = json.loads(args.config.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("training configuration must be a JSON object")
        cfg = update_config(cfg, loaded)
    paths = spec.paths()
    for key in ("backend", "device", "num_envs", "seed", "output", "resume"):
        value = getattr(args, key)
        if value is not None:
            if key in paths:
                cfg = set_value(cfg, paths[key], value)
            elif not (key == "output" and spec.configure is not None):
                raise ValueError(f"this workflow does not support --{key.replace('_', '-')}")
    if spec.configure is not None:
        cfg = spec.configure(cfg, args)
    overridden = set()
    for override in args.overrides:
        name, separator, raw = override.partition("=")
        if not separator:
            raise ValueError(f"expected PATH=JSON for --set, got {override!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"--set {name}: use a JSON value (quote strings), got {raw!r}") from error
        cfg = set_value(cfg, name, value)
        overridden.add(name)
    if "device" in paths:
        backend_override = args.backend is not None or paths["backend"] in overridden
        explicit_device = args.device is not None or paths["device"] in overridden
        config_backend_only = paths["backend"] in loaded and paths["device"] not in loaded
        if get_value(cfg, paths["device"]) is None or (not explicit_device and (backend_override or config_backend_only)):
            cfg = set_value(cfg, paths["device"], "cuda:0" if get_value(cfg, paths["backend"]) == "mujoco-warp" else "cpu")
    validate_config(spec, cfg)
    return cfg


def main(argv: Sequence[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    # Parse the selector independently of workflow-specific flags and --help.
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--task-module", action="append", default=[])
    selector.add_argument("--list", action="store_true")
    selected, rest = selector.parse_known_args(argv)
    try:
        for module in selected.task_module:
            importlib.import_module(module)
        if selected.list:
            # Still reject misspelled flags rather than silently ignoring them.
            parser.parse_args(argv)
            for name in tasks.list_tasks():
                task = tasks.get_task(name)
                backends = ",".join(task.supported_backends)
                print(f"{name}\t{task.algorithm}\t{backends}\t{task.description}")
            return
        # Select the workflow before loading its algorithm-specific arguments.
        target = rest[0] if rest and not rest[0].startswith("-") else None
        if target is None:
            if "--help" in rest or "-h" in rest:
                parser.print_help()
                return
            parser.error("provide a task ID or module:factory; use --list to see built-in tasks")
        spec = resolve_spec(target)
        if spec.add_arguments is not None:
            spec.add_arguments(parser)
        args = parser.parse_args(argv)
        if args.task != target:
            parser.error("put the task ID or module:factory before workflow-specific options")
        cfg = resolve_config(spec, args)
    except (ValueError, TypeError, KeyError, AttributeError, ImportError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        result = {"task": args.task, "step_unit": spec.step_unit,
                  "supported_backends": spec.supported_backends, "config": config_dict(cfg)}
        print(json.dumps(result, indent=2, allow_nan=False))
        return result
    result = launch(spec, cfg)
    if result is not None and int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
