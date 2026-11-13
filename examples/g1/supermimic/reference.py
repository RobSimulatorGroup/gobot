"""Verify the official checkout and its companion before generating expectations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types

LOCK = Path(__file__).with_name("reference.lock.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkout(path: Path, commit: str, source_paths=None) -> dict:
    path = path.resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    actual = git("rev-parse", "HEAD")
    if actual != commit:
        raise ValueError(f"{path}: expected {commit}, found {actual}")
    if git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
    ):
        raise ValueError(f"{path}: reference sources have local changes")
    # Ignore skip-worktree/assume-unchanged flags: compare numerical bytes with Git.
    source_paths = source_paths or ["src"]
    files = git(
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        *source_paths,
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
    ).splitlines()
    hashes = {}
    for name in files:
        if not (
            name.endswith(".py") or name in ("pyproject.toml", "uv.lock", "LICENSE")
        ):
            continue
        expected = subprocess.check_output(
            ["git", "-C", str(path), "show", f"{commit}:{name}"]
        )
        actual_bytes = (path / name).read_bytes()
        if actual_bytes != expected:
            raise ValueError(f"{path / name}: differs from pinned Git object")
        hashes[name] = hashlib.sha256(actual_bytes).hexdigest()
    return {"commit": commit, "files": hashes}


def audit(reference: Path, mjlab: Path) -> dict:
    lock = json.loads(LOCK.read_text())
    return {
        "protocol": lock["protocol"],
        "sources": {
            "supermimic": verify_checkout(reference, lock["supermimic"]["commit"]),
            "mjlab_research": verify_checkout(
                mjlab,
                lock["mjlab_research"]["commit"],
                lock["mjlab_research"]["numerical_source_paths"],
            ),
        },
        "runtime": {
            "python": sys.version,
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "rsl-rl-lib",
                    "tensordict",
                    "mujoco",
                    "mujoco-warp",
                    "warp-lang",
                )
            },
        },
        "upstream_locked_runtime": lock["upstream_locked_runtime"],
        "scope": "same-stack tensor comparison; original-stack physics and training are pending",
    }


def numerical_packages(reference: Path, mjlab: Path) -> None:
    """Skip registration initializers, leaving all numerical source functions intact."""
    if any(name == "gobot" or name.startswith("gobot.") for name in sys.modules):
        raise RuntimeError("reference expectations must run without importing Gobot")
    roots = {
        "supermimic": reference.resolve() / "src/supermimic",
        "mjlab": mjlab.resolve() / "src/mjlab",
    }
    for name in (
        "supermimic",
        "supermimic.tasks",
        "mjlab",
        "mjlab.utils",
        "mjlab.utils.lab_api",
    ):
        top, *rest = name.split(".")
        package = types.ModuleType(name)
        package.__path__ = [str(roots[top].joinpath(*rest))]
        if name == "mjlab":
            package.MJLAB_SRC_PATH = roots[top]
        sys.modules[name] = package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mjlab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.reference, args.mjlab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified official sources; report: {args.output}")
