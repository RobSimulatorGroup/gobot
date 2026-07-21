#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


NATIVE_PAYLOAD = (
    "libgobot.so",
    "libgobot_luisa_renderer.so",
    "luisa/libluisa-backend-cuda.so",
    "luisa/libluisa-runtime.so",
    "luisa/luisa_nvrtc",
)
GO1_POLICY_PAYLOAD = (
    "examples/go1/policies/go1_velocity.onnx",
)
OTHER_EXAMPLE_PAYLOAD = (
    "examples/cartpole/policies/cartpole.onnx",
    "examples/cartpole/policies/cartpole.pt",
)
ELF_PAYLOAD = (
    "libgobot_luisa_renderer.so",
    "luisa/libluisa-backend-cuda.so",
    "luisa/luisa_nvrtc",
)
ALLOWED_MISSING_LIBRARIES = {"libcuda.so.1"}


def missing_libraries(path: Path) -> set[str]:
    result = subprocess.run(
        ["ldd", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    missing: set[str] = set()
    for line in result.stdout.splitlines():
        if "=> not found" in line:
            missing.add(line.split("=>", 1)[0].strip())
    return missing


def main() -> int:
    import gobot
    from gobot_cli import editor as editor_launcher

    package_root = Path(gobot.__file__).resolve().parent
    missing_payload = [
        name
        for name in (*NATIVE_PAYLOAD, *GO1_POLICY_PAYLOAD, *OTHER_EXAMPLE_PAYLOAD)
        if not (package_root / name).is_file()
    ]
    if missing_payload:
        raise RuntimeError("installed wheel is missing: " + ", ".join(missing_payload))
    policy_directory = package_root / "examples/go1/policies"
    installed_policies = sorted(
        path.relative_to(package_root).as_posix()
        for path in policy_directory.iterdir()
        if path.is_file()
    )
    if installed_policies != list(GO1_POLICY_PAYLOAD):
        raise RuntimeError(
            "installed wheel must contain exactly one Go1 policy: "
            + ", ".join(installed_policies)
        )

    if gobot.render.RenderBuffer is not gobot._core.RenderBuffer:
        raise RuntimeError("gobot.render.RenderBuffer does not expose the native wheel type")
    if gobot.render.RenderFrame is not gobot._core.RenderFrame:
        raise RuntimeError("gobot.render.RenderFrame does not expose the native wheel type")
    if gobot.render.RenderProduct is not gobot._core.RenderProduct:
        raise RuntimeError("gobot.render.RenderProduct does not expose the native wheel type")
    if "torch" in sys.modules:
        raise RuntimeError("importing gobot must not eagerly import or initialize Torch")

    if importlib.util.find_spec("gobot.rl") is None:
        raise RuntimeError("installed wheel does not expose gobot.rl")

    resolved_package_root = editor_launcher._distribution_gobot_dir()
    if resolved_package_root is None or resolved_package_root.resolve() != package_root:
        raise RuntimeError("editor launcher cannot resolve the installed Gobot package")
    launcher_env = editor_launcher._with_editor_python_environment(sys.executable)
    python_paths = launcher_env.get("PYTHONPATH", "").split(os.pathsep)
    if not python_paths or Path(python_paths[0]).resolve() != package_root.parent:
        raise RuntimeError("editor launcher does not prioritize the installed Gobot environment")
    launcher_env["PYTHONNOUSERSITE"] = "1"
    find_rl = """
import importlib.util
from pathlib import Path
import sys

import gobot

expected = Path(sys.argv[1]).resolve()
actual = Path(gobot.__file__).resolve().parent
assert actual == expected, (actual, expected)
assert importlib.util.find_spec("gobot.rl") is not None
"""
    with tempfile.TemporaryDirectory() as temporary_directory:
        subprocess.run(
            [sys.executable, "-S", "-c", find_rl, str(package_root)],
            check=True,
            cwd=temporary_directory,
            env=launcher_env,
        )

    unexpected_missing: dict[str, set[str]] = {}
    for relative_path in ELF_PAYLOAD:
        missing = missing_libraries(package_root / relative_path)
        unexpected = missing - ALLOWED_MISSING_LIBRARIES
        if unexpected:
            unexpected_missing[relative_path] = unexpected
    if unexpected_missing:
        details = "; ".join(
            f"{path}: {', '.join(sorted(libraries))}"
            for path, libraries in sorted(unexpected_missing.items())
        )
        raise RuntimeError("installed wheel has unresolved native dependencies: " + details)

    print(f"installed Gobot wheel payload verified at {package_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
