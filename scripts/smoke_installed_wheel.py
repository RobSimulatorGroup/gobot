#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import sys
import tempfile


NATIVE_PAYLOAD = (
    "libgobot_libuipc_solver.so",
    "libgobot.so",
    "libgobot_luisa_renderer.so",
    "libuipc/Release/bin/libuipc_backend_cuda.so",
    "licenses/mujoco/LICENSE",
    "licenses/mujoco/LICENSES_THIRD_PARTY.md",
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
OPENUSD_PAYLOAD = (
    "openusd/lib/libtbb.so.12",
    "openusd/lib/libusd_usd.so",
    "openusd/lib/libusd_usdGeom.so",
    "openusd/lib/usd/plugInfo.json",
    "openusd/plugin/usd/plugInfo.json",
    "openusd/share/licenses/OpenUSD/LICENSE.txt",
    "openusd/share/licenses/oneTBB/LICENSE.txt",
)
ELF_PAYLOAD = (
    "libgobot_libuipc_solver.so",
    "libgobot_luisa_renderer.so",
    "libuipc/Release/bin/libuipc_backend_cuda.so",
    "luisa/libluisa-backend-cuda.so",
    "luisa/luisa_nvrtc",
)
ALLOWED_MISSING_LIBRARIES = {
    "libcuda.so.1",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudart.so.12",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libnvJitLink.so.12",
}
DEFAULT_DISTRIBUTIONS = (
    "mujoco",
    "mujoco-warp",
    "newton",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nvjitlink-cu12",
    "torch",
    "warp-lang",
)


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
        for name in (
            *NATIVE_PAYLOAD,
            *OPENUSD_PAYLOAD,
            *GO1_POLICY_PAYLOAD,
            *OTHER_EXAMPLE_PAYLOAD,
        )
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

    if os.environ.get("GOBOT_SMOKE_REQUIRE_DEFAULT_DEPENDENCIES") == "1":
        missing_distributions = []
        for distribution in DEFAULT_DISTRIBUTIONS:
            try:
                version(distribution)
            except PackageNotFoundError:
                missing_distributions.append(distribution)
        if missing_distributions:
            raise RuntimeError(
                "default pip install is missing distributions: "
                + ", ".join(missing_distributions)
            )
        from gobot.ipc._libuipc_provider import (
            _preload_libuipc_cuda_libraries,
        )

        _preload_libuipc_cuda_libraries()
        availability = gobot.rl.NewtonProvider.availability()
        if not availability.available:
            raise RuntimeError(
                "default pip install cannot resolve Newton provider dependencies: "
                + availability.reason
            )
        import torch

        if torch.version.cuda is None:
            raise RuntimeError(
                "default pip install resolved a CPU-only Torch build; "
                "Gobot's default Newton runtime requires CUDA-enabled Torch"
            )

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

        usd_path = Path(temporary_directory) / "wheel_smoke.usda"
        usd_path.write_text(
            """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Mesh "Triangle"
    {
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
""",
            encoding="utf-8",
        )
        usd_resource = gobot.load_resource(str(usd_path), "PackedScene")
        if usd_resource.get("type") != "PackedScene":
            raise RuntimeError(
                "installed wheel cannot import USD with its bundled OpenUSD runtime"
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
