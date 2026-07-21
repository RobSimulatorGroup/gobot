#!/usr/bin/env python3

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
import stat
import sys
from zipfile import ZipFile

from packaging.requirements import Requirement


REQUIRED_PAYLOAD = {
    "gobot/__init__.py",
    "gobot/gobot_editor",
    "gobot/libgobot.so",
    "gobot/libgobot_editor_runtime.so",
    "gobot/libgobot_luisa_renderer.so",
    "gobot/luisa/libluisa-ast.so",
    "gobot/luisa/libluisa-backend-cuda.so",
    "gobot/luisa/libluisa-core.so",
    "gobot/luisa/libluisa-dsl.so",
    "gobot/luisa/libluisa-runtime.so",
    "gobot/luisa/libluisa-xir.so",
    "gobot/luisa/luisa_nvrtc",
    "gobot/rl/__init__.py",
    "gobot/rl/locomotion/__init__.py",
    "gobot/rl/providers/__init__.py",
    "gobot_cli/editor.py",
}
REQUIRED_DEPENDENCIES = {
    "imageio",
    "imageio-ffmpeg",
    "mujoco",
    "mujoco-warp",
    "numpy",
    "onnx",
    "onnxruntime",
    "rsl-rl-lib",
    "scipy",
    "tensorboard",
    "tensordict",
    "torch",
    "warp-lang",
}
SYSTEM_DRIVER_LIBRARY_PREFIXES = (
    "libcuda",
    "libEGL",
    "libGLdispatch",
    "libGLX",
    "libOpenGL",
    "libGL-",
    "libGL.so",
)


def normalized(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def verify_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with ZipFile(path) as wheel:
        names = set(wheel.namelist())
        missing_payload = sorted(REQUIRED_PAYLOAD - names)
        if missing_payload:
            errors.append("missing payload: " + ", ".join(missing_payload))
        if not any(
            name.startswith("gobot/_core.") and name.endswith(".so")
            for name in names
        ):
            errors.append("missing payload: gobot/_core.<python-abi>.so")
        nvrtc_path = "gobot/luisa/luisa_nvrtc"
        if nvrtc_path in names:
            nvrtc_mode = wheel.getinfo(nvrtc_path).external_attr >> 16
            if not nvrtc_mode & stat.S_IXUSR:
                errors.append("gobot/luisa/luisa_nvrtc is not executable")
        packaged_cache = sorted(name for name in names if "/luisa/.cache/" in name)
        if packaged_cache:
            errors.append("contains runtime shader cache files")
        packaged_driver_libraries = sorted(
            name
            for name in names
            if ".libs/" in name
            and Path(name).name.startswith(SYSTEM_DRIVER_LIBRARY_PREFIXES)
        )
        if packaged_driver_libraries:
            errors.append(
                "bundles system GPU driver libraries: "
                + ", ".join(packaged_driver_libraries)
            )

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            errors.append(f"expected one METADATA file, found {len(metadata_names)}")
        else:
            metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
            dependencies = {
                normalized(Requirement(value).name)
                for value in metadata.get_all("Requires-Dist", [])
            }
            missing_dependencies = sorted(REQUIRED_DEPENDENCIES - dependencies)
            if missing_dependencies:
                errors.append(
                    "missing default dependencies: " + ", ".join(missing_dependencies)
                )
            extras = metadata.get_all("Provides-Extra", [])
            if extras:
                errors.append("release wheel still declares extras: " + ", ".join(extras))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that Gobot release wheels contain the full CUDA renderer payload."
    )
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    failed = False
    for wheel in args.wheels:
        errors = verify_wheel(wheel)
        if errors:
            failed = True
            for error in errors:
                print(f"{wheel}: {error}", file=sys.stderr)
        else:
            print(f"{wheel}: full Gobot payload verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
