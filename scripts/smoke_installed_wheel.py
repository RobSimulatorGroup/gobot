#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


NATIVE_PAYLOAD = (
    "libgobot.so",
    "libgobot_luisa_renderer.so",
    "luisa/libluisa-backend-cuda.so",
    "luisa/libluisa-runtime.so",
    "luisa/luisa_nvrtc",
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

    package_root = Path(gobot.__file__).resolve().parent
    missing_payload = [name for name in NATIVE_PAYLOAD if not (package_root / name).is_file()]
    if missing_payload:
        raise RuntimeError("installed wheel is missing: " + ", ".join(missing_payload))

    if gobot.render.RenderBuffer is not gobot._core.RenderBuffer:
        raise RuntimeError("gobot.render.RenderBuffer does not expose the native wheel type")
    if gobot.render.RenderFrame is not gobot._core.RenderFrame:
        raise RuntimeError("gobot.render.RenderFrame does not expose the native wheel type")
    if gobot.render.RenderProduct is not gobot._core.RenderProduct:
        raise RuntimeError("gobot.render.RenderProduct does not expose the native wheel type")
    if "torch" in sys.modules:
        raise RuntimeError("importing gobot must not eagerly import or initialize Torch")

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
