"""PEP 517 wrapper that prepares Gobot's default native SDKs.

Published Gobot wheels already contain these runtimes. Source and editable
builds use this wrapper so the same complete feature set is built before
scikit-build-core configures the main project.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scikit_build_core import build as _backend


ROOT = Path(__file__).resolve().parent
SKIP_BOOTSTRAP_ENV = "GOBOT_SKIP_DEPENDENCY_BOOTSTRAP"

_SUBMODULE_MARKERS = {
    "3rdparty/assimp": "CMakeLists.txt",
    "3rdparty/gli": "CMakeLists.txt",
    "3rdparty/imgui": "imgui.cpp",
    "3rdparty/luisa_compute": "CMakeLists.txt",
    "3rdparty/libuipc": "CMakeLists.txt",
    "3rdparty/meshoptimizer": "CMakeLists.txt",
    "3rdparty/onetbb": "CMakeLists.txt",
    "3rdparty/openusd": "pxr/pxrConfig.cmake.in",
    "3rdparty/pybind11": "CMakeLists.txt",
    "3rdparty/rttr": "CMakeLists.txt",
    "3rdparty/stb": "stb_image.h",
}
_LUISA_NESTED_MARKERS = (
    "src/ext/reproc/CMakeLists.txt",
    "src/ext/spdlog/CMakeLists.txt",
)
_LIBUIPC_NESTED_MARKERS = (
    "external/muda/CMakeLists.txt",
    "scripts/SymEigen/SymEigen.py",
)
_SDK_ARTIFACTS = (
    "build/luisa_compute/install/bin/luisa_nvrtc",
    "build/luisa_compute/install/lib/cmake/LuisaCompute/LuisaComputeTargets.cmake",
    "build/gsplat_inference/install/lib/libgobot_gsplat_inference.a",
    "build/openusd/install/pxrConfig.cmake",
    "build/openusd/install/lib/usd/plugInfo.json",
    "build/openusd/install/share/licenses/OpenUSD/LICENSE.txt",
    "build/openusd/install/share/licenses/oneTBB/LICENSE.txt",
)


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "on", "true", "yes"}


def _find_build_tool(name: str) -> str | None:
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        candidate = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return os.fspath(candidate)
    return shutil.which(name)


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Gobot dependency bootstrap could not run {command[0]!r}. "
            "Install the source-build prerequisites or install the published "
            "Gobot wheel instead."
        ) from error
    except subprocess.CalledProcessError as error:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Gobot dependency bootstrap command failed ({error.returncode}): {rendered}"
        ) from error


def _missing_submodule_paths(root: Path) -> list[str]:
    return [
        path
        for path, marker in _SUBMODULE_MARKERS.items()
        if not (root / path / marker).is_file()
    ]


def _submodule_status(
    root: Path,
    paths: tuple[str, ...] = (),
    *,
    recursive: bool = False,
) -> tuple[str, ...]:
    command = ["git", "submodule", "status"]
    if recursive:
        command.append("--recursive")
    if paths:
        command.extend(["--", *paths])
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Gobot could not verify the commits of its native dependency submodules."
        ) from error

    return tuple(line for line in result.stdout.splitlines() if line)


def _status_is_pinned(lines: tuple[str, ...]) -> bool:
    return bool(lines) and all(line.startswith(" ") for line in lines)


def _status_has_mismatch(lines: tuple[str, ...]) -> bool:
    return any(not line.startswith((" ", "-")) for line in lines)


def _ensure_source_dependencies(root: Path) -> None:
    missing = _missing_submodule_paths(root)
    has_git_metadata = (root / ".git").exists()
    if has_git_metadata and shutil.which("git") is None:
        raise RuntimeError("Gobot source checkouts require Git to verify submodule pins.")
    git_checkout = has_git_metadata
    required_submodules = tuple(_SUBMODULE_MARKERS)
    top_level_status = (
        _submodule_status(root, required_submodules) if git_checkout else ()
    )
    if _status_has_mismatch(top_level_status):
        raise RuntimeError(
            "A required Gobot submodule is checked out at a commit different "
            "from the superproject gitlink. Preserve or commit that work, then "
            "restore the pinned submodule before building."
        )
    update_top_level = git_checkout and (
        bool(missing)
        or not top_level_status
        or any(line.startswith("-") for line in top_level_status)
    )
    if update_top_level:
        # Populate missing sources at the superproject's pinned gitlinks. The
        # already-pinned fast path is local-only and avoids a network check on
        # every editable rebuild; a divergent initialized checkout fails above
        # instead of overwriting user work.
        _run(["git", "submodule", "sync", "--recursive"], cwd=root)
        _run(
            [
                "git",
                "-c",
                "protocol.version=2",
                "submodule",
                "update",
                "--init",
                "--depth=1",
                *required_submodules,
            ],
            cwd=root,
        )
    elif missing:
        paths = ", ".join(missing)
        raise RuntimeError(
            "Gobot's source archive is missing required native dependency sources: "
            f"{paths}. Install a published binary wheel, or build from a Git "
            "checkout with its submodules available."
        )

    still_missing = _missing_submodule_paths(root)
    if still_missing:
        raise RuntimeError(
            "Git completed without initializing required Gobot submodules: "
            + ", ".join(still_missing)
        )
    if update_top_level and not _status_is_pinned(
        _submodule_status(root, required_submodules)
    ):
        raise RuntimeError(
            "Required Gobot submodules do not match the commits pinned by the "
            "superproject after git submodule update."
        )

    libuipc_root = root / "3rdparty/libuipc"
    missing_libuipc_nested = [
        marker
        for marker in _LIBUIPC_NESTED_MARKERS
        if not (libuipc_root / marker).is_file()
    ]
    libuipc_nested_status = (
        _submodule_status(libuipc_root, recursive=True) if git_checkout else ()
    )
    if _status_has_mismatch(libuipc_nested_status):
        raise RuntimeError(
            "A nested libuipc submodule differs from the commit pinned by libuipc."
        )
    if git_checkout and (
        missing_libuipc_nested
        or not libuipc_nested_status
        or any(line.startswith("-") for line in libuipc_nested_status)
    ):
        _run(["git", "submodule", "sync", "--recursive"], cwd=libuipc_root)
        _run(
            ["git", "submodule", "update", "--init", "--depth=1", "--recursive"],
            cwd=libuipc_root,
        )
    elif missing_libuipc_nested:
        raise RuntimeError(
            "Gobot's source archive is missing libuipc nested sources: "
            + ", ".join(missing_libuipc_nested)
        )
    still_missing_libuipc_nested = [
        marker
        for marker in _LIBUIPC_NESTED_MARKERS
        if not (libuipc_root / marker).is_file()
    ]
    if still_missing_libuipc_nested:
        raise RuntimeError(
            "Git completed without initializing libuipc nested sources: "
            + ", ".join(still_missing_libuipc_nested)
        )
    if git_checkout and not _status_is_pinned(
        _submodule_status(libuipc_root, recursive=True)
    ):
        raise RuntimeError(
            "libuipc nested submodules do not match their pinned commits."
        )

    luisa_root = root / "3rdparty/luisa_compute"
    missing_luisa_nested = [
        marker
        for marker in _LUISA_NESTED_MARKERS
        if not (luisa_root / marker).is_file()
    ]
    luisa_nested_status = (
        _submodule_status(luisa_root, recursive=True) if git_checkout else ()
    )
    if _status_has_mismatch(luisa_nested_status):
        raise RuntimeError(
            "A LuisaCompute nested submodule is checked out at a commit "
            "different from its pinned gitlink. Preserve or commit that work, "
            "then restore the pinned submodule before building."
        )
    update_luisa_nested = git_checkout and (
        bool(missing_luisa_nested)
        or not luisa_nested_status
        or any(line.startswith("-") for line in luisa_nested_status)
    )
    if update_luisa_nested:
        _run(["git", "submodule", "sync", "--recursive"], cwd=luisa_root)
        _run(
            [
                "git",
                "-c",
                "protocol.version=2",
                "submodule",
                "update",
                "--init",
                "--depth=1",
                "--recursive",
            ],
            cwd=luisa_root,
        )
    elif missing_luisa_nested:
        raise RuntimeError(
            "LuisaCompute's nested sources are missing from a non-Git source tree. "
            "Install the published Gobot wheel or use a complete Git checkout."
        )

    still_missing_luisa_nested = [
        marker
        for marker in _LUISA_NESTED_MARKERS
        if not (luisa_root / marker).is_file()
    ]
    if still_missing_luisa_nested:
        raise RuntimeError(
            "Git completed without initializing LuisaCompute nested sources: "
            + ", ".join(still_missing_luisa_nested)
        )
    if update_luisa_nested and not _status_is_pinned(
        _submodule_status(luisa_root, recursive=True)
    ):
        raise RuntimeError(
            "LuisaCompute nested submodules do not match their pinned commits "
            "after git submodule update."
        )


def _ensure_native_sdks() -> None:
    if _is_truthy(os.environ.get(SKIP_BOOTSTRAP_ENV)):
        return

    build_root = ROOT / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    lock_path = build_root / ".dependency-bootstrap.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _ensure_source_dependencies(ROOT)

        cmake = _find_build_tool("cmake")
        ninja = _find_build_tool("ninja")
        if cmake is None or ninja is None:
            raise RuntimeError(
                "Gobot source builds require CMake 3.27+ and Ninja. "
                "They are declared as PEP 517 build requirements; ensure the "
                "build frontend did not disable build dependency installation."
            )

        dependency_build = build_root / "dependencies-pep517"
        print(
            "*** Preparing Gobot native dependency SDKs (Luisa, gsplat, OpenUSD)",
            flush=True,
        )
        _run(
            [
                cmake,
                "-S",
                os.fspath(ROOT),
                "-B",
                os.fspath(dependency_build),
                "-G",
                "Ninja",
                "-DGOB_BUILD_DEPENDENCIES_ONLY=ON",
                "-DGOB_DEPENDENCY_BUILD_LUISA=ON",
                "-DGOB_DEPENDENCY_BUILD_GSPLAT=ON",
                "-DGOB_DEPENDENCY_BUILD_OPENUSD=ON",
                f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
            ]
        )
        _run(
            [
                cmake,
                "--build",
                os.fspath(dependency_build),
                "--target",
                "gobot_dependencies",
            ]
        )

        missing_artifacts = [
            relative for relative in _SDK_ARTIFACTS if not (ROOT / relative).is_file()
        ]
        if missing_artifacts:
            raise RuntimeError(
                "Gobot dependency bootstrap completed without required artifacts: "
                + ", ".join(missing_artifacts)
            )


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_native_sdks()
    return _backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _ensure_native_sdks()
    return _backend.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
) -> str:
    return _backend.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return _backend.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return _backend.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(
    config_settings: dict[str, str | list[str]] | None = None,
) -> list[str]:
    return _backend.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
) -> str:
    return _backend.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
) -> str:
    return _backend.prepare_metadata_for_build_editable(metadata_directory, config_settings)


__all__: tuple[str, ...] = (
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
)
