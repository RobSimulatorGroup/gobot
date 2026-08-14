from __future__ import annotations

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT / "python"))

from gobot_cli import editor as launcher


def test_wheel_install_does_not_rebuild() -> None:
    run = Mock()
    with (
        patch.object(launcher, "_editable_source_root", return_value=None),
        patch.object(launcher.subprocess, "run", run),
    ):
        launcher._rebuild_editable_native_artifacts()
    run.assert_not_called()


def test_editable_install_rebuilds_with_environment_tools() -> None:
    invocations = []

    def run(command, **kwargs):
        invocations.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "CMakeCache.txt").write_text("", encoding="utf-8")
        finder = SimpleNamespace(
            path=directory,
            dir=directory,
            build_options=[],
            install_options=[],
        )
        loader = SimpleNamespace(_skbuild_finder=finder)
        with (
            patch.object(launcher, "_editable_source_root", return_value=Path("/source")),
            patch.object(launcher, "__loader__", loader),
            patch.object(
                launcher, "_editable_build_needs_reconfigure", return_value=False
            ),
            patch.dict(launcher.os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": ""}),
            patch.object(launcher.subprocess, "run", side_effect=run),
        ):
            launcher._rebuild_editable_native_artifacts()

    assert len(invocations) == 2
    build_command, build_kwargs = invocations[0]
    install_command, install_kwargs = invocations[1]
    assert build_command == [
        "cmake",
        "--build",
        ".",
        "--parallel",
        str(min(os.cpu_count() or 1, 4)),
    ]
    assert install_command[-2:] == ["--component", "python"]
    assert build_kwargs["env"]["PATH"].split(os.pathsep)[0] == os.fspath(
        Path(launcher.sys.executable).absolute().parent
    )
    assert install_kwargs["cwd"] == build_kwargs["cwd"]


def test_editable_install_rebuilds_stale_bundled_gsplat() -> None:
    invocations = []

    def run(command, **kwargs):
        invocations.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    with tempfile.TemporaryDirectory() as directory:
        source_root = Path(directory)
        build_dir = source_root / "build" / "cp-test"
        install_dir = source_root / "build" / "gsplat_inference" / "install"
        gsplat_source = source_root / "3rdparty" / "gsplat_inference" / "src" / "renderer.cu"
        dependency_project = source_root / "cmake" / "GobotDependencies.cmake"
        library = install_dir / "lib" / "libgobot_gsplat_inference.a"
        for parent in (
            build_dir,
            gsplat_source.parent,
            dependency_project.parent,
            library.parent,
        ):
            parent.mkdir(parents=True, exist_ok=True)
        build_dir.joinpath("CMakeCache.txt").write_text(
            "GOB_BUILD_GSPLAT_INFERENCE:BOOL=ON\n"
            f"GOB_GSPLAT_INFERENCE_ROOT:PATH={install_dir}\n",
            encoding="utf-8",
        )
        library.write_bytes(b"old")
        gsplat_source.write_text("// changed\n", encoding="utf-8")
        dependency_project.write_text("# dependency project\n", encoding="utf-8")
        os.utime(library, ns=(1_000_000_000, 1_000_000_000))
        os.utime(gsplat_source, ns=(2_000_000_000, 2_000_000_000))

        finder = SimpleNamespace(
            path=os.fspath(build_dir),
            dir=os.fspath(source_root / "site-packages"),
            build_options=[],
            install_options=[],
        )
        loader = SimpleNamespace(_skbuild_finder=finder)
        with (
            patch.object(launcher, "_editable_source_root", return_value=source_root),
            patch.object(launcher, "__loader__", loader),
            patch.object(
                launcher, "_editable_build_needs_reconfigure", return_value=False
            ),
            patch.object(launcher.subprocess, "run", side_effect=run),
        ):
            launcher._rebuild_editable_native_artifacts()

    assert len(invocations) == 4
    assert invocations[0][0][:3] == ["cmake", "-S", os.fspath(source_root)]
    assert "-DGOB_BUILD_DEPENDENCIES_ONLY=ON" in invocations[0][0]
    assert f"-DGOB_GSPLAT_INSTALL_DIR={install_dir}" in invocations[0][0]
    assert invocations[1][0][:3] == [
        "cmake",
        "--build",
        os.fspath(source_root / "build" / "dependencies-gsplat"),
    ]
    assert invocations[1][0][3:5] == ["--target", "gobot_gsplat_sdk"]
    assert invocations[2][0][:3] == ["cmake", "--build", "."]


def test_editable_install_repairs_isolated_cmake_cache() -> None:
    invocations = []

    def run(command, **kwargs):
        invocations.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    with tempfile.TemporaryDirectory() as directory:
        source_root = Path(directory)
        build_dir = source_root / "build" / "cp-test"
        install_dir = source_root / ".venv" / "lib" / "python" / "site-packages"
        build_dir.mkdir(parents=True)
        install_dir.mkdir(parents=True)
        build_dir.joinpath("CMakeCache.txt").write_text(
            "CMAKE_COMMAND:INTERNAL=/deleted/build-env/bin/cmake\n"
            "CMAKE_GENERATOR:INTERNAL=Ninja\n"
            "CMAKE_INSTALL_PREFIX:PATH=/tmp/deleted-wheel/platlib\n"
            "Python3_EXECUTABLE:FILEPATH=/deleted/build-env/bin/python\n"
            "GOB_BUILD_GSPLAT_INFERENCE:BOOL=OFF\n",
            encoding="utf-8",
        )
        finder = SimpleNamespace(
            path=os.fspath(build_dir),
            dir=os.fspath(install_dir),
            build_options=[],
            install_options=[],
        )
        loader = SimpleNamespace(_skbuild_finder=finder)
        with (
            patch.object(launcher, "_editable_source_root", return_value=source_root),
            patch.object(launcher, "__loader__", loader),
            patch.object(launcher.shutil, "which", return_value="/stable/bin/ninja"),
            patch.object(launcher.subprocess, "run", side_effect=run),
        ):
            launcher._rebuild_editable_native_artifacts()

    assert len(invocations) == 3
    configure_command, configure_kwargs = invocations[0]
    assert configure_command[:3] == ["cmake", "-S", os.fspath(source_root)]
    assert "-B" in configure_command
    assert (
        f"-DPython3_EXECUTABLE:FILEPATH={Path(launcher.sys.executable).absolute()}"
        in configure_command
    )
    assert f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}" in configure_command
    assert "-DCMAKE_MAKE_PROGRAM:FILEPATH=/stable/bin/ninja" in configure_command
    assert configure_kwargs["cwd"] == source_root
    assert invocations[1][0][:3] == ["cmake", "--build", "."]
    assert invocations[2][0][:3] == ["cmake", "--install", "."]


def test_editable_install_repairs_missing_cached_ninja() -> None:
    with tempfile.TemporaryDirectory() as directory:
        build_dir = Path(directory)
        install_dir = build_dir / "site-packages"
        install_dir.mkdir()
        build_dir.joinpath("CMakeCache.txt").write_text(
            f"CMAKE_COMMAND:INTERNAL={launcher.sys.executable}\n"
            "CMAKE_GENERATOR:INTERNAL=Ninja\n"
            "CMAKE_MAKE_PROGRAM:FILEPATH=/deleted/build-env/bin/ninja\n"
            f"CMAKE_INSTALL_PREFIX:PATH={install_dir}\n"
            f"Python3_EXECUTABLE:FILEPATH={launcher.sys.executable}\n",
            encoding="utf-8",
        )

        assert launcher._editable_build_needs_reconfigure(
            build_dir, install_dir, os.environ.copy()
        )


def test_editor_stops_instead_of_launching_stale_binary() -> None:
    with (
        patch.object(
            launcher,
            "_rebuild_editable_native_artifacts",
            side_effect=RuntimeError("native rebuild failed"),
        ),
        patch.object(launcher, "_register_packaged_examples") as register_examples,
        redirect_stderr(io.StringIO()) as stderr,
    ):
        assert launcher.editor() == 1

    register_examples.assert_not_called()
    assert "native rebuild failed" in stderr.getvalue()


def test_editor_preloads_current_environment_cuda_runtime() -> None:
    runtime = Path("/environment/nvidia/cuda_runtime/lib/libcudart.so.12")
    with (
        patch.object(
            launcher, "_find_current_cuda_runtime_library", return_value=runtime
        ),
        patch.object(launcher, "_distribution_gobot_dir", return_value=None),
        patch.dict(
            launcher.os.environ,
            {"LD_PRELOAD": "/user/libinstrumentation.so"},
            clear=False,
        ),
    ):
        environment = launcher._with_editor_python_environment(
            "/environment/lib/libpython.so"
        )

    assert environment["LD_PRELOAD"].split(os.pathsep) == [
        os.fspath(runtime),
        "/user/libinstrumentation.so",
    ]


def test_editor_cuda_runtime_is_optional() -> None:
    with (
        patch.object(
            launcher, "_find_current_cuda_runtime_library", return_value=None
        ),
        patch.object(launcher, "_distribution_gobot_dir", return_value=None),
        patch.dict(launcher.os.environ, {}, clear=True),
    ):
        environment = launcher._with_editor_python_environment(
            "/environment/lib/libpython.so"
        )

    assert "LD_PRELOAD" not in environment


def main() -> None:
    test_wheel_install_does_not_rebuild()
    test_editable_install_rebuilds_with_environment_tools()
    test_editable_install_rebuilds_stale_bundled_gsplat()
    test_editable_install_repairs_isolated_cmake_cache()
    test_editable_install_repairs_missing_cached_ninja()
    test_editor_stops_instead_of_launching_stale_binary()
    test_editor_preloads_current_environment_cuda_runtime()
    test_editor_cuda_runtime_is_optional()


if __name__ == "__main__":
    main()
