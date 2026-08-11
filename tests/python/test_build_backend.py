from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _unexpected_backend_hook(*args: object, **kwargs: object) -> None:
    raise AssertionError("unpatched scikit-build-core hook was called")


# scikit-build-core is a PEP 517 build requirement, not a Gobot runtime
# dependency. Native CMake tests exercise this wrapper without installing a
# Python build frontend, so provide just the delegation surface under test.
_backend_stub = SimpleNamespace(
    build_editable=_unexpected_backend_hook,
    build_sdist=_unexpected_backend_hook,
    build_wheel=_unexpected_backend_hook,
    get_requires_for_build_editable=_unexpected_backend_hook,
    get_requires_for_build_sdist=_unexpected_backend_hook,
    get_requires_for_build_wheel=_unexpected_backend_hook,
    prepare_metadata_for_build_editable=_unexpected_backend_hook,
    prepare_metadata_for_build_wheel=_unexpected_backend_hook,
)
_scikit_build_core_stub = ModuleType("scikit_build_core")
_scikit_build_core_stub.build = _backend_stub  # type: ignore[attr-defined]
with patch.dict(sys.modules, {"scikit_build_core": _scikit_build_core_stub}):
    import gobot_build_backend as backend


def test_python_build_defaults_enable_complete_native_runtime() -> None:
    pyproject = (backend.ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'build-backend = "gobot_build_backend"' in pyproject
    assert 'backend-path = ["."]' in pyproject
    for define in (
        "GOB_BUILD_ASSIMP",
        "GOB_BUILD_MUJOCO",
        "GOB_BUILD_EGL",
        "GOB_BUILD_LUISA_RENDERER",
        "GOB_BUILD_LIBUIPC",
        "GOB_BUILD_GSPLAT_INFERENCE",
        "GOB_BUILD_OPENUSD",
        "GOB_BUNDLE_OPENUSD_RUNTIME",
    ):
        assert f'{define} = "ON"' in pyproject
    for requirement in (
        '"mujoco==3.10.0;',
        '"mujoco-warp==3.10.0.2;',
        '"newton[onnx,sim]==1.4.0;',
        '"nvidia-cublas-cu12>=12.8,<13;',
        '"nvidia-cuda-runtime-cu12>=12.8,<13;',
        '"nvidia-cusolver-cu12>=11.7,<12;',
        '"nvidia-cusparse-cu12>=12.5,<13;',
        '"nvidia-nvjitlink-cu12>=12.8,<13;',
        '"torch>=2.7;',
        '"warp-lang==1.15.0;',
    ):
        assert requirement in pyproject
    assert "[project.optional-dependencies]" not in pyproject


def test_release_wheel_provisions_libuipc_sources_and_native_dependencies() -> None:
    workflow = (
        backend.ROOT / ".github" / "workflows" / "python-publish.yml"
    ).read_text(encoding="utf-8")

    assert "3rdparty/libuipc" in workflow
    assert "git -C 3rdparty/libuipc submodule sync --recursive" in workflow
    assert "git -C 3rdparty/libuipc -c protocol.version=2 submodule update" in workflow
    assert "libtbb-dev" in workflow
    assert "liburdfdom-dev" in workflow
    assert "cuda-profiler-api-12-8=12.8.90-1" in workflow
    assert "libcublas-dev-12-8=12.8.4.1-1" in workflow
    assert "libcusolver-dev-12-8=11.7.3.90-1" in workflow
    assert "libcusparse-dev-12-8=12.5.8.93-1" in workflow
    assert "libnvjitlink-dev-12-8=12.8.93-1" in workflow
    assert "build/cuda-toolkit/bin/nvcc" in workflow
    assert "build/cuda-toolkit/nvvm/bin/cicc" in workflow
    assert "build/cuda-toolkit/include/cuda_profiler_api.h" in workflow
    assert "build/cuda-toolkit/lib64/libcudadevrt.a" in workflow
    assert 'CUDA_PACKAGE_TARGET_ROOT="build/cuda-toolkit/targets/x86_64-linux"' in workflow
    assert "ln -s targets/x86_64-linux/include" in workflow
    assert 'GOBOT_CUDA_BUILD_SDK_ROOT="$RUNNER_TEMP/gobot-cuda-toolkit"' in workflow
    assert "-DCUDAToolkit_ROOT=$GOBOT_CUDA_BUILD_SDK_ROOT" in workflow
    assert "--exclude libcublas.so.12" in workflow
    assert "--exclude libcusolver.so.11" in workflow
    assert "--exclude libcusparse.so.12" in workflow
    assert "--exclude libnvJitLink.so.12" in workflow
    assert (
        r"GOB_LIBUIPC_CUDA_ARCHITECTURES='75-real\;80-real\;86-real\;89-real"
        in workflow
    )


def test_removed_warp_ipc_demo_is_not_packaged() -> None:
    cmake = (backend.ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'DESTINATION "gobot/examples" COMPONENT python' in cmake
    assert 'PATTERN "warp_ipc/assets" EXCLUDE' not in cmake
    assert 'PATTERN "*.ptx" EXCLUDE' in cmake
    assert not (backend.ROOT / "examples" / "warp_ipc").exists()


def test_wheel_and_editable_hooks_prepare_dependencies() -> None:
    with (
        patch.object(backend, "_ensure_native_sdks") as ensure,
        patch.object(backend._backend, "build_wheel", return_value="gobot.whl") as wheel,
        patch.object(
            backend._backend, "build_editable", return_value="gobot-editable.whl"
        ) as editable,
    ):
        assert backend.build_wheel("dist") == "gobot.whl"
        assert backend.build_editable("dist") == "gobot-editable.whl"

    assert ensure.call_count == 2
    wheel.assert_called_once_with("dist", None, None)
    editable.assert_called_once_with("dist", None, None)


def test_metadata_and_sdist_do_not_prepare_dependencies() -> None:
    with (
        patch.object(backend, "_ensure_native_sdks") as ensure,
        patch.object(
            backend._backend,
            "prepare_metadata_for_build_wheel",
            return_value="gobot.dist-info",
        ),
        patch.object(backend._backend, "build_sdist", return_value="gobot.tar.gz"),
    ):
        assert (
            backend.prepare_metadata_for_build_wheel("metadata")
            == "gobot.dist-info"
        )
        assert backend.build_sdist("dist") == "gobot.tar.gz"

    ensure.assert_not_called()


def test_skip_environment_avoids_dependency_commands() -> None:
    with (
        patch.dict(os.environ, {backend.SKIP_BOOTSTRAP_ENV: "1"}),
        patch.object(backend, "_run") as run,
    ):
        backend._ensure_native_sdks()
    run.assert_not_called()


def test_build_tool_prefers_current_virtual_environment() -> None:
    with tempfile.TemporaryDirectory() as directory:
        environment = Path(directory)
        tool = environment / "bin/cmake"
        tool.parent.mkdir()
        tool.touch(mode=0o755)
        with (
            patch.object(backend.sys, "prefix", os.fspath(environment)),
            patch.object(backend.sys, "base_prefix", "/usr"),
            patch.object(backend.shutil, "which", return_value="/usr/bin/cmake"),
        ):
            assert backend._find_build_tool("cmake") == os.fspath(tool)


def test_native_sdk_bootstrap_uses_dependency_only_cmake_project() -> None:
    invocations: list[tuple[list[str], Path]] = []

    def run(command: list[str], *, cwd: Path = backend.ROOT) -> None:
        invocations.append((command, cwd))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (
            patch.object(backend, "ROOT", root),
            patch.object(backend, "_SDK_ARTIFACTS", ()),
            patch.object(backend, "_ensure_source_dependencies"),
            patch.object(
                backend,
                "_find_build_tool",
                side_effect=lambda name: f"/tools/{name}",
            ),
            patch.object(backend, "_run", side_effect=run),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop(backend.SKIP_BOOTSTRAP_ENV, None)
            backend._ensure_native_sdks()

    assert len(invocations) == 2
    configure = invocations[0][0]
    build = invocations[1][0]
    assert configure[:3] == ["/tools/cmake", "-S", os.fspath(root)]
    assert "-DGOB_BUILD_DEPENDENCIES_ONLY=ON" in configure
    assert "-DGOB_DEPENDENCY_BUILD_LUISA=ON" in configure
    assert "-DGOB_DEPENDENCY_BUILD_GSPLAT=ON" in configure
    assert "-DGOB_DEPENDENCY_BUILD_OPENUSD=ON" in configure
    assert build[-2:] == ["--target", "gobot_dependencies"]


def test_checkout_submodules_are_revalidated_at_pinned_gitlinks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        root.joinpath(".git").mkdir()
        luisa_root = root / "3rdparty/luisa_compute"
        for marker in backend._LUISA_NESTED_MARKERS:
            path = luisa_root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        libuipc_root = root / "3rdparty/libuipc"
        for marker in backend._LIBUIPC_NESTED_MARKERS:
            path = libuipc_root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        with (
            patch.object(
                backend,
                "_missing_submodule_paths",
                return_value=[],
            ),
            patch.object(
                backend,
                "_submodule_status",
                side_effect=[
                    ("-abc 3rdparty/openusd",),
                    (" abc 3rdparty/openusd",),
                    ("-def external/muda",),
                    (" def external/muda",),
                    ("-123 src/ext/reproc",),
                    (" 123 src/ext/reproc",),
                ],
            ),
            patch.object(backend.shutil, "which", return_value="/usr/bin/git"),
            patch.object(backend, "_run") as run,
        ):
            backend._ensure_source_dependencies(root)

    assert run.call_count == 6
    update = run.call_args_list[1].args[0]
    assert "3rdparty/assimp" in update
    assert "3rdparty/luisa_compute" in update
    assert "3rdparty/openusd" in update
    assert "--depth=1" in update
    libuipc_update = run.call_args_list[3].args[0]
    assert "--recursive" in libuipc_update
    assert run.call_args_list[3].kwargs["cwd"] == libuipc_root
    luisa_update = run.call_args_list[5].args[0]
    assert "--recursive" in luisa_update
    assert run.call_args_list[5].kwargs["cwd"] == luisa_root


def test_pinned_checkout_does_not_run_submodule_update() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        root.joinpath(".git").mkdir()
        luisa_root = root / "3rdparty/luisa_compute"
        for marker in backend._LUISA_NESTED_MARKERS:
            path = luisa_root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        libuipc_root = root / "3rdparty/libuipc"
        for marker in backend._LIBUIPC_NESTED_MARKERS:
            path = libuipc_root / marker
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        with (
            patch.object(backend, "_missing_submodule_paths", return_value=[]),
            patch.object(
                backend,
                "_submodule_status",
                return_value=(" abc pinned/submodule",),
            ),
            patch.object(backend.shutil, "which", return_value="/usr/bin/git"),
            patch.object(backend, "_run") as run,
        ):
            backend._ensure_source_dependencies(root)

    run.assert_not_called()


def test_mismatched_checkout_is_not_overwritten() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        root.joinpath(".git").mkdir()
        with (
            patch.object(backend, "_missing_submodule_paths", return_value=[]),
            patch.object(
                backend,
                "_submodule_status",
                return_value=("+abc 3rdparty/openusd",),
            ),
            patch.object(backend.shutil, "which", return_value="/usr/bin/git"),
            patch.object(backend, "_run") as run,
        ):
            try:
                backend._ensure_source_dependencies(root)
            except RuntimeError as error:
                message = str(error)
            else:
                raise AssertionError("mismatched submodule was accepted")

    run.assert_not_called()
    assert "different from the superproject gitlink" in message


def test_source_archive_without_submodules_fails_clearly() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(
            backend, "_missing_submodule_paths", return_value=["3rdparty/openusd"]
        ):
            try:
                backend._ensure_source_dependencies(root)
            except RuntimeError as error:
                message = str(error)
            else:
                raise AssertionError("missing source dependencies were accepted")

    assert "published binary wheel" in message
    assert "3rdparty/openusd" in message


def main() -> None:
    test_python_build_defaults_enable_complete_native_runtime()
    test_release_wheel_provisions_libuipc_sources_and_native_dependencies()
    test_removed_warp_ipc_demo_is_not_packaged()
    test_wheel_and_editable_hooks_prepare_dependencies()
    test_metadata_and_sdist_do_not_prepare_dependencies()
    test_skip_environment_avoids_dependency_commands()
    test_build_tool_prefers_current_virtual_environment()
    test_native_sdk_bootstrap_uses_dependency_only_cmake_project()
    test_checkout_submodules_are_revalidated_at_pinned_gitlinks()
    test_pinned_checkout_does_not_run_submodule_update()
    test_mismatched_checkout_is_not_overwritten()
    test_source_archive_without_submodules_fails_clearly()


if __name__ == "__main__":
    main()
