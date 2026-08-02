from __future__ import annotations

from contextlib import redirect_stdout
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType
from typing import Callable
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "newton_g1"
REVISION = "261cd1f429619d8ef4f546bd788ab9dea906b5e1"


def load_downloader() -> ModuleType:
    name = "_gobot_test_newton_g1_download_assets"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / "download_assets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


class Response(io.BytesIO):
    def __init__(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class TrackedResponse(Response):
    def __init__(self, data: bytes, on_close: Callable[[], None]) -> None:
        super().__init__(data)
        self._on_close = on_close
        self._notified = False

    def close(self) -> None:
        if not self._notified:
            self._notified = True
            self._on_close()
        super().close()


def test_manifest_is_complete_and_pinned() -> None:
    downloader = load_downloader()
    manifest = json.loads((EXAMPLE / "asset_manifest.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["files"]}

    assert manifest["revision"] == REVISION
    assert downloader.REVISION == REVISION
    assert len(manifest["files"]) == 4
    assert len(paths) == 4
    assert sum(entry["size"] for entry in manifest["files"]) == 39_970_770
    assert downloader.TOTAL_BYTES == 39_970_770
    assert all(len(entry["git_blob_sha1"]) == 40 for entry in manifest["files"])

    assert {path for path in paths if "/usd/" in path} == {
        "unitree_g1/usd/g1_isaac.usd",
    }
    assert {path for path in paths if "/rl_policies/" in path} == {
        "unitree_g1/rl_policies/LICENSE",
        "unitree_g1/rl_policies/g1_29dof.yaml",
        "unitree_g1/rl_policies/mjw_g1_29DOF.onnx",
    }
    assert not {path for path in paths if "/mjcf/" in path}
    assert not {path for path in paths if "/meshes/" in path}

    assert downloader.DEFAULT_BASE_URLS == (
        f"https://raw.githubusercontent.com/newton-physics/newton-assets/{REVISION}",
        f"https://cdn.jsdelivr.net/gh/newton-physics/newton-assets@{REVISION}",
    )
    assert downloader.SOURCE_USD.as_posix() == "unitree_g1/usd/g1_isaac.usd"
    assert downloader.SCENE_CACHE_VERSION == 8
    identity = downloader.scene_cache_identity("test-version")
    assert identity["task_config_sha256"] == hashlib.sha256(
        downloader.TASK_CONFIG_PATH.read_bytes()
    ).hexdigest()


def test_large_asset_uses_git_blob_api_with_raw_response() -> None:
    downloader = load_downloader()
    content = b"official-usd"
    asset = downloader.Asset("unitree_g1/usd/test.usd", len(content), blob_sha1(content))
    requests: list[object] = []

    def opener(request: object, timeout: int) -> Response:
        requests.append(request)
        assert timeout == downloader.NETWORK_TIMEOUT_SECONDS
        return Response(content, headers={"Content-Length": str(len(content))})

    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "test.usd.part"
        with patch.object(downloader, "GITHUB_RAW_SIZE_LIMIT", 0):
            urls = downloader.asset_urls(downloader.DEFAULT_BASE_URLS, asset)
            assert urls[0] == f"{downloader.GITHUB_BLOB_API}/{asset.git_blob_sha1}"
            downloader.download_from_url(
                urls[0],
                destination,
                asset,
                progress=lambda current: None,
                opener=opener,
            )

        assert destination.read_bytes() == content

    assert len(requests) == 1
    assert requests[0].get_header("Accept") == "application/vnd.github.raw+json"


def test_scene_cache_is_versioned_and_skips_current_import() -> None:
    downloader = load_downloader()
    calls: list[tuple[Path, Path, Path]] = []

    def importer(project_dir: Path, source: Path, destination: Path) -> str:
        calls.append((project_dir, source, destination))
        destination.write_text('{"__TYPE__":"PackedScene"}\n', encoding="utf-8")
        return "0.1.test"

    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        output = project / "assets"
        source = output.joinpath(*downloader.SOURCE_USD.parts)
        source.parent.mkdir(parents=True)
        source.write_text("#usda 1.0\n", encoding="utf-8")
        progress = io.StringIO()
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(progress),
        ):
            assert downloader.ensure_generated_scene(
                project,
                output,
                gobot_version="0.1.test",
                importer=importer,
            )
            assert not downloader.ensure_generated_scene(
                project,
                output,
                gobot_version="0.1.test",
                importer=importer,
            )

        scene = output.joinpath(*downloader.GENERATED_SCENE.parts)
        stamp = output.joinpath(*downloader.GENERATED_SCENE_STAMP.parts)
        assert scene.is_file() and stamp.is_file()
        assert json.loads(stamp.read_text(encoding="utf-8")) == (
            downloader.scene_cache_identity("0.1.test")
        )
        assert len(calls) == 1
        assert calls[0] == (project.resolve(), source.resolve(), scene.resolve())
        messages = [
            json.loads(line.removeprefix("GOBOT_PROGRESS "))["message"]
            for line in progress.getvalue().splitlines()
        ]
        assert messages[-1] == "G1 Gobot scene cache is current"


def test_scene_cache_reimports_when_an_external_mesh_is_missing() -> None:
    downloader = load_downloader()
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        scene = project / "assets/generated/g1_29dof.jscn"
        stamp = project / "assets/generated/g1_29dof.import.json"
        scene.parent.mkdir(parents=True)
        identity = downloader.scene_cache_identity("0.1.test")
        scene.write_text(
            json.dumps(
                {
                    "__TYPE__": "PackedScene",
                    "__EXT_RESOURCES__": [
                        {
                            "__TYPE__": "ArrayMesh",
                            "__PATH__": "res://assets/generated/g1_29dof.meshes/mesh_0000.ply",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        stamp.write_text(json.dumps(identity), encoding="utf-8")

        assert not downloader.is_scene_cache_current(
            scene, stamp, identity, project_dir=project
        )
        mesh = project / "assets/generated/g1_29dof.meshes/mesh_0000.ply"
        mesh.parent.mkdir(parents=True)
        mesh.write_bytes(b"ply\n")
        assert not downloader.is_scene_cache_current(
            scene, stamp, identity, project_dir=project
        )
        mesh.write_bytes(
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"comment Generated by Gobot\n"
            b"element vertex 1\n"
            b"property uchar gobot_padding\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"element face 1\n"
            b"property list uchar uint vertex_indices\n"
            b"end_header\n"
            + b"\0" * 26
        )
        assert downloader.is_scene_cache_current(
            scene, stamp, identity, project_dir=project
        )


def test_downloader_lazily_uses_public_gobot_scene_api() -> None:
    source = (EXAMPLE / "download_assets.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in getattr(node, "names", ())
    }
    assert "gobot" not in top_level_imports
    assert "gobot.load_scene(" in source
    assert "gobot.save_scene(" in source
    assert 'node.type_name == "Robot3D"' in source


def test_download_resumes_part_and_reports_aggregate_progress() -> None:
    downloader = load_downloader()
    content = b"abcdef"
    asset = downloader.Asset("unitree_g1/test.bin", len(content), blob_sha1(content))
    requests: list[object] = []

    def opener(request: object, timeout: int) -> Response:
        requests.append(request)
        assert request.get_header("Range") == "bytes=3-"
        assert timeout == downloader.NETWORK_TIMEOUT_SECONDS
        return Response(
            b"def",
            status=206,
            headers={"Content-Range": "bytes 3-5/6", "Content-Length": "3"},
        )

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        target = output / asset.path
        target.parent.mkdir(parents=True)
        target.with_name(target.name + ".part").write_bytes(b"abc")
        progress = io.StringIO()
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(progress),
        ):
            downloader.install_assets(
                output,
                assets=(asset,),
                base_urls=("https://primary.invalid",),
                opener=opener,
            )

        assert target.read_bytes() == content
        assert not target.with_name(target.name + ".part").exists()

    assert len(requests) == 1
    lines = progress.getvalue().splitlines()
    assert lines
    payload = json.loads(lines[-1].removeprefix("GOBOT_PROGRESS "))
    assert payload == {
        "current": 6,
        "total": 6,
        "message": "Downloading G1 assets (1/1): test.bin",
    }


def test_hash_failure_switches_to_fallback_mirror_immediately() -> None:
    downloader = load_downloader()
    content = b"expected"
    asset = downloader.Asset("unitree_g1/test.bin", len(content), blob_sha1(content))
    urls: list[str] = []

    def opener(request: object, timeout: int) -> Response:
        del timeout
        urls.append(request.full_url)
        if request.full_url.startswith("https://primary.invalid"):
            return Response(b"corrupt!")
        assert request.get_header("Range") is None
        return Response(content)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(io.StringIO()),
        ):
            downloader.install_assets(
                output,
                assets=(asset,),
                base_urls=("https://primary.invalid", "https://fallback.invalid"),
                opener=opener,
            )
        assert (output / asset.path).read_bytes() == content

    assert urls == [
        "https://primary.invalid/unitree_g1/test.bin",
        "https://fallback.invalid/unitree_g1/test.bin",
    ]


def test_failed_mirrors_are_retried_in_rounds() -> None:
    downloader = load_downloader()
    asset = downloader.Asset("unitree_g1/test.bin", 1, blob_sha1(b"x"))
    urls: list[str] = []

    def opener(request: object, timeout: int) -> Response:
        del timeout
        urls.append(request.full_url)
        raise RuntimeError("offline")

    with tempfile.TemporaryDirectory() as directory:
        with (
            patch.object(downloader, "RETRIES_PER_URL", 2),
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(io.StringIO()),
        ):
            try:
                downloader.install_assets(
                    Path(directory),
                    assets=(asset,),
                    base_urls=("https://primary.invalid", "https://fallback.invalid"),
                    jobs=1,
                    opener=opener,
                )
            except downloader.DownloadError:
                pass
            else:
                raise AssertionError("offline mirrors were accepted")

    assert urls == [
        "https://primary.invalid/unitree_g1/test.bin",
        "https://fallback.invalid/unitree_g1/test.bin",
        "https://primary.invalid/unitree_g1/test.bin",
        "https://fallback.invalid/unitree_g1/test.bin",
    ]


def test_verified_cache_never_opens_network() -> None:
    downloader = load_downloader()
    content = b"cached"
    asset = downloader.Asset("unitree_g1/test.bin", len(content), blob_sha1(content))

    def unexpected_opener(*args: object, **kwargs: object) -> Response:
        del args, kwargs
        raise AssertionError("cache hit attempted network access")

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        target = output / asset.path
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        progress = io.StringIO()
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(progress),
        ):
            downloader.install_assets(
                output,
                assets=(asset,),
                opener=unexpected_opener,
            )

    line = progress.getvalue().strip()
    assert json.loads(line.removeprefix("GOBOT_PROGRESS "))["current"] == len(content)


def test_incomplete_download_is_retained_for_resume() -> None:
    downloader = load_downloader()
    content = b"abcdef"
    asset = downloader.Asset("unitree_g1/test.bin", len(content), blob_sha1(content))

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(io.StringIO()),
        ):
            try:
                downloader.install_assets(
                    output,
                    assets=(asset,),
                    base_urls=("https://primary.invalid",),
                    opener=lambda request, timeout: Response(b"abc"),
                )
            except downloader.DownloadError as error:
                assert "incomplete response" in str(error)
            else:
                raise AssertionError("incomplete response was accepted")

        temporary = output / "unitree_g1" / "test.bin.part"
        assert temporary.read_bytes() == b"abc"


def test_failed_forced_download_does_not_replace_installed_asset() -> None:
    downloader = load_downloader()
    content = b"abcdef"
    asset = downloader.Asset("unitree_g1/test.bin", len(content), blob_sha1(content))

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        target = output / asset.path
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(io.StringIO()),
        ):
            try:
                downloader.install_assets(
                    output,
                    assets=(asset,),
                    base_urls=("https://primary.invalid",),
                    force=True,
                    opener=lambda request, timeout: Response(b"abcdeg"),
                )
            except downloader.DownloadError as error:
                assert "SHA-1 mismatch" in str(error)
            else:
                raise AssertionError("corrupt replacement was accepted")

        assert target.read_bytes() == content
        assert not target.with_name(target.name + ".part").exists()


def run_parallelism_case(
    downloader: ModuleType,
    *,
    expected_workers: int,
    jobs: int | None,
) -> None:
    contents = {
        f"file_{index}.bin": bytes((index + 1, index + 2))
        for index in range(expected_workers + 2)
    }
    assets = tuple(
        downloader.Asset(
            f"unitree_g1/{name}",
            len(content),
            blob_sha1(content),
        )
        for name, content in contents.items()
    )
    lock = threading.Lock()
    all_workers_started = threading.Event()
    active = 0
    maximum_active = 0

    def opener(request: object, timeout: int) -> Response:
        del timeout
        name = request.full_url.rsplit("/", 1)[-1]
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == expected_workers:
                all_workers_started.set()

        notified = False

        def finished() -> None:
            nonlocal active, notified
            with lock:
                if not notified:
                    notified = True
                    active -= 1

        if not all_workers_started.wait(timeout=5):
            finished()
            raise AssertionError("expected download workers did not start concurrently")
        return TrackedResponse(contents[name], finished)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        progress = io.StringIO()
        arguments = {
            "assets": assets,
            "base_urls": ("https://primary.invalid",),
            "opener": opener,
        }
        if jobs is not None:
            arguments["jobs"] = jobs
        with (
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(progress),
        ):
            downloader.install_assets(output, **arguments)

        for name, content in contents.items():
            assert (output / "unitree_g1" / name).read_bytes() == content

    assert maximum_active == expected_workers
    payloads = [
        json.loads(line.removeprefix("GOBOT_PROGRESS "))
        for line in progress.getvalue().splitlines()
    ]
    assert payloads
    currents = [payload["current"] for payload in payloads]
    total = sum(len(content) for content in contents.values())
    assert currents == sorted(currents)
    assert all(0 <= current <= total for current in currents)
    assert payloads[-1]["current"] == total
    assert payloads[-1]["total"] == total


def test_downloads_use_default_four_worker_limit() -> None:
    downloader = load_downloader()
    assert downloader.DEFAULT_JOBS == 4
    run_parallelism_case(downloader, expected_workers=4, jobs=None)


def test_jobs_override_limits_parallel_downloads() -> None:
    downloader = load_downloader()
    run_parallelism_case(downloader, expected_workers=2, jobs=2)


def test_jobs_must_be_a_positive_integer() -> None:
    downloader = load_downloader()
    assert downloader.positive_int("3") == 3
    for value in ("0", "-2", "nope"):
        try:
            downloader.positive_int(value)
        except downloader.argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"invalid --jobs value was accepted: {value!r}")

    content = b"x"
    asset = downloader.Asset("unitree_g1/test.bin", 1, blob_sha1(content))
    with tempfile.TemporaryDirectory() as directory:
        for value in (0, -1, True):
            try:
                downloader.install_assets(Path(directory), assets=(asset,), jobs=value)
            except ValueError as error:
                assert "positive integer" in str(error)
            else:
                raise AssertionError(f"invalid jobs value was accepted: {value!r}")


def test_terminal_failure_stops_queued_downloads() -> None:
    downloader = load_downloader()
    assets = tuple(
        downloader.Asset(f"unitree_g1/file_{index}.bin", 1, blob_sha1(b"x"))
        for index in range(3)
    )
    requests: list[str] = []

    def opener(request: object, timeout: int) -> Response:
        del timeout
        requests.append(request.full_url)
        raise RuntimeError("offline")

    with tempfile.TemporaryDirectory() as directory:
        with (
            patch.object(downloader, "RETRIES_PER_URL", 1),
            patch.dict("os.environ", {"GOBOT_PROJECT_HOOK": "1"}),
            redirect_stdout(io.StringIO()),
        ):
            try:
                downloader.install_assets(
                    Path(directory),
                    assets=assets,
                    base_urls=("https://primary.invalid",),
                    jobs=1,
                    opener=opener,
                )
            except downloader.DownloadError as error:
                assert "file_0.bin" in str(error)
            else:
                raise AssertionError("terminal download failure was ignored")

    assert requests == ["https://primary.invalid/unitree_g1/file_0.bin"]


def main() -> None:
    test_manifest_is_complete_and_pinned()
    test_large_asset_uses_git_blob_api_with_raw_response()
    test_scene_cache_is_versioned_and_skips_current_import()
    test_scene_cache_reimports_when_an_external_mesh_is_missing()
    test_downloader_lazily_uses_public_gobot_scene_api()
    test_download_resumes_part_and_reports_aggregate_progress()
    test_hash_failure_switches_to_fallback_mirror_immediately()
    test_failed_mirrors_are_retried_in_rounds()
    test_verified_cache_never_opens_network()
    test_incomplete_download_is_retained_for_resume()
    test_failed_forced_download_does_not_replace_installed_asset()
    test_downloads_use_default_four_worker_limit()
    test_jobs_override_limits_parallel_downloads()
    test_jobs_must_be_a_positive_integer()
    test_terminal_failure_stops_queued_downloads()


if __name__ == "__main__":
    main()
