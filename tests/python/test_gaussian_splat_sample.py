from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import runpy
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "gaussian_splatting"
SCENE_PATH = "res://deep_blending_playroom.gsplat"

# Official INRIA playroom/cameras.json, first training camera: ID 29 (DSC05573).
SOURCE_POSITION = (3.33556638847436, 1.3063862878622587, -2.957697290551365)
SOURCE_ROTATION = (
    (0.9997802852533486, -0.019623769394678754, 0.00736809971954897),
    (0.020917397258199133, 0.9567833663374836, -0.29004836216026997),
    (-0.0013578330808429269, 0.29013875572673314, 0.9569836251026804),
)
WORLD_OFFSET = (0.0, 0.0, 3.0)


def transform_direction(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return vector[0], vector[2], -vector[1]


def transform_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
    rotated = transform_direction(point)
    return tuple(value + offset for value, offset in zip(rotated, WORLD_OFFSET))


def assert_vector_close(actual: list[float], expected: tuple[float, float, float]) -> None:
    assert len(actual) == 3
    assert all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(actual, expected))


def test_download_and_manifest_are_pinned() -> None:
    downloader = runpy.run_path(str(EXAMPLE / "download_sample.py"))
    assert downloader["REVISION"] == "ed0588b29edea35e36dad784f73c1f502cc8a0d2"
    assert downloader["EXPECTED_SIZE"] == 370_875_860
    assert downloader["EXPECTED_SHA256"] == (
        "201bc92b65594727a3ecfbe7e658c09ac3f8be753e2e2024047cd3ea1fe31d8c"
    )

    manifest = json.loads((EXAMPLE / "deep_blending_playroom.gsplat").read_text())
    assert manifest["ply"] == "assets/playroom-7000.ply"
    assert manifest["source_to_gobot"] == [
        1, 0, 0, 0,
        0, 0, 1, 0,
        0, -1, 0, 3,
        0, 0, 0, 1,
    ]


def test_download_resumes_partial_file() -> None:
    downloader = runpy.run_path(str(EXAMPLE / "download_sample.py"))
    download = downloader["download"]
    download.__globals__["EXPECTED_SIZE"] = 6

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

    def urlopen(request: object, timeout: int) -> Response:
        assert request.get_header("Range") == "bytes=3-"
        assert timeout == 30
        return Response(b"def")

    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "sample.part"
        destination.write_bytes(b"abc")
        with patch("urllib.request.urlopen", side_effect=urlopen):
            with redirect_stdout(io.StringIO()):
                download("https://example.invalid/sample.ply", destination)
        assert destination.read_bytes() == b"abcdef"


def test_initial_view_matches_training_camera() -> None:
    project = json.loads((EXAMPLE / "project.gobot").read_text())
    assert project["main_scene"] == SCENE_PATH
    assert set(project["editor_scene_views"]) == {SCENE_PATH}
    view = project["editor_scene_views"][SCENE_PATH]

    eye = transform_point(SOURCE_POSITION)
    source_forward = tuple(row[2] for row in SOURCE_ROTATION)
    forward = transform_direction(source_forward)
    at = tuple(origin + direction for origin, direction in zip(eye, forward))
    source_up = tuple(-row[1] for row in SOURCE_ROTATION)
    up = transform_direction(source_up)

    assert_vector_close(view["eye"], eye)
    assert_vector_close(view["at"], at)
    assert_vector_close(view["up"], up)
    assert math.isclose(view["fov_y"], 43.59549644014618, abs_tol=1e-12)
    assert math.isclose(sum(component * component for component in forward), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(component * component for component in up), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(a * b for a, b in zip(forward, up)), 0.0, abs_tol=1e-12)


def main() -> None:
    test_download_and_manifest_are_pinned()
    test_download_resumes_partial_file()
    test_initial_view_matches_training_camera()


if __name__ == "__main__":
    main()
