"""Exercise the executable exit status, including errors before a world exists."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("GOBOT_RUN_RENDER_GPU_TEST") != "1", reason="requires editor display")
@pytest.mark.parametrize("mode,stage", [("start", "play_start"), ("runtime", "play_runtime"), ("timeout", "timeout")])
def test_editor_benchmark_failure_is_not_a_successful_capture(tmp_path, mode, stage):
    (tmp_path / "project.gobot").write_text(json.dumps({"main_scene": "res://test.jscn"}))
    if mode != "timeout":
        scene = {"__VERSION__": 3, "__META_TYPE__": "SCENE", "__TYPE__": "PackedScene",
                 "__SUB_RESOURCES__": [], "__EXT_RESOURCES__": [
                     {"__ID__": "script", "__PATH__": "res://failure.py", "__TYPE__": "PythonScript"}],
                 "__NODES__": [{"name": "failure", "type": "Node3D", "parent": -1,
                                "properties": {"script": "ExtResource(script)"}}]}
        (tmp_path / "test.jscn").write_text(json.dumps(scene))
        callback = "_ready(self)" if mode == "start" else "_process(self, delta)"
        (tmp_path / "failure.py").write_text(
            f'import gobot\nclass Script(gobot.NodeScript):\n    def {callback}:\n        raise RuntimeError("benchmark injected failure")\n')
    report = tmp_path / "capture.json"
    command = [str(Path(sys.executable).with_name("gobot_editor")), "--path", str(tmp_path),
               "--benchmark-play", "--benchmark-warmup", "0", "--benchmark-frames", "60",
               "--benchmark-timeout", "3", "--benchmark-output", str(report)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert result.returncode == 2, result.stdout + result.stderr
    capture = json.loads(report.read_text())
    assert capture["status"] == "failed"
    assert capture["failure_stage"] == stage, capture
    assert capture["exit_code"] == 2 and capture["error"]
    if mode != "timeout":
        assert "benchmark injected failure" in capture["error"]
