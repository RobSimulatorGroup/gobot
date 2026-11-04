"""Validate release payloads against the current authored CartPole example."""
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from zipfile import ZipFile, ZipInfo

from scripts import smoke_installed_wheel, verify_release_wheel

ROOT = Path(__file__).resolve().parents[2]
CARTPOLE = ROOT / "examples/cartpole"


class ReleaseWheelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gobot-wheel-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_wheel(self, omit=()):
        wheel_path = self.root / "gobot-0.1.18-cp310-cp310-manylinux_2_35_x86_64.whl"
        payload = {
            name: b"fixture"
            for name in verify_release_wheel.REQUIRED_PAYLOAD
            if not name.startswith("gobot/examples/cartpole/")
        }
        # Source inputs are independent of the validator's list. In particular,
        # there are no retired trained policies in this authored example.
        for relative in ("project.gobot", "cartpole.jscn", "cartpole_task.py",
                         "cartpole_swingup.py", "scripts/cartpole.py", "README.md"):
            payload["gobot/examples/cartpole/" + relative] = (CARTPOLE / relative).read_bytes()
        payload["gobot/_core.cpython-310-x86_64-linux-gnu.so"] = b"fixture"
        metadata = "Metadata-Version: 2.1\nName: gobot\nVersion: 0.1.18\n"
        metadata += "".join("Requires-Dist: " + name + "\n" for name in
                            sorted(verify_release_wheel.REQUIRED_DEPENDENCIES))
        payload["gobot-0.1.18.dist-info/METADATA"] = metadata.encode()
        with ZipFile(wheel_path, "w") as wheel:
            for name, content in payload.items():
                if name not in omit:
                    info = ZipInfo(name)
                    info.external_attr = (stat.S_IFREG | 0o755) << 16
                    wheel.writestr(info, content)
        return wheel_path

    def installed_payload(self, wheel_path):
        directory = self.root / "installed"
        with ZipFile(wheel_path) as wheel:
            wheel.extractall(directory)
        return directory / "gobot"

    def test_current_cartpole_passes_archive_and_installed_checks_without_policies(self):
        wheel = self.make_wheel()
        self.assertEqual(verify_release_wheel.verify_wheel(wheel), [])
        package = self.installed_payload(wheel)
        self.assertFalse((package / "examples/cartpole/policies").exists())
        smoke_installed_wheel.verify_payload(package)

    def test_missing_planner_fails_both_release_checks(self):
        missing = "gobot/examples/cartpole/cartpole_swingup.py"
        wheel = self.make_wheel(omit=(missing,))
        self.assertIn("missing payload: " + missing, verify_release_wheel.verify_wheel(wheel))
        with self.assertRaisesRegex(RuntimeError, "cartpole_swingup.py"):
            smoke_installed_wheel.verify_payload(self.installed_payload(wheel))

    def test_go1_release_policy_remains_required(self):
        missing = "gobot/examples/go1/policies/go1_velocity.onnx"
        wheel = self.make_wheel(omit=(missing,))
        self.assertIn("missing payload: " + missing, verify_release_wheel.verify_wheel(wheel))
        with self.assertRaisesRegex(RuntimeError, "go1_velocity.onnx"):
            smoke_installed_wheel.verify_payload(self.installed_payload(wheel))

    def test_cmake_installs_current_example_and_excludes_retired_local_policies(self):
        # Exercise the production examples install rule without building native
        # runtimes or copying unrelated large sample assets.
        cmake = shutil.which("cmake")
        self.assertIsNotNone(cmake)
        source = self.root / "source"
        example = source / "examples/cartpole"
        example.mkdir(parents=True)
        for relative in ("project.gobot", "cartpole.jscn", "cartpole_task.py",
                         "cartpole_swingup.py", "scripts/cartpole.py"):
            target = example / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CARTPOLE / relative, target)
        (example / "policies").mkdir()
        for name in ("cartpole.pt", "cartpole.onnx"):
            (example / "policies" / name).write_bytes(b"retired local artifact")
        cmake_source = (ROOT / "CMakeLists.txt").read_text()
        start = cmake_source.index('install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"')
        end = cmake_source.index('\ninstall(FILES', start)
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\nproject(example_install NONE)\n"
            + cmake_source[start:end] + "\n")
        build, install = self.root / "build", self.root / "install"
        subprocess.run([cmake, "-S", str(source), "-B", str(build)], check=True, capture_output=True)
        subprocess.run([cmake, "--install", str(build), "--prefix", str(install), "--component", "python"],
                       check=True, capture_output=True)
        installed = install / "gobot"
        for name in smoke_installed_wheel.CARTPOLE_PAYLOAD:
            self.assertTrue((installed / name).is_file(), name)
        self.assertFalse((installed / "examples/cartpole/policies").exists())


if __name__ == "__main__":
    unittest.main()
