"""The FR3 worker controller can run without importing Play or rendering."""
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[2] / "examples/libuipc"
sys.path.insert(0, str(PROJECT))
import fr3_grasp as grasp
from libuipc_runtime import GraspController


class Provider:
    def __init__(self):
        self.joints = [dict(path="world/fr3/" + name) for name in grasp.FR3_JOINT_NAMES]
        self.targets = {}

    def set_joint_target(self, path, value):
        self.targets[path.rsplit("/", 1)[-1]] = value


class LibuipcGraspTest(unittest.TestCase):
    def test_trajectory_grasp_lift_release_and_reset(self):
        provider = Provider()
        control = GraspController(provider)
        initial = provider.targets.copy()
        self.assertEqual(initial["fr3_finger_joint1"], .017)
        for _ in range(600):
            control.step(.01)
        self.assertAlmostEqual(provider.targets["fr3_finger_joint1"], .0146)
        self.assertAlmostEqual(provider.targets["fr3_joint2"], .023901723 - .07)
        self.assertAlmostEqual(provider.targets["fr3_joint4"], -2.3683236 + .055)
        for _ in range(400):
            control.step(.01)
        self.assertEqual(provider.targets, initial)
        control.step(.01)
        control.reset((0,))
        self.assertEqual(control.time, 0.)
        self.assertEqual(provider.targets, initial)
        with self.assertRaises(ValueError):
            control.reset((1,))
        with self.assertRaises(ValueError):
            control.apply_commands({"move": 1})

    def test_all_required_joints_must_resolve_uniquely(self):
        provider = Provider()
        provider.joints.pop()
        with self.assertRaisesRegex(ValueError, "no unique"):
            GraspController(provider)
        provider = Provider()
        provider.joints.append(provider.joints[0])
        with self.assertRaisesRegex(ValueError, "no unique"):
            GraspController(provider)

    def test_worker_import_has_no_engine_or_play_dependency(self):
        code = '''import sys
from libuipc_runtime import GraspController
from fr3_grasp import FR3_JOINT_NAMES
class Provider:
    joints = [{"path": "fr3/" + name} for name in FR3_JOINT_NAMES]
    def set_joint_target(self, path, value): pass
GraspController(Provider()).step(.01)
assert "libuipc_demo" not in sys.modules
assert "gobot" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=PROJECT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
