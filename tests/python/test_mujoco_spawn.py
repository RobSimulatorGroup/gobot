"""Offset evaluation spawns must clear terrain without changing robot pose."""
from types import SimpleNamespace
import math
import numpy as np
import pytest

from gobot.sim.providers.mujoco_spawn import MuJoCoSpawnPlacement


def placement(terrain):
    xml = f'''<mujoco><worldbody>{terrain}
      <body name="robot_base"><freejoint/><geom type="sphere" size=".05" pos="0 0 .3"/>
        <body name="foot" pos=".3 0 0"><joint name="robot_ankle"/>
          <geom type="sphere" size=".1"/></body>
      </body></worldbody></mujoco>'''
    artifact = SimpleNamespace(content=xml, robot_prefix=lambda name: 'robot_')
    return MuJoCoSpawnPlacement(artifact, robot_name='robot', base_link='base', joint_names=('ankle',))


def test_sloping_ground_lifts_toes_and_preserves_clear_spawns():
    angle = .4
    spawn = placement(f'<geom type="plane" size="0 0 .1" quat="{math.cos(angle/2)} 0 {-math.sin(angle/2)} 0"/>')
    positions = np.array([[0., 0., .1], [0., 0., .7]], dtype=np.float32)
    result = spawn.positions(positions, np.tile([1., 0., 0., 0.], (2, 1)), np.zeros((2, 1)))
    # Sphere/plane clearance is measured along the slope normal.
    expected = .3 * math.tan(angle) + .1 / math.cos(angle) + .002
    assert result[0, 2] == pytest.approx(expected, abs=2.e-5)
    np.testing.assert_array_equal(result[:, :2], positions[:, :2])
    np.testing.assert_array_equal(result[1], positions[1])
    assert positions[0, 2] == np.float32(.1)
    np.testing.assert_array_equal(spawn.positions(result, np.tile([1., 0., 0., 0.], (2, 1)), np.zeros((2, 1))), result)


def test_blocked_spawn_fails_instead_of_silently_placing_inside_geometry():
    spawn = placement('<geom type="box" pos="0 0 .5" size="1 1 1"/>')
    with pytest.raises(RuntimeError, match='cannot clear scene geometry'):
        spawn.positions([[0., 0., .1]], [[1., 0., 0., 0.]], [[0.]])
