"""The display must preserve force signs/units and never count self-contact as support."""
import numpy as np
import pytest
from examples.g1.tracking.mjlab_contacts import pack_contacts, foot_resultants, contact_arrows


def pack(geoms, normals, forces, *, active=None):
    count = len(geoms)
    return pack_contacts(
        body_ids=(1, 2), body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        geom_bodies=np.array([0, 1, 2]), geoms=np.array(geoms),
        positions=np.tile([2, 3, 0], (count, 1)), normals=np.array(normals),
        forces=np.array(forces), active=np.ones(count, bool) if active is None else active,
        capacity=8,
    )


def test_force_on_robot_is_independent_of_geom_pair_order():
    a = pack([[0, 1]], [[0, 0, 1]], [[2, -3, 120]])
    b = pack([[1, 0]], [[0, 0, -1]], [[-2, 3, -120]])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[0, 7:10], [2, -3, 120])
    np.testing.assert_array_equal(foot_resultants(a), [[2, -3, 120], [0, 0, 0]])


def test_inactive_contact_and_self_contact_are_not_ground_support():
    a = pack([[0, 1], [1, 2], [0, 2]], [[0, 0, 1]] * 3,
             [[0, 0, 100], [0, 0, 40], [0, 0, 60]], active=[False, True, True])
    assert int(a[:, 0].sum()) == 2
    assert a[0, 13] == 1 and a[0, 12] == 0
    np.testing.assert_array_equal(foot_resultants(a), [[0, 0, 0], [0, 0, 60]])
    # An airborne sample replaces the old contacts with a zero buffer.
    empty = pack([], [], [])
    assert not empty.any()


def test_force_arrows_use_metres_per_newton_and_common_display_translation():
    a = pack([[0, 1]], [[0, 0, 1]], [[3, 4, 0]])
    settings = dict(draw_contacts=False, draw_contact_forces=True,
                    contact_force_scale=.1, contact_force_max_length=.4)
    arrows = contact_arrows(a, offset=(-3.1, -3, 0), settings=settings)
    assert len(arrows) == 2  # individual force + explicitly labeled resultant
    np.testing.assert_allclose(arrows[0].start, [-1.1, 0, 0])
    np.testing.assert_allclose(arrows[0].vector, [.24, .32, 0], atol=1e-7)
    assert arrows[1].label == "L foot sum: 5.0 N"
    settings["draw_contact_forces"] = False
    assert not contact_arrows(a, offset=(0, 0, 0), settings=settings)
    settings["draw_contacts"] = True
    normals = contact_arrows(a, offset=(0, 0, 0), settings=settings)
    assert len(normals) == 1
    np.testing.assert_allclose(normals[0].vector, [0, 0, .09])


def test_nonfinite_contact_is_rejected():
    with pytest.raises(FloatingPointError):
        pack([[0, 1]], [[0, 0, 1]], [[0, 0, np.nan]])
