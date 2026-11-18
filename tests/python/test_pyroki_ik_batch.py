"""Torch port of PyRoki tests/test_ik_vmap.py at 388e43e (MIT, Chung Min Kim).

The Panda fixture retains the seven arm joints through panda_hand. Each batch
test runs offline and checks numerical results in addition to upstream shapes.
"""

from copy import deepcopy

import mujoco
import numpy as np
import pytest
import torch

from panda_planning import PANDA_URDF, check_pose, panda, reference_pose
from gobot.planning import Capsule, Planner, PlannerBatch, PoseTarget


BASE = {"panda": "panda_link0"}
HAND = ("panda", "panda_hand")
ELBOW = ("panda", "panda_link4")
ROOT = ("panda", "panda_link0")
SEED = torch.tensor([0, -0.5, 0, -2, 0, 1.5, 0.8], dtype=torch.float64)
DOWN = torch.diag(SEED.new_tensor([1, -1, -1]))


def hand_capsule(radius):
    return Capsule((0, 0, -0.02), (0, 0, 0.02), radius)


def test_unit_collision_batching(panda):
    # Two capsules and the same one active pair in all five rows.
    models = [
        Planner(panda, BASE, link_capsules={
            ROOT: [], ELBOW: [hand_capsule(0.10 + i * 0.01)], HAND: [hand_capsule(0.10 + i * 0.01)],
        }) for i in range(5)
    ]
    q = SEED.expand(5, -1)
    d = PlannerBatch(models).collision_distances(q)
    assert d.shape == (5, 1)
    torch.testing.assert_close(d, torch.stack([model.collision_distances(SEED) for model in models]))
    torch.testing.assert_close(d.diff(dim=0), d.new_full((4, 1), -0.02))


def test_unit_collision_batching_topology_mismatch(panda):
    # Equal shapes and joint order can still encode different active pairs.
    first = Planner(panda, BASE, ignored_link_pairs=[(ROOT, HAND)])
    second = Planner(panda, BASE, ignored_link_pairs=[(ELBOW, HAND)])
    assert len(first.collision_pairs) == len(second.collision_pairs)
    assert first.collision_pairs != second.collision_pairs
    with pytest.raises(ValueError, match="topology"):
        PlannerBatch([first, second])


@pytest.mark.parametrize("wide_workspace", (False, True), ids=("nearby", "unit_cube"))
def test_integration_ik_basic(panda, wide_workspace):
    planner = Planner(panda, BASE, link_capsules={ROOT: [], ELBOW: [], HAND: []})
    positions = torch.rand(5, 3, generator=torch.Generator().manual_seed(0), dtype=SEED.dtype)
    if not wide_workspace:
        positions = SEED.new_tensor([0.5, 0.0, 0.5]) + 0.12 * (positions - 0.5)
    target = PoseTarget(HAND, positions, DOWN.expand(5, -1, -1))
    result = PlannerBatch(planner).solve_ik([target], seed=SEED)
    assert result.positions.shape == (5, 7)
    if wide_workspace:
        assert result.success.any() and (~result.success).any()
    else:
        assert result.success.all()
    reference = mujoco.MjModel.from_xml_path(str(PANDA_URDF))
    for q, success, position in zip(result.positions, result.success, positions, strict=True):
        if wide_workspace:
            scalar = planner.solve_ik([PoseTarget(HAND, position, DOWN)], seed=SEED)
            assert success.item() == scalar.success
        if success:
            check_pose(reference, planner, q, HAND, position, DOWN)
        else:
            assert q.isnan().all()


def test_integration_ik_mixed_batching(panda):
    # The third hand capsule makes this otherwise reachable pose infeasible.
    models = [Planner(panda, BASE, link_capsules={HAND: [hand_capsule(r)]}) for r in (0.03, 0.04, 1.5)]
    positions = SEED.new_tensor([0.5, 0, 0.5]).expand(3, -1)
    result = PlannerBatch(models).solve_ik([PoseTarget(HAND, positions, DOWN)], seed=SEED)
    assert result.positions.shape == (3, 7)
    assert result.success.tolist() == [True, True, False]
    assert result.positions[2].isnan().all()
    reference = mujoco.MjModel.from_xml_path(str(PANDA_URDF))
    for i, model in enumerate(models[:2]):
        check_pose(reference, model, result.positions[i], HAND, positions[i], DOWN)
        assert model.collision_distances(result.positions[i]).min() >= 0.02 - 1e-6


def test_integration_ik_fully_batched(panda):
    models, references, positions, rotations = [], [], [], []
    links = [HAND, ("panda", "panda_link7"), HAND]
    delta = SEED.new_tensor([0.1, -0.06, 0.02, 0.04, -0.1, 0.10, -0.05])
    seeds = torch.stack([SEED - delta, SEED, SEED + delta])
    for row, link in enumerate(links):
        # Translate the entire robot and extend the fixed hand mount along
        # its local z axis. Snapshot collision frames are in world space.
        reference = mujoco.MjModel.from_xml_path(str(PANDA_URDF))
        reference.body("panda_link0").pos[0] += row * 0.1
        reference.body("panda_hand").pos[2] += row * 0.02
        snapshot = deepcopy(panda)
        robot = snapshot["robots"][0]
        for body in robot["links"] + robot["joints"]:
            matrix = body["global_transform"]["matrix"]
            shift = np.array([row * 0.1, 0., 0.])
            if body["name"] == "panda_hand":
                shift += matrix[:3, 2] * (row * 0.02)
            matrix[:3, 3] += shift
            for shape in body.get("shapes", ()):
                shape["transform"]["matrix"][:3, 3] += shift
        model = Planner(snapshot, BASE, link_capsules={HAND: [hand_capsule(0.02 + row * 0.01)]})
        models.append(model)
        references.append(reference)
        position, rotation = reference_pose(reference, model, seeds[row] + delta * 0.7, link)
        positions.append(position)
        rotations.append(rotation)

    target = PoseTarget(links, torch.stack(positions), torch.stack(rotations))
    result = PlannerBatch(models).solve_ik([target], seed=seeds)
    assert result.positions.shape == (3, 7)
    assert result.success.all()
    for row, model in enumerate(models):
        check_pose(references[row], model, result.positions[row], links[row], positions[row], rotations[row])
        scalar = model.solve_ik([PoseTarget(links[row], positions[row], rotations[row])], seed=seeds[row])
        assert scalar.success, scalar.reason
        torch.testing.assert_close(result.positions[row], scalar.positions, atol=1e-6, rtol=0)
        assert model.collision_distances(result.positions[row]).min() >= 0.02 - 1e-6
