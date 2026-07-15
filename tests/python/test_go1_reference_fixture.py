from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.go1.tools.go1_parity import (  # noqa: E402
    PARITY_ACTIONS,
    PARITY_NUM_ENVS,
    PARITY_SEED,
    PARITY_TERRAIN_ROWS,
    PARITY_TERRAIN_SEED,
    PARITY_TERRAIN_TYPES,
    REFERENCE_REVISION,
    REFERENCE_COMPAT_PATCH_ID,
    REFERENCE_TASK_ID,
    TRACE_SCHEMA_VERSION,
    read_trace,
)
from examples.go1.train.go1_velocity_cfg import (  # noqa: E402
    GO1_JOINT_NAMES,
    GO1_ROUGH_REWARD_TERM_NAMES,
)
from examples.go1.tools.compare_go1_traces import compare_traces  # noqa: E402


FIXTURE = REPO_ROOT / "tests/fixtures/go1/reference_trace_v1.json"


def main() -> int:
    trace = read_trace(FIXTURE)
    metadata = trace["metadata"]
    assert trace["schema_version"] == TRACE_SCHEMA_VERSION
    assert metadata["source_revision"] == REFERENCE_REVISION
    assert metadata["source_compat_patch_id"] == REFERENCE_COMPAT_PATCH_ID
    assert metadata["task_id"] == REFERENCE_TASK_ID
    assert metadata["seed"] == PARITY_SEED
    assert metadata["terrain_seed"] == PARITY_TERRAIN_SEED
    assert metadata["joint_names"] == list(GO1_JOINT_NAMES)
    assert metadata["reward_names"] == list(GO1_ROUGH_REWARD_TERM_NAMES)
    assert metadata["actor_dim"] == 235
    assert metadata["critic_dim"] == 259
    assert len(trace["snapshots"]) == len(PARITY_ACTIONS) + 1
    assert trace["mujoco_model"]["dimensions"]["nq"] == 19
    assert trace["mujoco_model"]["dimensions"]["nv"] == 18
    assert len(trace["mujoco_model"]["heightfields"]) == 40
    matrix = trace["terrain_matrix"]
    assert matrix["rows"] == PARITY_TERRAIN_ROWS
    assert matrix["cols"] == len(PARITY_TERRAIN_TYPES)
    assert matrix["type_names"] == list(PARITY_TERRAIN_TYPES)
    assert len(matrix["levels"]) == PARITY_NUM_ENVS
    assert len(matrix["origins"]) == PARITY_NUM_ENVS
    assert len(matrix["reset"]["height_scan"]) == PARITY_NUM_ENVS
    assert trace["snapshots"][0]["action"] is None
    for snapshot, action in zip(trace["snapshots"][1:], PARITY_ACTIONS, strict=True):
        assert snapshot["action"] == list(action)

    assert compare_traces(trace, deepcopy(trace)) == []

    version_regression = deepcopy(trace)
    version_regression["metadata"]["packages"]["mujoco"] = "different-version"
    assert any(
        "metadata.packages.mujoco" in error
        for error in compare_traces(trace, version_regression)
    )

    impact_regression = deepcopy(trace)
    wave_index = PARITY_TERRAIN_TYPES.index("wave")
    impact_regression["terrain_matrix"]["first_step"]["qpos"][wave_index][0] += 1.0
    assert compare_traces(trace, impact_regression) == []

    invalid_impact = deepcopy(trace)
    invalid_impact["terrain_matrix"]["first_step"]["qpos"][wave_index][0] = float("nan")
    assert any("non-finite" in error for error in compare_traces(trace, invalid_impact))

    non_wave_regression = deepcopy(trace)
    non_wave_regression["terrain_matrix"]["first_step"]["qpos"][0][0] += 1.0
    assert any(
        "terrain_matrix.first_step.qpos" in error
        for error in compare_traces(trace, non_wave_regression)
    )

    reset_geometry_regression = deepcopy(trace)
    reset_geometry_regression["terrain_matrix"]["reset"]["height_scan"][wave_index][0] += 1.0
    assert any(
        "terrain_matrix.reset.height_scan" in error
        for error in compare_traces(trace, reset_geometry_regression)
    )

    visual_dimension_difference = deepcopy(trace)
    visual_dimension_difference["mujoco_model"]["dimensions"]["ncam"] += 1
    assert compare_traces(trace, visual_dimension_difference) == []

    core_dimension_regression = deepcopy(trace)
    core_dimension_regression["mujoco_model"]["dimensions"]["nq"] += 1
    assert any(
        "mujoco_model.dimensions.nq" in error
        for error in compare_traces(trace, core_dimension_regression)
    )

    terrain_hash_regression = deepcopy(trace)
    terrain_hash_regression["mujoco_model"]["heightfields"][0][
        "physical_steps_sha256"
    ] = "different"
    assert any(
        "mujoco_model.heightfields[0].physical_steps_sha256" in error
        for error in compare_traces(trace, terrain_hash_regression)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
