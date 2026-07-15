"""Compare pinned reference and Gobot Go1 task traces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .go1_parity import (
    REFERENCE_COMPAT_PATCH_ID,
    REFERENCE_REVISION,
    TRACE_SCHEMA_VERSION,
    read_trace,
)


@dataclass(frozen=True)
class NumericTolerance:
    atol: float
    rtol: float = 0.0


TOLERANCES: dict[str, NumericTolerance] = {
    "qpos": NumericTolerance(1.0e-5),
    "qvel": NumericTolerance(1.0e-4),
    "actor": NumericTolerance(3.0e-3),
    "command": NumericTolerance(1.0e-6),
    "terrain.origin": NumericTolerance(1.0e-5),
    "sensors.terrain_distances": NumericTolerance(1.0e-2),
    "sensors.terrain_hit_z": NumericTolerance(1.0e-5),
    "sensors.foot_height": NumericTolerance(1.0e-2),
    "sensors.current_air_time": NumericTolerance(1.0e-6),
    # Friction-force decomposition at first impact differs between MuJoCo Warp
    # 3.8 and 3.10 even though qpos, qvel, contact state, and rewards agree.
    "sensors.contact_force": NumericTolerance(15.0),
    "reward": NumericTolerance(1.0e-6),
    "reward_terms": NumericTolerance(1.0e-5),
}

CRITIC_STATE_DIM = 247
CRITIC_STATE_TOLERANCE = NumericTolerance(1.0e-2)
CRITIC_CONTACT_FORCE_TOLERANCE = NumericTolerance(4.0)

TERRAIN_MATRIX_TOLERANCES: dict[str, NumericTolerance] = {
    "qpos": NumericTolerance(1.0e-4),
    "qvel": NumericTolerance(1.0e-3),
    "height_scan": NumericTolerance(3.0e-3),
    "foot_height": NumericTolerance(2.0e-2),
    "reward": NumericTolerance(1.0e-4),
    "reward_terms": NumericTolerance(1.0e-3),
}

TERRAIN_ORIGIN_TOLERANCE = NumericTolerance(1.0e-5)
DYNAMICS_PACKAGE_NAMES = ("mujoco", "mujoco-warp", "warp-lang", "torch")
CORE_MODEL_DIMENSIONS = ("nq", "nv", "nu", "njnt", "nhfield")

EXACT_FIELDS = (
    "step",
    "action",
    "terrain.level",
    "terrain.type",
    "sensors.contact_found",
    "terminated",
    "truncated",
)


def _nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[part]
    return current


def _compare_metadata(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if reference.get("schema_version") != TRACE_SCHEMA_VERSION:
        errors.append(f"reference schema_version is not {TRACE_SCHEMA_VERSION}")
    if candidate.get("schema_version") != TRACE_SCHEMA_VERSION:
        errors.append(f"candidate schema_version is not {TRACE_SCHEMA_VERSION}")
    ref_meta = reference["metadata"]
    candidate_meta = candidate["metadata"]
    if ref_meta.get("source_revision") != REFERENCE_REVISION:
        errors.append(
            "reference source_revision is "
            f"{ref_meta.get('source_revision')}, expected {REFERENCE_REVISION}"
        )
    if ref_meta.get("source_compat_patch_id") != REFERENCE_COMPAT_PATCH_ID:
        errors.append(
            "reference source_compat_patch_id is "
            f"{ref_meta.get('source_compat_patch_id')}, "
            f"expected {REFERENCE_COMPAT_PATCH_ID}"
        )
    for name in (
        "task_id",
        "seed",
        "terrain_seed",
        "physics_dt",
        "decimation",
        "step_dt",
        "joint_names",
        "reward_names",
        "actor_dim",
        "critic_dim",
    ):
        if ref_meta.get(name) != candidate_meta.get(name):
            errors.append(
                f"metadata.{name}: reference={ref_meta.get(name)!r}, "
                f"candidate={candidate_meta.get(name)!r}"
            )
    reference_packages = ref_meta.get("packages", {})
    candidate_packages = candidate_meta.get("packages", {})
    for name in DYNAMICS_PACKAGE_NAMES:
        if reference_packages.get(name) != candidate_packages.get(name):
            errors.append(
                f"metadata.packages.{name}: "
                f"reference={reference_packages.get(name)!r}, "
                f"candidate={candidate_packages.get(name)!r}"
            )
    return errors


def _numeric_error(
    *,
    field: str,
    snapshot_index: int,
    expected_value: Any,
    actual_value: Any,
    tolerance: NumericTolerance,
    row_mask: np.ndarray | None = None,
) -> str | None:
    if expected_value is None or actual_value is None:
        if expected_value != actual_value:
            return f"snapshot[{snapshot_index}].{field}: one trace has no value"
        return None
    expected_array = np.asarray(expected_value, dtype=np.float64)
    actual_array = np.asarray(actual_value, dtype=np.float64)
    if expected_array.shape != actual_array.shape:
        return (
            f"snapshot[{snapshot_index}].{field}: shape {expected_array.shape} != "
            f"{actual_array.shape}"
        )
    if not bool(np.all(np.isfinite(expected_array))):
        return f"snapshot[{snapshot_index}].{field}: reference contains non-finite values"
    if not bool(np.all(np.isfinite(actual_array))):
        return f"snapshot[{snapshot_index}].{field}: candidate contains non-finite values"
    reported_rows: np.ndarray | None = None
    if row_mask is not None:
        if expected_array.ndim == 0 or expected_array.shape[0] != row_mask.shape[0]:
            return (
                f"snapshot[{snapshot_index}].{field}: cannot apply row mask of shape "
                f"{row_mask.shape} to {expected_array.shape}"
            )
        reported_rows = np.flatnonzero(row_mask)
        expected_array = expected_array[row_mask]
        actual_array = actual_array[row_mask]
    allowed = tolerance.atol + tolerance.rtol * np.abs(expected_array)
    difference = np.abs(expected_array - actual_array)
    violation = difference - allowed
    if bool(np.all(violation <= 0.0)):
        return None
    flat_index = int(np.nanargmax(violation))
    location = np.unravel_index(flat_index, difference.shape)
    reported_location = location
    if reported_rows is not None:
        reported_location = (int(reported_rows[location[0]]), *location[1:])
    return (
        f"snapshot[{snapshot_index}].{field}{reported_location}: reference="
        f"{expected_array[location]:.8g}, candidate={actual_array[location]:.8g}, "
        f"abs_diff={difference[location]:.8g}, atol={tolerance.atol:g}, "
        f"rtol={tolerance.rtol:g}"
    )


def _masked_exact_error(
    *,
    field: str,
    expected_value: Any,
    actual_value: Any,
    row_mask: np.ndarray,
) -> str | None:
    expected_array = np.asarray(expected_value)
    actual_array = np.asarray(actual_value)
    if expected_array.shape != actual_array.shape:
        return f"{field}: shape {expected_array.shape} != {actual_array.shape}"
    if expected_array.ndim == 0 or expected_array.shape[0] != row_mask.shape[0]:
        return f"{field}: cannot apply row mask of shape {row_mask.shape}"
    if np.array_equal(expected_array[row_mask], actual_array[row_mask]):
        return None
    return f"{field} differs on deterministic terrain rows"


def _compare_heightfields(
    reference_model: dict[str, Any], candidate_model: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    reference = reference_model.get("heightfields", [])
    candidate = candidate_model.get("heightfields", [])
    if len(reference) != len(candidate):
        return [
            f"mujoco_model.heightfields count: reference={len(reference)}, "
            f"candidate={len(candidate)}"
        ]
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        for field in ("rows", "cols", "physical_steps_sha256"):
            if expected.get(field) != actual.get(field):
                errors.append(f"mujoco_model.heightfields[{index}].{field} differs")
        expected_size = np.asarray(expected.get("size", []), dtype=np.float64)
        actual_size = np.asarray(actual.get("size", []), dtype=np.float64)
        if (
            expected_size.shape != (4,)
            or actual_size.shape != (4,)
            or not np.allclose(expected_size[[0, 1, 3]], actual_size[[0, 1, 3]], atol=1.0e-12)
        ):
            errors.append(
                f"mujoco_model.heightfields[{index}].horizontal_size_or_base differs"
            )
    return errors


def compare_traces(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors = _compare_metadata(reference, candidate)
    reference_model = reference.get("mujoco_model", {})
    candidate_model = candidate.get("mujoco_model", {})
    reference_dimensions = reference_model.get("dimensions", {})
    candidate_dimensions = candidate_model.get("dimensions", {})
    for field in CORE_MODEL_DIMENSIONS:
        if reference_dimensions.get(field) != candidate_dimensions.get(field):
            errors.append(
                f"mujoco_model.dimensions.{field}: "
                f"reference={reference_dimensions.get(field)!r}, "
                f"candidate={candidate_dimensions.get(field)!r}"
            )
    if reference_model.get("options") != candidate_model.get("options"):
        errors.append("mujoco_model.options differs")
    errors.extend(_compare_heightfields(reference_model, candidate_model))
    reference_matrix = reference.get("terrain_matrix")
    candidate_matrix = candidate.get("terrain_matrix")
    if not isinstance(reference_matrix, dict) or not isinstance(candidate_matrix, dict):
        errors.append("both traces must contain terrain_matrix data")
    else:
        for field in ("rows", "cols", "type_names", "levels", "types"):
            if reference_matrix.get(field) != candidate_matrix.get(field):
                errors.append(
                    f"terrain_matrix.{field}: reference={reference_matrix.get(field)!r}, "
                    f"candidate={candidate_matrix.get(field)!r}"
                )
        origin_error = _numeric_error(
            field="terrain_matrix.origins",
            snapshot_index=0,
            expected_value=reference_matrix.get("origins"),
            actual_value=candidate_matrix.get("origins"),
            tolerance=TERRAIN_ORIGIN_TOLERANCE,
        )
        if origin_error is not None:
            errors.append(origin_error)

        reset_reference = reference_matrix.get("reset", {})
        reset_candidate = candidate_matrix.get("reset", {})
        for field, tolerance in TERRAIN_MATRIX_TOLERANCES.items():
            expected_value = reset_reference.get(field)
            actual_value = reset_candidate.get(field)
            if expected_value is None and actual_value is None:
                continue
            error = _numeric_error(
                field=f"terrain_matrix.reset.{field}",
                snapshot_index=0,
                expected_value=expected_value,
                actual_value=actual_value,
                tolerance=tolerance,
            )
            if error is not None:
                errors.append(error)
        for field in ("contact_found", "terminated", "truncated"):
            if reset_reference.get(field) != reset_candidate.get(field):
                errors.append(f"terrain_matrix.reset.{field} differs")

        first_step_reference = reference_matrix.get("first_step", {})
        first_step_candidate = candidate_matrix.get("first_step", {})
        terrain_types = np.asarray(reference_matrix.get("types", []), dtype=np.int64)
        type_names = reference_matrix.get("type_names", [])
        deterministic_rows = np.ones(terrain_types.shape, dtype=bool)
        if "wave" in type_names:
            deterministic_rows &= terrain_types != type_names.index("wave")
        for field, tolerance in TERRAIN_MATRIX_TOLERANCES.items():
            expected_value = first_step_reference.get(field)
            actual_value = first_step_candidate.get(field)
            if expected_value is None and actual_value is None:
                continue
            error = _numeric_error(
                field=f"terrain_matrix.first_step.{field}",
                snapshot_index=0,
                expected_value=expected_value,
                actual_value=actual_value,
                tolerance=tolerance,
                row_mask=deterministic_rows,
            )
            if error is not None:
                errors.append(error)
        for field in ("contact_found", "terminated", "truncated"):
            error = _masked_exact_error(
                field=f"terrain_matrix.first_step.{field}",
                expected_value=first_step_reference.get(field),
                actual_value=first_step_candidate.get(field),
                row_mask=deterministic_rows,
            )
            if error is not None:
                errors.append(error)
    ref_steps = reference.get("snapshots", [])
    candidate_steps = candidate.get("snapshots", [])
    if len(ref_steps) != len(candidate_steps):
        errors.append(
            f"snapshot count: reference={len(ref_steps)}, candidate={len(candidate_steps)}"
        )
        return errors

    for index, (expected, actual) in enumerate(
        zip(ref_steps, candidate_steps, strict=True)
    ):
        for field in EXACT_FIELDS:
            if _nested(expected, field) != _nested(actual, field):
                errors.append(
                    f"snapshot[{index}].{field}: reference={_nested(expected, field)!r}, "
                    f"candidate={_nested(actual, field)!r}"
                )
        for field, tolerance in TOLERANCES.items():
            error = _numeric_error(
                field=field,
                snapshot_index=index,
                expected_value=_nested(expected, field),
                actual_value=_nested(actual, field),
                tolerance=tolerance,
            )
            if error is not None:
                errors.append(error)

        expected_critic = np.asarray(expected["critic"], dtype=np.float64)
        actual_critic = np.asarray(actual["critic"], dtype=np.float64)
        critic_checks = (
            (
                "critic_state",
                expected_critic[:CRITIC_STATE_DIM],
                actual_critic[:CRITIC_STATE_DIM],
                CRITIC_STATE_TOLERANCE,
            ),
            (
                "critic_contact_force",
                expected_critic[CRITIC_STATE_DIM:],
                actual_critic[CRITIC_STATE_DIM:],
                CRITIC_CONTACT_FORCE_TOLERANCE,
            ),
        )
        for field, expected_value, actual_value, tolerance in critic_checks:
            error = _numeric_error(
                field=field,
                snapshot_index=index,
                expected_value=expected_value,
                actual_value=actual_value,
                tolerance=tolerance,
            )
            if error is not None:
                errors.append(error)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = read_trace(args.reference)
    candidate = read_trace(args.candidate)
    errors = compare_traces(reference, candidate)
    if errors:
        print("Go1 parity failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Go1 parity passed (identical dynamics stack)")


if __name__ == "__main__":
    main()
