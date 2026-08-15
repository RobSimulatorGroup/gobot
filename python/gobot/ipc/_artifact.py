"""Validated portable artifacts consumed by Gobot's native IPC provider."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_SCHEMA_VERSION = 4
_MIN_READABLE_SCHEMA_VERSION = 3
_PRODUCER = "gobot"
_FORMAT = "gobot-ipc"
_MESH_ENCODING = "gobot.tetrahedral-mesh.le.v1"
_MESH_MAGIC = b"GOBTIPC1"
_SURFACE_MESH_ENCODING = "gobot.triangle-mesh.le.v1"
_SURFACE_MESH_MAGIC = b"GOBTTRI1"
_TOPOLOGY_MAGIC = b"GOBTIPCTOP1\0"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


def _parse_canonical_json(value: str) -> Mapping[str, Any]:
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in " \t\r\n":
            raise ValueError("compiled IPC artifact manifest contains non-canonical whitespace")

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("compiled IPC artifact manifest contains duplicate object keys")
        if keys != sorted(keys):
            raise ValueError("compiled IPC artifact manifest object keys are not sorted")
        return dict(pairs)

    def reject_constant(constant: str) -> Any:
        raise ValueError(f"compiled IPC artifact manifest contains {constant}")

    parsed = json.loads(
        value,
        object_pairs_hook=object_hook,
        parse_constant=reject_constant,
    )

    def validate_finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("compiled IPC artifact manifest contains a non-finite number")
        if isinstance(item, Mapping):
            for child in item.values():
                validate_finite(child)
        elif isinstance(item, list):
            for child in item:
                validate_finite(child)

    validate_finite(parsed)
    if not isinstance(parsed, Mapping):
        raise ValueError("compiled IPC artifact manifest must be a JSON object")
    return parsed


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    return value


def _require_bool(value: Any, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be a boolean")
    return value


def _require_int(
    value: Any,
    description: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{description} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{description} must be at most {maximum}")
    return value


def _require_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{description} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _require_vector(value: Any, width: int, description: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{description} must contain {width} numbers")
    return tuple(
        _require_number(component, f"{description} component") for component in value
    )


def _validate_transform(value: Any, description: str) -> None:
    transform = _require_mapping(value, description)
    matrix = _require_vector(
        transform.get("matrix_row_major"), 16, f"{description} matrix_row_major"
    )
    if any(abs(matrix[index]) > 1.0e-8 for index in (12, 13, 14)) or not math.isclose(
        matrix[15], 1.0, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError(f"{description} must be an affine 4x4 matrix")
    column_x = (matrix[0], matrix[4], matrix[8])
    column_y = (matrix[1], matrix[5], matrix[9])
    column_z = (matrix[2], matrix[6], matrix[10])
    determinant = (
        column_x[0]
        * (column_y[1] * column_z[2] - column_y[2] * column_z[1])
        - column_y[0]
        * (column_x[1] * column_z[2] - column_x[2] * column_z[1])
        + column_z[0]
        * (column_x[1] * column_y[2] - column_x[2] * column_y[1])
    )
    column_scale = math.sqrt(sum(value * value for value in column_x))
    column_scale *= math.sqrt(sum(value * value for value in column_y))
    column_scale *= math.sqrt(sum(value * value for value in column_z))
    tolerance = math.ulp(1.0) * 128.0 * column_scale
    if (
        not math.isfinite(determinant)
        or not math.isfinite(column_scale)
        or column_scale <= 0.0
        or determinant <= tolerance
    ):
        raise ValueError(
            f"{description} must be finite, non-singular, and orientation-preserving"
        )


def _validate_unique_paths(values: Sequence[Any], description: str) -> None:
    paths: list[str] = []
    for item in values:
        mapping = _require_mapping(item, description)
        path = str(mapping.get("path", ""))
        name = str(mapping.get("name", ""))
        if not path or not name:
            raise ValueError(f"{description} entries require non-empty name and path")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError(f"IPC manifest contains duplicate {description} paths")


def _validate_collision_shape(
    value: Any,
    description: str,
    decoded_surface_meshes: Mapping[str, Mapping[str, Any]],
    *,
    require_link_transform: bool,
) -> Mapping[str, Any]:
    collision = _require_mapping(value, description)
    _validate_transform(collision.get("transform"), f"{description} transform")
    if require_link_transform:
        _validate_transform(
            collision.get("link_transform"), f"{description} link transform"
        )
    _require_bool(collision.get("disabled"), f"{description} disabled flag")
    for field in ("collision_layer", "collision_mask"):
        _require_int(
            collision.get(field),
            f"{description} {field.replace('_', ' ')}",
            minimum=0,
            maximum=(1 << 32) - 1,
        )
    contact_offset = _require_number(
        collision.get("contact_offset"), f"{description} contact offset"
    )
    rest_offset = _require_number(
        collision.get("rest_offset"), f"{description} rest offset"
    )
    if contact_offset < 0.0 or rest_offset > contact_offset:
        raise ValueError(
            f"{description} offsets require contact_offset >= 0 and "
            "rest_offset <= contact_offset"
        )
    material = _require_mapping(
        collision.get("material"), f"{description} material"
    )
    for field in (
        "sliding_friction",
        "torsional_friction",
        "rolling_friction",
        "contact_compliance",
        "contact_damping",
    ):
        if _require_number(material.get(field), f"physics material {field}") < 0.0:
            raise ValueError(f"physics material {field} must be non-negative")
    restitution = _require_number(
        material.get("restitution"), "physics material restitution"
    )
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("physics material restitution must be in [0, 1]")

    shape_type = collision.get("shape_type")
    if shape_type == "box":
        size = _require_vector(collision.get("size"), 3, f"{description} size")
        if any(component <= 0.0 for component in size):
            raise ValueError(f"{description} size must be positive")
    elif shape_type == "sphere":
        if _require_number(collision.get("radius"), f"{description} radius") <= 0.0:
            raise ValueError(f"{description} radius must be positive")
    elif shape_type in ("capsule", "cylinder"):
        for field in ("radius", "height"):
            if _require_number(
                collision.get(field), f"{description} {field}"
            ) <= 0.0:
                raise ValueError(f"{description} {field} must be positive")
    elif shape_type == "triangle_mesh":
        mesh_blob = str(collision.get("mesh_blob", ""))
        if mesh_blob not in decoded_surface_meshes:
            raise ValueError(f"{description} references an unknown mesh blob")
        decoded_surface = decoded_surface_meshes[mesh_blob]
        for field in ("vertex_count", "triangle_count"):
            if _require_int(
                collision.get(field), f"{description} {field}", minimum=1
            ) != int(decoded_surface[field]):
                raise ValueError(
                    f"{description} {field} does not match its mesh blob"
                )
    else:
        raise ValueError(f"unsupported {description} type {shape_type!r}")
    return collision


def _decode_mesh_blob(data: bytes) -> Mapping[str, Any]:
    if len(data) < 24 or data[:8] != _MESH_MAGIC:
        raise ValueError("compiled IPC mesh blob has invalid magic or a truncated header")
    version, vertex_count, tetrahedron_count, surface_triangle_count = struct.unpack_from(
        "<IIII", data, 8
    )
    if version != 1:
        raise ValueError(f"unsupported compiled IPC mesh blob version {version}")
    if vertex_count < 4 or tetrahedron_count < 1:
        raise ValueError("compiled IPC mesh blob has invalid topology dimensions")
    vertex_bytes = vertex_count * 3 * 8
    tetrahedron_bytes = tetrahedron_count * 4 * 4
    surface_bytes = surface_triangle_count * 3 * 4
    expected_length = 24 + vertex_bytes + tetrahedron_bytes + surface_bytes
    if len(data) != expected_length:
        raise ValueError(
            "compiled IPC mesh blob byte length does not match its topology dimensions"
        )

    vertex_values = struct.unpack_from(f"<{vertex_count * 3}d", data, 24)
    if not all(math.isfinite(value) for value in vertex_values):
        raise ValueError("compiled IPC mesh blob contains a non-finite vertex")
    vertices = tuple(
        vertex_values[offset : offset + 3]
        for offset in range(0, len(vertex_values), 3)
    )
    tetrahedron_offset = 24 + vertex_bytes
    tetrahedra = struct.unpack_from(
        f"<{tetrahedron_count * 4}I", data, tetrahedron_offset
    )
    surface_offset = tetrahedron_offset + tetrahedron_bytes
    surface = struct.unpack_from(
        f"<{surface_triangle_count * 3}I", data, surface_offset
    )
    if any(index >= vertex_count for index in tetrahedra + surface):
        raise ValueError("compiled IPC mesh blob contains an out-of-range vertex index")
    face_counts: dict[tuple[int, int, int], int] = {}
    oriented_faces: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    referenced_vertices: set[int] = set()
    unique_tetrahedra: set[tuple[int, int, int, int]] = set()
    for offset in range(0, len(tetrahedra), 4):
        tetrahedron = tetrahedra[offset : offset + 4]
        if len(set(tetrahedron)) != 4:
            raise ValueError("compiled IPC mesh blob contains a repeated tet vertex")
        tetrahedron_key = tuple(sorted(tetrahedron))
        if tetrahedron_key in unique_tetrahedra:
            raise ValueError("compiled IPC mesh blob contains a duplicate tetrahedron")
        unique_tetrahedra.add(tetrahedron_key)
        referenced_vertices.update(tetrahedron)
        a, b, c, d = (vertices[index] for index in tetrahedron)
        ba = tuple(b[index] - a[index] for index in range(3))
        ca = tuple(c[index] - a[index] for index in range(3))
        da = tuple(d[index] - a[index] for index in range(3))
        cross = (
            ca[1] * da[2] - ca[2] * da[1],
            ca[2] * da[0] - ca[0] * da[2],
            ca[0] * da[1] - ca[1] * da[0],
        )
        signed_six_volume = sum(ba[index] * cross[index] for index in range(3))
        edge_scale = max(
            math.dist(left, right)
            for left, right in (
                (a, b),
                (a, c),
                (a, d),
                (b, c),
                (b, d),
                (c, d),
            )
        )
        volume_tolerance = math.ulp(1.0) * 128.0 * edge_scale**3
        if (
            not math.isfinite(signed_six_volume)
            or not math.isfinite(volume_tolerance)
            or signed_six_volume <= volume_tolerance
        ):
            raise ValueError(
                "compiled IPC mesh blob contains a non-positive or degenerate tetrahedron"
            )
        tet_faces = (
            (tetrahedron[1], tetrahedron[2], tetrahedron[3]),
            (tetrahedron[0], tetrahedron[3], tetrahedron[2]),
            (tetrahedron[0], tetrahedron[1], tetrahedron[3]),
            (tetrahedron[0], tetrahedron[2], tetrahedron[1]),
        )
        for face in tet_faces:
            key = tuple(sorted(face))
            existing = oriented_faces.get(key)
            if existing is None:
                oriented_faces[key] = face
            else:
                reversed_existing = (existing[0], existing[2], existing[1])
                opposite_cycles = (
                    reversed_existing,
                    (reversed_existing[1], reversed_existing[2], reversed_existing[0]),
                    (reversed_existing[2], reversed_existing[0], reversed_existing[1]),
                )
                if face not in opposite_cycles:
                    raise ValueError(
                        "compiled IPC mesh blob shared faces must have opposite orientation"
                    )
            face_counts[key] = face_counts.get(key, 0) + 1
            if face_counts[key] > 2:
                raise ValueError("compiled IPC mesh blob contains a non-manifold tet face")

    if len(referenced_vertices) != vertex_count:
        raise ValueError(
            "compiled IPC mesh blob contains a vertex not referenced by any tetrahedron"
        )

    boundary_faces = {
        key: oriented_faces[key] for key, count in face_counts.items() if count == 1
    }
    if surface_triangle_count != len(boundary_faces):
        raise ValueError(
            "compiled IPC mesh blob surface must contain every boundary face exactly once"
        )
    visited_faces: set[tuple[int, int, int]] = set()
    for offset in range(0, len(surface), 3):
        face = surface[offset : offset + 3]
        if len(set(face)) != 3:
            raise ValueError("compiled IPC mesh blob surface repeats a triangle vertex")
        key = tuple(sorted(face))
        if key not in boundary_faces:
            raise ValueError("compiled IPC mesh blob surface contains a non-boundary face")
        if key in visited_faces:
            raise ValueError("compiled IPC mesh blob surface contains a duplicate boundary face")
        visited_faces.add(key)
        expected = boundary_faces[key]
        cyclic = (
            expected,
            (expected[1], expected[2], expected[0]),
            (expected[2], expected[0], expected[1]),
        )
        if face not in cyclic:
            raise ValueError(
                "compiled IPC mesh blob surface triangle orientation must face outward"
            )

    topology_data = bytearray(_TOPOLOGY_MAGIC)
    topology_data.extend(
        struct.pack("<III", vertex_count, tetrahedron_count, surface_triangle_count)
    )
    topology_data.extend(struct.pack(f"<{len(tetrahedra)}I", *tetrahedra))
    topology_data.extend(struct.pack(f"<{len(surface)}I", *surface))
    return MappingProxyType(
        {
            "vertex_count": vertex_count,
            "tetrahedron_count": tetrahedron_count,
            "surface_triangle_count": surface_triangle_count,
            "topology_sha256": _sha256(bytes(topology_data)),
        }
    )


def _decode_surface_mesh_blob(data: bytes) -> Mapping[str, Any]:
    if len(data) < 20 or data[:8] != _SURFACE_MESH_MAGIC:
        raise ValueError(
            "compiled IPC triangle mesh blob has invalid magic or a truncated header"
        )
    version, vertex_count, triangle_count = struct.unpack_from("<III", data, 8)
    if version != 1:
        raise ValueError(f"unsupported compiled IPC triangle mesh blob version {version}")
    if vertex_count < 3 or triangle_count < 1:
        raise ValueError("compiled IPC triangle mesh blob has invalid topology dimensions")
    vertex_bytes = vertex_count * 3 * 8
    triangle_bytes = triangle_count * 3 * 4
    expected_length = 20 + vertex_bytes + triangle_bytes
    if len(data) != expected_length:
        raise ValueError(
            "compiled IPC triangle mesh blob byte length does not match its topology dimensions"
        )
    vertex_values = struct.unpack_from(f"<{vertex_count * 3}d", data, 20)
    if not all(math.isfinite(value) for value in vertex_values):
        raise ValueError("compiled IPC triangle mesh blob contains a non-finite vertex")
    vertices = tuple(
        vertex_values[offset : offset + 3]
        for offset in range(0, len(vertex_values), 3)
    )
    triangles = struct.unpack_from(
        f"<{triangle_count * 3}I", data, 20 + vertex_bytes
    )
    if any(index >= vertex_count for index in triangles):
        raise ValueError(
            "compiled IPC triangle mesh blob contains an out-of-range vertex index"
        )
    unique_triangles: set[tuple[int, int, int]] = set()
    for offset in range(0, len(triangles), 3):
        triangle = triangles[offset : offset + 3]
        if len(set(triangle)) != 3:
            raise ValueError(
                "compiled IPC triangle mesh blob contains a repeated triangle vertex"
            )
        triangle_key = tuple(sorted(triangle))
        if triangle_key in unique_triangles:
            raise ValueError(
                "compiled IPC triangle mesh blob contains a duplicate triangle"
            )
        unique_triangles.add(triangle_key)
        a, b, c = (vertices[index] for index in triangle)
        edge_ab = tuple(b[index] - a[index] for index in range(3))
        edge_ac = tuple(c[index] - a[index] for index in range(3))
        cross = (
            edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
            edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
            edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
        )
        double_area = math.sqrt(sum(value * value for value in cross))
        edge_scale = max(
            math.dist(a, b),
            math.dist(a, c),
            math.dist(b, c),
        )
        tolerance = math.ulp(1.0) * 128.0 * edge_scale * edge_scale
        if (
            not math.isfinite(double_area)
            or not math.isfinite(tolerance)
            or edge_scale <= 0.0
            or double_area <= tolerance
        ):
            raise ValueError("compiled IPC triangle mesh blob contains a degenerate triangle")
    return MappingProxyType(
        {"vertex_count": vertex_count, "triangle_count": triangle_count}
    )


@dataclass(frozen=True)
class CompiledIpcSceneArtifact:
    """Schema-v4 IPC scene manifest and its content-addressed binary blobs."""

    schema_version: int
    producer: str
    producer_version: str
    format: str
    manifest: str
    manifest_sha256: str
    blobs: Mapping[str, bytes] | Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        schema_version = _require_int(
            self.schema_version, "compiled IPC artifact schema version", minimum=1
        )
        if not _MIN_READABLE_SCHEMA_VERSION <= schema_version <= _SCHEMA_VERSION:
            if schema_version < _MIN_READABLE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported compiled IPC artifact schema {schema_version}; schema 3 "
                    "requires backend-neutral contact materials and explicit PhysicsCoupling entries"
                )
            raise ValueError(
                f"unsupported compiled IPC artifact schema {schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        producer = str(self.producer).lower()
        if producer != _PRODUCER:
            raise ValueError(f"IPC artifact producer must be {_PRODUCER!r}, got {producer!r}")
        producer_version = str(self.producer_version)
        if not producer_version:
            raise ValueError("compiled IPC artifact has no producer version")
        artifact_format = str(self.format).lower()
        if artifact_format != _FORMAT:
            raise ValueError(f"IPC artifact format must be {_FORMAT!r}, got {artifact_format!r}")

        manifest = str(self.manifest)
        try:
            manifest_data = _parse_canonical_json(manifest)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"compiled IPC artifact has invalid manifest JSON: {error}") from error
        manifest_sha256 = str(self.manifest_sha256)
        expected_manifest_sha256 = _sha256(manifest.encode("utf-8"))
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "compiled IPC artifact manifest digest mismatch: "
                f"expected {expected_manifest_sha256}, got {manifest_sha256}"
            )
        manifest_schema = _require_int(
            manifest_data.get("schema_version"),
            "compiled IPC manifest schema version",
            minimum=1,
        )
        if manifest_schema != schema_version:
            raise ValueError("compiled IPC artifact manifest schema does not match its envelope")
        if str(manifest_data.get("producer", "")).lower() != producer:
            raise ValueError("compiled IPC artifact manifest producer does not match its envelope")
        if str(manifest_data.get("producer_version", "")) != producer_version:
            raise ValueError(
                "compiled IPC artifact manifest producer version does not match its envelope"
            )
        if str(manifest_data.get("format", "")).lower() != artifact_format:
            raise ValueError("compiled IPC artifact manifest format does not match its envelope")
        if not isinstance(manifest_data.get("scene_name"), str) or not manifest_data[
            "scene_name"
        ]:
            raise ValueError("compiled IPC artifact manifest has no scene name")
        if schema_version == 3:
            legacy_static_colliders = manifest_data.get("static_colliders", [])
            if legacy_static_colliders != []:
                raise ValueError(
                    "IPC schema v3 artifacts cannot contain static_colliders"
                )
            manifest_data = dict(manifest_data)
            manifest_data["static_colliders"] = []

        blob_table = manifest_data.get("blobs", [])
        if not isinstance(blob_table, list):
            raise ValueError("compiled IPC artifact manifest blob table must be a list")
        table_by_id: dict[str, Mapping[str, Any]] = {}
        table_ids: list[str] = []
        for raw_entry in blob_table:
            entry = _require_mapping(raw_entry, "IPC manifest blob")
            blob_id = str(entry.get("id", ""))
            digest = str(entry.get("sha256", ""))
            encoding = str(entry.get("encoding", ""))
            byte_length = _require_int(
                entry.get("byte_length"), "IPC manifest blob byte length", minimum=0
            )
            digest_hex = blob_id.removeprefix("sha256:")
            if (
                not blob_id.startswith("sha256:")
                or blob_id != digest
                or len(digest_hex) != 64
                or any(character not in "0123456789abcdef" for character in digest_hex)
            ):
                raise ValueError("IPC manifest blob id must equal its SHA-256 digest")
            if encoding not in (_MESH_ENCODING, _SURFACE_MESH_ENCODING):
                raise ValueError(f"unsupported IPC blob encoding {encoding!r}")
            if blob_id in table_by_id:
                raise ValueError(f"IPC manifest contains duplicate blob {blob_id!r}")
            table_by_id[blob_id] = entry
            table_ids.append(blob_id)
        if table_ids != sorted(table_ids):
            raise ValueError("IPC manifest blob table must be sorted by content id")

        supplied_blobs: dict[str, bytes] = {}
        if isinstance(self.blobs, Mapping):
            entries: Sequence[Any] = tuple(self.blobs.items())
            for raw_id, raw_data in entries:
                blob_id = str(raw_id)
                if blob_id in supplied_blobs:
                    raise ValueError(
                        f"compiled IPC artifact contains duplicate blob {blob_id!r}"
                    )
                supplied_blobs[blob_id] = bytes(raw_data)
        else:
            for raw_entry in self.blobs:
                entry = _require_mapping(raw_entry, "compiled IPC blob")
                blob_id = str(entry.get("id", ""))
                if blob_id in supplied_blobs:
                    raise ValueError(f"compiled IPC artifact contains duplicate blob {blob_id!r}")
                if str(entry.get("sha256", "")) != blob_id:
                    raise ValueError("compiled IPC blob envelope digest does not match its id")
                if str(entry.get("encoding", "")) != str(
                    table_by_id.get(blob_id, {}).get("encoding", "")
                ):
                    raise ValueError("compiled IPC blob encoding does not match the manifest")
                data = bytes(entry.get("data", b""))
                if "byte_length" in entry and _require_int(
                    entry["byte_length"],
                    "compiled IPC blob envelope byte length",
                    minimum=0,
                ) != len(data):
                    raise ValueError(
                        "compiled IPC blob envelope byte length does not match its data"
                    )
                supplied_blobs[blob_id] = data
        if set(supplied_blobs) != set(table_by_id):
            missing = sorted(set(table_by_id) - set(supplied_blobs))
            extra = sorted(set(supplied_blobs) - set(table_by_id))
            raise ValueError(f"compiled IPC blob table mismatch; missing={missing}, extra={extra}")
        decoded_meshes: dict[str, Mapping[str, Any]] = {}
        decoded_surface_meshes: dict[str, Mapping[str, Any]] = {}
        for blob_id, data in supplied_blobs.items():
            if _sha256(data) != blob_id:
                raise ValueError(f"compiled IPC blob {blob_id!r} failed SHA-256 validation")
            if len(data) != int(table_by_id[blob_id]["byte_length"]):
                raise ValueError(f"compiled IPC blob {blob_id!r} has the wrong byte length")
            encoding = str(table_by_id[blob_id]["encoding"])
            try:
                if encoding == _MESH_ENCODING:
                    decoded_meshes[blob_id] = _decode_mesh_blob(data)
                else:
                    decoded_surface_meshes[blob_id] = _decode_surface_mesh_blob(data)
            except ValueError as error:
                raise ValueError(f"compiled IPC blob {blob_id!r} is invalid: {error}") from error

        deformables = manifest_data.get("deformable_bodies", [])
        tactile = manifest_data.get("tactile_sensors", [])
        robots = manifest_data.get("robots", [])
        static_colliders = manifest_data.get("static_colliders")
        couplings = manifest_data.get("couplings")
        deformable_attachments = manifest_data.get(
            "deformable_attachments", []
        )
        if not isinstance(couplings, list):
            raise ValueError("IPC manifest coupling table must be a list")
        if not isinstance(static_colliders, list):
            raise ValueError("IPC manifest static collider table must be a list")
        if not isinstance(deformable_attachments, list):
            raise ValueError(
                "IPC manifest deformable attachment table must be a list"
            )
        for values, name in (
            (deformables, "deformable body"),
            (tactile, "tactile sensor"),
            (robots, "robot"),
        ):
            if not isinstance(values, list):
                raise ValueError(f"IPC manifest {name} table must be a list")
            _validate_unique_paths(values, name)
        _validate_unique_paths(static_colliders, "static collider")
        static_paths = tuple(str(value["path"]) for value in static_colliders)
        if static_paths != tuple(sorted(static_paths)):
            raise ValueError("IPC static collider table must be sorted by path")
        for raw_collider in static_colliders:
            _validate_collision_shape(
                raw_collider,
                "static collision shape",
                decoded_surface_meshes,
                require_link_transform=False,
            )
        for raw_body in deformables:
            body = _require_mapping(raw_body, "deformable body")
            mesh_blob = str(body.get("mesh_blob", ""))
            if mesh_blob not in decoded_meshes:
                raise ValueError("deformable body references an unknown mesh blob")
            decoded = decoded_meshes[mesh_blob]
            for field in ("vertex_count", "tetrahedron_count", "surface_triangle_count"):
                if _require_int(
                    body.get(field), f"deformable body {field}", minimum=0
                ) != int(decoded[field]):
                    raise ValueError(
                        f"deformable body {field} does not match its mesh blob"
                    )
            density = _require_number(body.get("density"), "deformable body density")
            young_modulus = _require_number(
                body.get("young_modulus"), "deformable body Young modulus"
            )
            poisson_ratio = _require_number(
                body.get("poisson_ratio"), "deformable body Poisson ratio"
            )
            damping = _require_number(body.get("damping"), "deformable body damping")
            if (
                density <= 0.0
                or young_modulus <= 0.0
                or not -1.0 < poisson_ratio < 0.5
                or damping < 0.0
            ):
                raise ValueError("deformable body has invalid material parameters")
            _require_bool(body.get("kinematic"), "deformable body kinematic flag")
            _require_bool(
                body.get("self_collision"), "deformable body self-collision flag"
            )
            _require_int(
                body.get("collision_layer"),
                "deformable body collision layer",
                minimum=0,
                maximum=0xFFFFFFFF,
            )
            _require_int(
                body.get("collision_mask"),
                "deformable body collision mask",
                minimum=0,
                maximum=0xFFFFFFFF,
            )
            _validate_transform(body.get("transform"), "deformable body transform")
        sensor_attachment_paths: list[tuple[str, str]] = []
        for raw_sensor in tactile:
            sensor = _require_mapping(raw_sensor, "tactile sensor")
            gel_mesh_blob = str(sensor.get("gel_mesh_blob", ""))
            if gel_mesh_blob not in decoded_meshes:
                raise ValueError("tactile sensor references an unknown gel mesh blob")
            decoded = decoded_meshes[gel_mesh_blob]
            for field, decoded_field in (
                ("gel_vertex_count", "vertex_count"),
                ("gel_tetrahedron_count", "tetrahedron_count"),
            ):
                if _require_int(
                    sensor.get(field), f"tactile sensor {field}", minimum=1
                ) != int(decoded[decoded_field]):
                    raise ValueError(f"tactile sensor {field} does not match its gel mesh blob")
            resolution = sensor.get("resolution", [])
            if (
                not isinstance(resolution, list)
                or len(resolution) != 2
            ):
                raise ValueError("tactile sensor resolution must be [height, width]")
            height = _require_int(
                resolution[0], "tactile sensor image height", minimum=1
            )
            width = _require_int(
                resolution[1], "tactile sensor image width", minimum=1
            )
            topology_digest = str(sensor.get("gel_topology_sha256", ""))
            if topology_digest != decoded["topology_sha256"]:
                raise ValueError("tactile sensor gel topology digest does not match its blob")
            material = {
                name: _require_number(
                    sensor.get(name), f"tactile sensor {name.replace('_', ' ')}"
                )
                for name in (
                    "pixel_size",
                    "density",
                    "young_modulus",
                    "poisson_ratio",
                    "damping",
                    "friction_coefficient",
                )
            }
            if (
                material["pixel_size"] <= 0.0
                or material["density"] <= 0.0
                or material["young_modulus"] <= 0.0
                or not -1.0 < material["poisson_ratio"] < 0.5
                or material["damping"] < 0.0
                or material["friction_coefficient"] < 0.0
            ):
                raise ValueError("tactile sensor has invalid gel material parameters")
            near_plane = _require_number(
                sensor.get("near_plane"), "tactile sensor near plane"
            )
            far_plane = _require_number(
                sensor.get("far_plane"), "tactile sensor far plane"
            )
            if near_plane < 0.0 or far_plane <= near_plane:
                raise ValueError("tactile sensor near/far planes are invalid")
            if not isinstance(sensor.get("rgb_model"), str) or not sensor["rgb_model"]:
                raise ValueError("tactile sensor RGB model name must not be empty")
            _require_bool(sensor.get("enabled"), "tactile sensor enabled flag")
            for field in ("collision_layer", "collision_mask"):
                _require_int(
                    sensor.get(field),
                    f"tactile sensor {field.replace('_', ' ')}",
                    minimum=0,
                    maximum=0xFFFFFFFF,
                )
            _validate_transform(sensor.get("transform"), "tactile sensor transform")
            attachment = sensor.get("attachment")
            if attachment is not None:
                attachment_mapping = _require_mapping(
                    attachment, "tactile sensor attachment"
                )
                link_path = attachment_mapping.get("link_path")
                if not isinstance(link_path, str) or not link_path:
                    raise ValueError(
                        "tactile sensor attachment requires a non-empty link path"
                    )
                _validate_transform(
                    attachment_mapping.get("transform"),
                    "tactile sensor attachment transform",
                )
                sensor_attachment_paths.append((str(sensor["path"]), link_path))
            for field in ("coat_vertex_indices", "stick_vertex_indices"):
                raw_indices = sensor.get(field)
                if not isinstance(raw_indices, list):
                    raise ValueError(f"tactile sensor {field} must be a list")
                indices = tuple(
                    _require_int(
                        value, f"tactile sensor {field} entry", minimum=0
                    )
                    for value in raw_indices
                )
                if len(set(indices)) != len(indices) or any(
                    value < 0 or value >= int(decoded["vertex_count"])
                    for value in indices
                ):
                    raise ValueError(f"tactile sensor {field} is invalid")
            marker_positions = sensor.get("marker_positions", [])
            marker_tetrahedra = sensor.get("marker_tetrahedra", [])
            marker_barycentric = sensor.get("marker_barycentric", [])
            if not all(
                isinstance(values, list)
                for values in (marker_positions, marker_tetrahedra, marker_barycentric)
            ):
                raise ValueError("tactile sensor marker tables must be lists")
            marker_count = len(marker_positions)
            if marker_count != len(marker_tetrahedra) or marker_count != len(
                marker_barycentric
            ):
                raise ValueError("tactile sensor marker tables do not match")
            for marker_index, (position, tetrahedron, barycentric) in enumerate(
                zip(
                    marker_positions,
                    marker_tetrahedra,
                    marker_barycentric,
                    strict=True,
                )
            ):
                if (
                    not isinstance(position, list)
                    or len(position) != 2
                    or not 0.0
                    <= _require_number(
                        position[0], f"tactile sensor marker {marker_index} x"
                    )
                    < width
                    or not 0.0
                    <= _require_number(
                        position[1], f"tactile sensor marker {marker_index} y"
                    )
                    < height
                ):
                    raise ValueError(
                        f"tactile sensor marker {marker_index} is outside its image"
                    )
                tetrahedron_index = _require_int(
                    tetrahedron,
                    "tactile sensor marker tetrahedron",
                    minimum=0,
                )
                if tetrahedron_index >= int(decoded["tetrahedron_count"]):
                    raise ValueError("tactile sensor marker references an invalid tetrahedron")
                weights = _require_vector(
                    barycentric, 4, "tactile sensor marker barycentric weights"
                )
                if any(value < -1.0e-8 for value in weights) or abs(
                    sum(weights) - 1.0
                ) > 1.0e-5:
                    raise ValueError("tactile sensor marker has invalid barycentric weights")

        robot_link_paths: set[str] = set()
        robot_links_by_path: dict[str, tuple[str, Mapping[str, Any]]] = {}
        robot_collision_paths: set[str] = set()
        for raw_robot in robots:
            robot = _require_mapping(raw_robot, "robot")
            robot_path = str(robot["path"])
            robot_name = str(robot["name"])
            _validate_transform(robot.get("transform"), "robot transform")
            links = robot.get("links")
            joints = robot.get("joints")
            root_link_paths = robot.get("root_link_paths")
            if not isinstance(links, list) or not links:
                raise ValueError("IPC robot must contain at least one link")
            if not isinstance(joints, list) or not isinstance(root_link_paths, list):
                raise ValueError("IPC robot joint and root-link tables must be lists")
            _validate_unique_paths(links, "robot link")
            _validate_unique_paths(joints, "robot joint")
            link_by_name: dict[str, Mapping[str, Any]] = {}
            for raw_link in links:
                link = _require_mapping(raw_link, "robot link")
                name = str(link["name"])
                if name in link_by_name:
                    raise ValueError("IPC robot contains duplicate link names")
                link_by_name[name] = link
                link_path = str(link["path"])
                if not link_path.startswith(robot_path.rstrip("/") + "/"):
                    raise ValueError("IPC robot link path is outside its robot")
                if link_path in robot_link_paths:
                    raise ValueError("IPC manifest contains duplicate robot link paths")
                robot_link_paths.add(link_path)
                robot_links_by_path[link_path] = (robot_name, link)
                _validate_transform(link.get("transform"), "robot link transform")
                _validate_transform(
                    link.get("local_transform"), "robot link local transform"
                )
                _require_bool(link.get("has_inertial"), "robot link inertial flag")
                mass = _require_number(link.get("mass"), "robot link mass")
                if mass < 0.0:
                    raise ValueError("robot link mass must be non-negative")
                _require_vector(link.get("center_of_mass"), 3, "robot link center of mass")
                inertia = _require_vector(
                    link.get("inertia_diagonal"), 3, "robot link inertia diagonal"
                )
                if any(value < 0.0 for value in inertia):
                    raise ValueError("robot link inertia diagonal must be non-negative")
                _require_vector(
                    link.get("inertia_off_diagonal"),
                    3,
                    "robot link inertia off-diagonal",
                )
                orientation = _require_vector(
                    link.get("inertia_orientation_wxyz"),
                    4,
                    "robot link inertia orientation",
                )
                orientation_norm = math.sqrt(
                    sum(value * value for value in orientation)
                )
                if not math.isclose(
                    orientation_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5
                ):
                    raise ValueError("robot link inertia orientation must be normalized")
                _require_int(link.get("role"), "robot link role", minimum=0, maximum=1)
                collision_shapes = link.get("collision_shapes")
                if not isinstance(collision_shapes, list):
                    raise ValueError("IPC robot link collision shape table must be a list")
                _validate_unique_paths(collision_shapes, "robot collision shape")
                for raw_collision in collision_shapes:
                    collision = _require_mapping(
                        raw_collision, "robot collision shape"
                    )
                    collision_path = str(collision["path"])
                    if not collision_path.startswith(link_path.rstrip("/") + "/"):
                        raise ValueError(
                            "IPC robot collision shape path is outside its link"
                        )
                    if collision_path in robot_collision_paths:
                        raise ValueError(
                            "IPC manifest contains duplicate robot collision shape paths"
                        )
                    robot_collision_paths.add(collision_path)
                    _validate_transform(
                        collision.get("transform"),
                        "robot collision shape transform",
                    )
                    _validate_transform(
                        collision.get("link_transform"),
                        "robot collision shape link transform",
                    )
                    _require_bool(
                        collision.get("disabled"),
                        "robot collision shape disabled flag",
                    )
                    for field in ("collision_layer", "collision_mask"):
                        _require_int(
                            collision.get(field),
                            f"robot collision shape {field.replace('_', ' ')}",
                            minimum=0,
                            maximum=(1 << 32) - 1,
                        )
                    contact_offset = _require_number(
                        collision.get("contact_offset"),
                        "robot collision shape contact offset",
                    )
                    rest_offset = _require_number(
                        collision.get("rest_offset"),
                        "robot collision shape rest offset",
                    )
                    if contact_offset < 0.0 or rest_offset > contact_offset:
                        raise ValueError(
                            "robot collision shape offsets require contact_offset >= 0 "
                            "and rest_offset <= contact_offset"
                        )
                    material = _require_mapping(
                        collision.get("material"), "robot collision shape material"
                    )
                    for field in (
                        "sliding_friction",
                        "torsional_friction",
                        "rolling_friction",
                        "contact_compliance",
                        "contact_damping",
                    ):
                        if _require_number(material.get(field), f"physics material {field}") < 0.0:
                            raise ValueError(f"physics material {field} must be non-negative")
                    restitution = _require_number(
                        material.get("restitution"), "physics material restitution"
                    )
                    if not 0.0 <= restitution <= 1.0:
                        raise ValueError("physics material restitution must be in [0, 1]")

                    shape_type = collision.get("shape_type")
                    if shape_type == "box":
                        size = _require_vector(
                            collision.get("size"),
                            3,
                            "robot box collision shape size",
                        )
                        if any(value <= 0.0 for value in size):
                            raise ValueError(
                                "robot box collision shape size must be positive"
                            )
                    elif shape_type == "sphere":
                        if _require_number(
                            collision.get("radius"),
                            "robot sphere collision shape radius",
                        ) <= 0.0:
                            raise ValueError(
                                "robot sphere collision shape radius must be positive"
                            )
                    elif shape_type in ("capsule", "cylinder"):
                        for field in ("radius", "height"):
                            if _require_number(
                                collision.get(field),
                                f"robot {shape_type} collision shape {field}",
                            ) <= 0.0:
                                raise ValueError(
                                    f"robot {shape_type} collision shape {field} must be positive"
                                )
                    elif shape_type == "triangle_mesh":
                        mesh_blob = str(collision.get("mesh_blob", ""))
                        if mesh_blob not in decoded_surface_meshes:
                            raise ValueError(
                                "robot triangle collision shape references an unknown mesh blob"
                            )
                        decoded_surface = decoded_surface_meshes[mesh_blob]
                        for field in ("vertex_count", "triangle_count"):
                            if _require_int(
                                collision.get(field),
                                f"robot triangle collision shape {field}",
                                minimum=1,
                            ) != int(decoded_surface[field]):
                                raise ValueError(
                                    "robot triangle collision shape "
                                    f"{field} does not match its mesh blob"
                                )
                    else:
                        raise ValueError(
                            f"unsupported robot collision shape type {shape_type!r}"
                        )

            joint_names: set[str] = set()
            child_link_names: set[str] = set()
            parent_by_child: dict[str, str] = {}
            for raw_joint in joints:
                joint = _require_mapping(raw_joint, "robot joint")
                name = str(joint["name"])
                if name in joint_names:
                    raise ValueError("IPC robot contains duplicate joint names")
                joint_names.add(name)
                parent_link = str(joint.get("parent_link", ""))
                child_link = str(joint.get("child_link", ""))
                if parent_link not in link_by_name or child_link not in link_by_name:
                    raise ValueError("IPC robot joint references an unknown link")
                if child_link in child_link_names:
                    raise ValueError("IPC robot contains multiple joints for one child link")
                child_link_names.add(child_link)
                parent_by_child[child_link] = parent_link
                if joint.get("parent_link_path") != link_by_name[parent_link]["path"] or joint.get(
                    "child_link_path"
                ) != link_by_name[child_link]["path"]:
                    raise ValueError("IPC robot joint link paths do not match its topology")
                _validate_transform(joint.get("transform"), "robot joint transform")
                _validate_transform(
                    joint.get("local_transform"), "robot joint local transform"
                )
                axis = _require_vector(joint.get("axis"), 3, "robot joint axis")
                axis_norm = math.sqrt(sum(value * value for value in axis))
                if not math.isclose(axis_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
                    raise ValueError("robot joint axis must be normalized")
                _require_int(
                    joint.get("joint_type"), "robot joint type", minimum=0, maximum=5
                )
                _require_int(
                    joint.get("drive_mode"), "robot joint drive mode", minimum=0, maximum=3
                )
                nonnegative_fields = {
                    "effort_limit",
                    "velocity_limit",
                    "damping",
                    "armature",
                    "friction_loss",
                    "drive_stiffness",
                    "drive_damping",
                }
                numeric_fields = nonnegative_fields | {
                    "lower_limit",
                    "upper_limit",
                    "joint_position",
                    "initial_position",
                    "control_lower_limit",
                    "control_upper_limit",
                    "force_lower_limit",
                    "force_upper_limit",
                    "affine_actuator_control_gain",
                    "affine_actuator_force_offset",
                    "affine_actuator_position_gain",
                    "affine_actuator_velocity_gain",
                    "affine_actuator_inherit_range",
                }
                for field in numeric_fields:
                    value = _require_number(
                        joint.get(field), f"robot joint {field.replace('_', ' ')}"
                    )
                    if field in nonnegative_fields and value < 0.0:
                        raise ValueError(f"robot joint {field} must be non-negative")
                gear = joint.get("gear")
                if not isinstance(gear, list):
                    raise ValueError("robot joint gear must be a list")
                for value in gear:
                    _require_number(value, "robot joint gear entry")
                _require_bool(
                    joint.get("affine_actuator_enabled"),
                    "robot joint affine actuator flag",
                )

            expected_root_paths = sorted(
                str(link["path"])
                for name, link in link_by_name.items()
                if name not in child_link_names
            )
            if root_link_paths != expected_root_paths:
                raise ValueError("IPC robot root link paths do not match its topology")
            for link_name in link_by_name:
                visited: set[str] = set()
                current = link_name
                while current in parent_by_child:
                    if current in visited:
                        raise ValueError("IPC robot joint topology contains a cycle")
                    visited.add(current)
                    current = parent_by_child[current]

        coupling_paths: list[str] = []
        coupled_link_paths: set[str] = set()
        for expected_proxy_index, raw_coupling in enumerate(couplings):
            coupling = _require_mapping(raw_coupling, "physics coupling")
            coupling_path = coupling.get("coupling_path")
            link_path = coupling.get("link_path")
            robot_name = coupling.get("robot_name")
            link_name = coupling.get("link_name")
            if not isinstance(coupling_path, str) or not coupling_path:
                raise ValueError("PhysicsCoupling entry requires a non-empty coupling_path")
            if not isinstance(link_path, str) or not link_path:
                raise ValueError("PhysicsCoupling entry requires a non-empty link_path")
            if not isinstance(robot_name, str) or not robot_name:
                raise ValueError("PhysicsCoupling entry requires a non-empty robot_name")
            if not isinstance(link_name, str) or not link_name:
                raise ValueError("PhysicsCoupling entry requires a non-empty link_name")
            coupling_paths.append(coupling_path)
            if link_path in coupled_link_paths:
                raise ValueError(
                    "IPC manifest contains multiple PhysicsCoupling entries for one Link3D"
                )
            coupled_link_paths.add(link_path)
            if _require_int(
                coupling.get("proxy_index"),
                "PhysicsCoupling proxy index",
                minimum=0,
            ) != expected_proxy_index:
                raise ValueError(
                    "PhysicsCoupling proxy indices must be contiguous and match table order"
                )
            if coupling.get("mode") not in ("OneWay", "TwoWay"):
                raise ValueError("PhysicsCoupling mode must be OneWay or TwoWay")
            for field in ("force_scale", "torque_scale"):
                if _require_number(
                    coupling.get(field), f"PhysicsCoupling {field.replace('_', ' ')}"
                ) < 0.0:
                    raise ValueError(f"PhysicsCoupling {field} must be non-negative")
            target = robot_links_by_path.get(link_path)
            if target is None:
                raise ValueError("PhysicsCoupling references an unknown Robot3D Link3D")
            target_robot_name, target_link = target
            if target_robot_name != robot_name or str(target_link["name"]) != link_name:
                raise ValueError(
                    "PhysicsCoupling robot/link names do not match its canonical link path"
                )
            if not any(
                not bool(shape["disabled"])
                for shape in target_link["collision_shapes"]
            ):
                raise ValueError(
                    "PhysicsCoupling target Link3D has no enabled CollisionShape3D"
                )
        if len(set(coupling_paths)) != len(coupling_paths):
            raise ValueError("IPC manifest contains duplicate PhysicsCoupling paths")
        if coupling_paths != sorted(coupling_paths):
            raise ValueError("PhysicsCoupling table must be sorted by coupling_path")

        deformable_by_path = {
            str(body["path"]): body for body in deformables
        }
        coupling_by_link_path = {
            str(coupling["link_path"]): coupling for coupling in couplings
        }
        attachment_paths: list[str] = []
        attached_vertices: set[tuple[str, int]] = set()
        for raw_attachment in deformable_attachments:
            attachment = _require_mapping(
                raw_attachment, "deformable attachment"
            )
            attachment_path = attachment.get("attachment_path")
            body_path = attachment.get("deformable_body_path")
            link_path = attachment.get("rigid_link_path")
            if not isinstance(attachment_path, str) or not attachment_path:
                raise ValueError(
                    "DeformableAttachment3D entry requires a non-empty "
                    "attachment_path"
                )
            if not isinstance(body_path, str) or body_path not in deformable_by_path:
                raise ValueError(
                    "DeformableAttachment3D references an unknown deformable body"
                )
            coupling = coupling_by_link_path.get(str(link_path))
            if coupling is None:
                raise ValueError(
                    "DeformableAttachment3D references a rigid link without a "
                    "PhysicsCoupling"
                )
            if _require_int(
                attachment.get("proxy_index"),
                "DeformableAttachment3D proxy index",
                minimum=0,
            ) != int(coupling["proxy_index"]):
                raise ValueError(
                    "DeformableAttachment3D proxy index does not match its "
                    "PhysicsCoupling"
                )
            if _require_number(
                attachment.get("strength_rate"),
                "DeformableAttachment3D strength rate",
            ) <= 0.0:
                raise ValueError(
                    "DeformableAttachment3D strength rate must be positive"
                )
            raw_indices = attachment.get("vertex_indices")
            if not isinstance(raw_indices, list) or not raw_indices:
                raise ValueError(
                    "DeformableAttachment3D vertex_indices must be a non-empty list"
                )
            indices = tuple(
                _require_int(
                    value,
                    "DeformableAttachment3D vertex index",
                    minimum=0,
                )
                for value in raw_indices
            )
            vertex_count = int(deformable_by_path[body_path]["vertex_count"])
            if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
                raise ValueError(
                    "DeformableAttachment3D vertex indices must be sorted and unique"
                )
            if any(index >= vertex_count for index in indices):
                raise ValueError(
                    "DeformableAttachment3D contains an out-of-range vertex index"
                )
            for index in indices:
                key = (body_path, index)
                if key in attached_vertices:
                    raise ValueError(
                        "multiple DeformableAttachment3D entries target the same vertex"
                    )
                attached_vertices.add(key)
            attachment_paths.append(attachment_path)
        if len(set(attachment_paths)) != len(attachment_paths):
            raise ValueError(
                "IPC manifest contains duplicate DeformableAttachment3D paths"
            )
        if attachment_paths != sorted(attachment_paths):
            raise ValueError(
                "DeformableAttachment3D table must be sorted by attachment_path"
            )

        for sensor_path, link_path in sensor_attachment_paths:
            if link_path not in robot_link_paths:
                raise ValueError(
                    "tactile sensor attachment references an unknown robot link"
                )
            if not sensor_path.startswith(link_path.rstrip("/") + "/"):
                raise ValueError("tactile sensor attachment path is outside its link")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "producer_version", producer_version)
        object.__setattr__(self, "format", artifact_format)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "blobs", MappingProxyType(supplied_blobs))
        frozen_manifest_data = _freeze_json(manifest_data)
        object.__setattr__(self, "_manifest_data", frozen_manifest_data)
        object.__setattr__(self, "_blob_table", frozen_manifest_data["blobs"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledIpcSceneArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("compiled IPC scene artifact must be a mapping")
        return cls(
            schema_version=value.get("schema_version", 0),
            producer=str(value.get("producer", "")),
            producer_version=str(value.get("producer_version", "")),
            format=str(value.get("format", "")),
            manifest=str(value.get("manifest", "")),
            manifest_sha256=str(value.get("manifest_sha256", "")),
            blobs=value.get("blobs", ()),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        entries = []
        for raw_entry in self._blob_table:
            entry = dict(raw_entry)
            entry["data"] = self.blobs[str(entry["id"])]
            entries.append(entry)
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "format": self.format,
            "manifest": self.manifest,
            "manifest_sha256": self.manifest_sha256,
            "blobs": entries,
        }

    @property
    def digest(self) -> str:
        return self.manifest_sha256

    @property
    def manifest_data(self) -> Mapping[str, Any]:
        return self._manifest_data

    @property
    def deformable_bodies(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._manifest_data["deformable_bodies"])

    @property
    def tactile_sensors(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._manifest_data["tactile_sensors"])

    @property
    def robots(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._manifest_data["robots"])

    @property
    def static_colliders(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._manifest_data.get("static_colliders", ()))

    @property
    def couplings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._manifest_data["couplings"])

    @property
    def deformable_attachments(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._manifest_data.get("deformable_attachments", ()))


def validate_ipc_artifact(
    artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
) -> CompiledIpcSceneArtifact:
    if isinstance(artifact, CompiledIpcSceneArtifact):
        return CompiledIpcSceneArtifact.from_mapping(artifact.to_mapping())
    return CompiledIpcSceneArtifact.from_mapping(artifact)


__all__ = ["CompiledIpcSceneArtifact", "validate_ipc_artifact"]
