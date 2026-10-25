"""Authoring geometry for the hollow torus press head (metres)."""

from __future__ import annotations
import math

MAJOR_RADIUS = 0.005
TUBE_RADIUS = 0.002
START_HEIGHT = 0.004
PRESS_HEIGHT = 0.001


def vector(value):
    return {"matrix_data": {"rows": len(value), "cols": 1, "storage": list(value)}}


def surface(vertices, indices, normals=()):
    return dict(
        vertices=[vector(v) for v in vertices],
        indices=indices,
        normals=[vector(n) for n in normals],
        tangents=[],
        uv0=[],
        colors=[],
        material=None,
    )


def torus_surface(segments=64, tube_segments=24):
    vertices, normals, indices = [], [], []
    for i in range(segments):
        u = 2 * math.pi * i / segments
        for j in range(tube_segments):
            v = 2 * math.pi * j / tube_segments
            n = (math.cos(u) * math.cos(v), math.sin(u) * math.cos(v), math.sin(v))
            vertices.append(
                (
                    (MAJOR_RADIUS + TUBE_RADIUS * math.cos(v)) * math.cos(u),
                    (MAJOR_RADIUS + TUBE_RADIUS * math.cos(v)) * math.sin(u),
                    TUBE_RADIUS * math.sin(v),
                )
            )
            normals.append(n)
            a = i * tube_segments + j
            b = ((i + 1) % segments) * tube_segments + j
            c = ((i + 1) % segments) * tube_segments + (j + 1) % tube_segments
            d = i * tube_segments + (j + 1) % tube_segments
            indices.extend((a, b, c, a, c, d))
    return surface(vertices, indices, normals)


def replace_press(scene):
    resources = [
        r for r in scene["__SUB_RESOURCES__"] if not r["__ID__"].startswith("press_")
    ]
    resources.append(
        dict(__ID__="press_visual", __TYPE__="ArrayMesh", surfaces=[torus_surface()])
    )
    resources.append(
        dict(
            __ID__="press_material",
            __TYPE__="PhysicsMaterial3D",
            sliding_friction=1.0,
            torsional_friction=0.0,
            rolling_friction=0.0,
        )
    )
    # This example's press is the last authored body; remove its old descendants.
    nodes = scene["__NODES__"]
    press_index = next(i for i, n in enumerate(nodes) if n["name"] == "PressHead")
    nodes = nodes[: press_index + 1]
    nodes[press_index]["properties"]["position"] = vector((0.0, 0.0, START_HEIGHT))
    mass = nodes[press_index]["properties"]["mass"]
    inertia_xy = mass * (0.5 * MAJOR_RADIUS**2 + 0.625 * TUBE_RADIUS**2)
    inertia_z = mass * (MAJOR_RADIUS**2 + 0.75 * TUBE_RADIUS**2)
    nodes[press_index]["properties"]["inertia_diagonal"] = vector(
        (inertia_xy, inertia_xy, inertia_z)
    )
    nodes.append(
        dict(
            type="MeshInstance3D",
            name="Torus",
            parent=press_index,
            properties=dict(
                mesh="SubResource(press_visual)",
                surface_color=dict(red=0.92, green=0.45, blue=0.12, alpha=1.0),
            ),
        )
    )
    resources.append(
        dict(
            __ID__="press_shape",
            __TYPE__="TriangleMeshShape3D",
            mesh="SubResource(press_visual)",
        )
    )
    nodes.append(
        dict(
            type="CollisionShape3D",
            name="Collision",
            parent=press_index,
            properties=dict(
                shape="SubResource(press_shape)",
                physics_material="SubResource(press_material)",
                visible=False,
            ),
        )
    )
    scene["__SUB_RESOURCES__"] = resources
    scene["__NODES__"] = nodes
