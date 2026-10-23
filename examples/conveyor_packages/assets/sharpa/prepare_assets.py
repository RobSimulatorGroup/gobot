"""Derive a matched arm/hand pair from the pinned play2perfect URDF.

Run with the repository's Python environment. Reflections are baked into mesh
vertices, joint frames/axes and inertia tensors, never negative scene scales.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source/iiwa14_left_sharpa_adjusted_restricted.urdf"
REFLECTION = np.diag((1., -1., 1.))


def numbers(text):
    return np.fromstring(text, sep=" ")


def encoded(values):
    return " ".join(f"{v:.12g}" for v in values)


def collision_hull(points):
    for _ in range(40):
        hull = trimesh.convex.convex_hull(points)
        t = hull.triangles
        lengths = np.linalg.norm(t[:, [1, 2, 0]]-t, axis=2)
        area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)
        bad = area <= np.finfo(np.float32).eps*1024*lengths.max(1)**2
        if not bad.any():
            return hull
        # Remove the nearly collinear vertex opposite each longest edge, then
        # rebuild the closed hull. Dropping triangles alone would leave holes.
        remove = hull.faces[bad, (lengths[bad].argmax(1)+2)%3]
        points = np.delete(hull.vertices, np.unique(remove), axis=0)
    raise ValueError("unable to remove degenerate collision facets")


def prepare():
    original = ET.parse(SOURCE).getroot()
    trimesh.creation.cylinder(radius=.03, height=.055, sections=32).export(HERE / "wrist_adapter.obj")
    for side in ("left", "right"):
        root = deepcopy(original)
        root.set("name", f"sharpa_{side}")
        # Fold the empty, massless flange frame into the following fixed joint.
        # It is a coordinate frame, not an extra dynamic/kinematic body.
        flange = root.find("link[@name='iiwa14_link_ee']")
        flange_joint = root.find("joint[@name='iiwa14_joint_ee']")
        mount_joint = root.find("joint[@name='iiwa14_sharpa']")
        mount_joint.find("parent").set("link", "iiwa14_link_7")
        mount_joint.find("origin").set("xyz", flange_joint.find("origin").get("xyz"))
        root.remove(flange)
        root.remove(flange_joint)
        # The upstream mount is an empty 50 mm offset. Give the existing
        # 50 g adapter a visible sleeve and matching collision geometry.
        mount = root.find("link[@name='sharpa_mount']")
        for tag in ("visual", "collision"):
            sleeve = ET.SubElement(mount, tag)
            ET.SubElement(sleeve, "origin", xyz="0 0 0.025", rpy="0 0 0")
            ET.SubElement(ET.SubElement(sleeve, "geometry"), "cylinder", radius="0.03", length="0.055")
        mirrored = side == "right"
        collision_meshes = set(root.findall("./link/collision/geometry/mesh"))
        for element in root.iter():
            for key in ("name", "link"):
                if key in element.attrib:
                    element.set(key, element.get(key).replace("left_", f"{side}_"))
            if mirrored and element.tag == "origin":
                element.set("xyz", encoded(REFLECTION @ numbers(element.get("xyz", "0 0 0"))))
                rotation = Rotation.from_euler("xyz", numbers(element.get("rpy", "0 0 0"))).as_matrix()
                element.set("rpy", encoded(Rotation.from_matrix(REFLECTION @ rotation @ REFLECTION).as_euler("xyz")))
            elif mirrored and element.tag == "axis":
                # Angular velocity is an axial vector under a reflection.
                element.set("xyz", encoded(-REFLECTION @ numbers(element.get("xyz"))))
            elif mirrored and element.tag == "inertia":
                for name in ("ixy", "iyz"):
                    element.set(name, str(-float(element.get(name))))
            elif element.tag == "mesh":
                relative = Path(element.get("filename"))
                if element in collision_meshes:
                    destination = Path("collision") / side / relative.with_suffix(".obj")
                    target = HERE / destination
                    target.parent.mkdir(parents=True, exist_ok=True)
                    mesh = trimesh.load(SOURCE.parent / relative, force="mesh")
                    # URDF imports these as convex shapes. Weld sub-10-micron
                    # CAD slivers before constructing a closed collision hull.
                    points = np.unique(np.round(mesh.vertices, 5), axis=0)
                    if mirrored:
                        points = points @ REFLECTION
                    hull = collision_hull(points)
                    hull.export(target)
                    element.set("filename", str(destination))
                elif mirrored:
                    target = HERE / "mirrored" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        mesh = trimesh.load(SOURCE.parent / relative, force="mesh")
                        transform = np.eye(4); transform[:3, :3] = REFLECTION
                        mesh.apply_transform(transform)
                        mesh.export(target)
                    element.set("filename", str(Path("mirrored") / relative))
                else:
                    element.set("filename", str(Path("source") / relative))
        # The source green identifies tactile sensors; use a rubber finish in
        # the workcell while retaining its actual sensor/pad geometry.
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                material = visual.find("material")
                if material is None:
                    material = ET.SubElement(visual, "material", name="")
                color = material.find("color")
                if color is None:
                    color = ET.SubElement(material, "color")
                color.set("rgba", "0.065 0.075 0.085 1" if "elastomer" in link.get("name") or link.get("name") == "sharpa_mount" else
                          ("0.70 0.73 0.76 1" if "iiwa" not in link.get("name") else "0.26 0.32 0.38 1"))
        # Gobot's URDF visual import consumes meshes; retain the analytic
        # cylinder for collision and supply its triangulated visual equivalent.
        geometry = mount.find("visual/geometry")
        geometry.clear()
        ET.SubElement(geometry, "mesh", filename="wrist_adapter.obj")
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(HERE / f"{side}_arm.urdf", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    prepare()
