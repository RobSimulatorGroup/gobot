"""Import the exported mjlab scene into Gobot, then audit compiled physics.

MJCF is an offline import only. The training scene remains the resulting .jscn.
"""

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gobot
from gobot.rl.tracking.config import G1_JOINTS, TrackingConfig
from gobot.rl.training.workflow import config_dict
from gobot.sim.providers.base import CompiledSceneArtifact

SOLVER = dict(
    solver="Newton",
    integrator="ImplicitFast",
    cone="Pyramidal",
    jacobian="Auto",
    iterations=10,
    line_search_iterations=20,
    convex_collision_iterations=50,
    impedance_ratio=1.0,
    tolerance=1e-8,
    line_search_tolerance=0.01,
)


def model_audit(reference, compiled):
    def name(m, kind, i):
        n = getattr(m, kind)(i).name
        return n.removeprefix("robot/").removeprefix("g1_")

    result = {}
    errors = []

    def check(label, a, b, atol=2e-6, rtol=2e-5):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        equal = a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)
        result[label] = dict(
            passed=bool(equal),
            max_abs=float(np.max(np.abs(a - b)))
            if a.shape == b.shape and a.size
            else 0.0,
        )
        if not equal:
            errors.append(label)

    for kind, count, fields in [
        ("body", "nbody", ("mass", "pos", "quat", "ipos", "iquat", "inertia")),
        (
            "joint",
            "njnt",
            ("type", "axis", "pos", "range", "limited", "stiffness", "margin"),
        ),
    ]:
        old = {
            name(reference, kind, i): i
            for i in range(getattr(reference, count))
            if name(reference, kind, i)
        }
        new = {
            name(compiled, kind, i): i
            for i in range(getattr(compiled, count))
            if name(compiled, kind, i)
        }
        # world geometry and free-joint naming are authored wrapper details.
        if kind == "joint":
            old = {
                n: i
                for n, i in old.items()
                if reference.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
            }
        for n, i in old.items():
            if n == "world":
                continue
            if n not in new:
                errors.append(f"missing {kind}:{n}")
                continue
            prefix = "jnt_" if kind == "joint" else "body_"
            for field in fields:
                a = getattr(reference, prefix + field)[i]
                b = getattr(compiled, prefix + field)[new[n]]
                if field in ("quat", "iquat") and np.dot(a, b) < 0:
                    b = -b
                check(f"{kind}/{n}/{field}", a, b)
    old = {
        name(reference, "geom", i): i
        for i in range(reference.ngeom)
        if reference.geom_contype[i] or reference.geom_conaffinity[i]
    }
    new = {
        name(compiled, "geom", i): i
        for i in range(compiled.ngeom)
        if compiled.geom_contype[i] or compiled.geom_conaffinity[i]
    }
    for n, i in old.items():
        if n not in new:
            errors.append(f"missing geom:{n}")
            continue
        for field in (
            "type",
            "pos",
            "quat",
            "size",
            "friction",
            "solref",
            "solimp",
            "margin",
            "gap",
            "condim",
            "priority",
            "contype",
            "conaffinity",
        ):
            a = getattr(reference, "geom_" + field)[i]
            b = getattr(compiled, "geom_" + field)[new[n]]
            if field == "quat" and np.dot(a, b) < 0:
                b = -b
            if (
                field == "size"
                and reference.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
            ):
                continue
            check(f"geom/{n}/{field}", a, b)
    for n in G1_JOINTS:
        ri = reference.joint("robot/" + n).id
        ci = compiled.joint("g1_" + n).id
        for field in ("armature", "damping", "frictionloss"):
            check(
                f"joint/{n}/{field}",
                getattr(reference, "dof_" + field)[reference.jnt_dofadr[ri]],
                getattr(compiled, "dof_" + field)[compiled.jnt_dofadr[ci]],
            )
        ra = int(np.flatnonzero(reference.actuator_trnid[:, 0] == ri)[0])
        ca = int(np.flatnonzero(compiled.actuator_trnid[:, 0] == ci)[0])
        for field in (
            "gainprm",
            "biasprm",
            "forcerange",
            "forcelimited",
            "ctrlrange",
            "ctrllimited",
            "gear",
        ):
            if (
                field == "ctrlrange"
                and not reference.actuator_ctrllimited[ra]
                and not compiled.actuator_ctrllimited[ca]
            ):
                continue
            check(
                f"actuator/{n}/{field}",
                getattr(reference, "actuator_" + field)[ra],
                getattr(compiled, "actuator_" + field)[ca],
            )

    def exclusions(m):
        return sorted(
            tuple(
                sorted(
                    (name(m, "body", int(v) // 65536), name(m, "body", int(v) % 65536))
                )
            )
            for v in m.exclude_signature
        )

    result["contact_exclusions"] = dict(
        passed=exclusions(reference) == exclusions(compiled),
        reference=exclusions(reference),
        gobot=exclusions(compiled),
    )
    if not result["contact_exclusions"]["passed"]:
        errors.append("contact_exclusions")
    return dict(passed=not errors, errors=errors, checks=result)


def prepare(reference, output):
    reference = Path(reference).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(reference / "scene/assets", output / "assets")
    tree = ET.parse(reference / "scene/scene.xml")
    # Prefixes identify the upstream entity, not different robot semantics.
    for element in tree.iter():
        for key, value in list(element.attrib.items()):
            if key not in ("file",):
                element.set(
                    key,
                    value.replace(
                        "robot/", "g1_" if key in ("class", "childclass") else ""
                    ),
                )
    sensors = tree.getroot().find("sensor")
    if sensors is not None:
        for sensor in list(sensors):
            if sensor.tag == "contact":
                sensors.remove(
                    sensor
                )  # Runtime contact observation, not authored robot data.
    path = output / "source.xml"
    tree.write(path, encoding="unicode")
    gobot.set_project_path(str(output))
    gobot.import_mjcf_scene(str(path), "res://world.jscn", name="g1")
    (output / "project.gobot").write_text(
        json.dumps({"main_scene": "res://world.jscn"})
    )
    shutil.copy2(reference / "motion.npz", output / "motion.npz")
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(output))
        context.load_scene("res://world.jscn")
        nominal = mujoco.MjModel.from_xml_path(str(reference / "scene/scene.xml"))
        initial = nominal.key("init_state").qpos
        from gobot.rl.locomotion.math import find_node_by_name

        for name in G1_JOINTS:
            index = nominal.joint("robot/" + name).id
            find_node_by_name(context.root, name).initial_position = float(
                initial[nominal.jnt_qposadr[index]]
            )
        gobot.save_scene(context.root, str(output / "world.jscn"))
        context.fixed_time_step = 0.005
        context.set_mujoco_solver_settings(SOLVER)
        artifact = CompiledSceneArtifact.from_mapping(
            context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        )
        compiled = mujoco.MjModel.from_xml_string(artifact.content)
        nominal = mujoco.MjModel.from_xml_path(str(reference / "scene/scene.xml"))
        audit = model_audit(nominal, compiled)
        audit["compiled_artifact_digest"] = artifact.content_digest
        audit["motion_sha256"] = (
            __import__("hashlib")
            .sha256((output / "motion.npz").read_bytes())
            .hexdigest()
        )
        for name in G1_JOINTS:
            expected = float(
                initial[nominal.jnt_qposadr[nominal.joint("robot/" + name).id]]
            )
            actual = float(find_node_by_name(context.root, name).initial_position)
            ok = bool(np.isclose(expected, actual, atol=1e-7, rtol=1e-6))
            audit["checks"][f"initial/{name}"] = dict(
                passed=ok, max_abs=abs(expected - actual)
            )
            if not ok:
                audit["errors"].append(f"initial/{name}")
        audit["passed"] = not audit["errors"]
        (output / "compiled.xml").write_text(artifact.content)
        (output / "physics-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        cfg = TrackingConfig(
            project_path=output, solver_settings=SOLVER, nconmax=128, njmax=500
        )
        (output / "task.json").write_text(json.dumps(config_dict(cfg), indent=2) + "\n")
        print(
            json.dumps(dict(passed=audit["passed"], errors=audit["errors"]), indent=2)
        )
    finally:
        context.clear_scene()
    if not audit["passed"]:
        raise RuntimeError("physics audit failed; do not train this scene")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(**vars(p.parse_args()))
