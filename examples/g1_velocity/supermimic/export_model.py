"""Export the effective official GenCoord G1 model for scene authoring.

Run in a separate reference Python with pinned supermimic and mjlab_research on
PYTHONPATH. It never imports Gobot; the exported model/profile cross the boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import types
import xml.etree.ElementTree as ET

from .reference import LOCK, sha256, verify_checkout


def export(output):
    import mujoco
    import mjlab
    import supermimic
    import numpy as np
    import torch

    # Avoid unrelated PPO task registration; use the actual factory directly.
    for name, root in (
        ("mjlab.tasks", Path(mjlab.__file__).parent / "tasks"),
        (
            "mjlab.tasks.tracking.config.g1",
            Path(mjlab.__file__).parent / "tasks/tracking/config/g1",
        ),
        ("supermimic.tasks", Path(supermimic.__file__).parent / "tasks"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    from supermimic.tasks.env_cfgs import gencoord_env_cfg
    from mjlab.scene import Scene
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE

    lock = json.loads(LOCK.read_text())
    reference = Path(supermimic.__file__).resolve().parents[2]
    companion = Path(mjlab.__file__).resolve().parents[2]
    verify_checkout(reference, lock["supermimic"]["commit"])
    verify_checkout(companion, lock["mjlab_research"]["commit"])
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"use a new export directory: {output}")
    cfg = gencoord_env_cfg()
    scene = Scene(cfg.scene, device="cpu")
    compiled = scene.compile()
    cfg.sim.mujoco.apply(compiled)
    for name in (
        "timestep",
        "integrator",
        "gravity",
        "impratio",
        "cone",
        "jacobian",
        "solver",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
        "disableflags",
        "enableflags",
    ):
        setattr(scene.spec.option, name, getattr(compiled.opt, name))
    scene.write(output)
    tree = ET.parse(output / "scene.xml")
    for node in tree.iter():
        for key, value in list(node.attrib.items()):
            if key not in ("file", "meshdir", "texturedir") and value.startswith(
                "robot/"
            ):
                node.set(
                    key,
                    value.replace("robot/", "robot_", 1)
                    if key in ("class", "childclass")
                    else value.removeprefix("robot/"),
                )
    tree.getroot().set("model", "g1_29dof")
    tree.write(output / "world.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output / "world.xml"))
    checks = {}
    for name in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_priority",
        "geom_condim",
        "geom_contype",
        "geom_conaffinity",
        "geom_solref",
        "geom_solimp",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    ):
        before, after = (
            np.asarray(getattr(compiled, name)),
            np.asarray(getattr(model, name)),
        )
        if before.shape != after.shape or not np.allclose(
            before, after, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(f"export changes effective model: {name}")
        checks[name] = float(np.max(np.abs(before - after))) if before.size else 0.0
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("unexpected G1 topology")
    aids = []
    scales = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == jid)
        if len(matches) != 1:
            raise ValueError(f"ambiguous actuator {name}")
        aids.append(int(matches[0]))
        values = [
            v for pattern, v in G1_ACTION_SCALE.items() if re.fullmatch(pattern, name)
        ]
        if len(values) != 1:
            raise ValueError(f"ambiguous action scale {name}")
        scales.append(values[0])
    com_link = cfg.events["base_com"].params["asset_cfg"].body_names[0]
    com = model.body(com_link).ipos.tolist()
    sensor = model.sensor("imu_ang_vel")
    site = model.site(int(model.sensor_objid[sensor.id]))
    command = cfg.commands["motion"]
    profile = {
        "schema_version": 3,
        "protocol": "gobot_supermimic_official_model_v1",
        "reference_commit": lock["supermimic"]["commit"],
        "mjlab_commit": lock["mjlab_research"]["commit"],
        "robot_name": "g1_29dof",
        "base_link": "pelvis",
        "anchor_link": command.anchor_body_name,
        "joint_names": names,
        "tracked_links": command.body_names,
        "foot_shapes": [
            model.geom(i).name
            for i in range(model.ngeom)
            if re.fullmatch(
                r"(left|right)_foot[1-7]_collision", model.geom(i).name or ""
            )
        ],
        "default_joint_position": model.key("init_state").qpos[7:].tolist(),
        "kp": model.actuator_gainprm[aids, 0].tolist(),
        "kd": (-model.actuator_biasprm[aids, 2]).tolist(),
        "effort_limit": model.actuator_forcerange[aids, 1].tolist(),
        "action_scale": scales,
        "physics_dt": cfg.sim.mujoco.timestep,
        "decimation": cfg.decimation,
        "soft_joint_limit_factor": cfg.scene.entities[
            "robot"
        ].articulation.soft_joint_pos_limit_factor,
        "height_threshold": cfg.terminations["anchor_pos"].params["threshold"],
        "tilt_threshold": cfg.terminations["anchor_ori"].params["threshold"],
        "com_link": com_link,
        "com_nominal": com,
        "context_mean": [0.75, 0.0, 0.0, 0.0] + [0.0] * 29,
        "context_scale": [0.45, 0.025, 0.05, 0.05] + [0.01] * 29,
        "context_order": [
            "foot_friction",
            "base_com_x",
            "base_com_y",
            "base_com_z",
            *["encoder_bias/" + n for n in names],
        ],
        "imu": {
            "body": model.body(int(model.site_bodyid[site.id])).name,
            "quaternion": site.quat.tolist(),
        },
        "dr_modes": {
            name: cfg.events[name].mode
            for name in ("base_com", "encoder_bias", "foot_friction")
        },
        "solver_settings": {
            "solver": "Newton",
            "integrator": "ImplicitFast",
            "cone": "Pyramidal",
            "iterations": model.opt.iterations,
            "line_search_iterations": model.opt.ls_iterations,
            "convex_collision_iterations": model.opt.ccd_iterations,
            "tolerance": model.opt.tolerance,
            "line_search_tolerance": model.opt.ls_tolerance,
            "convex_collision_tolerance": model.opt.ccd_tolerance,
            "impedance_ratio": model.opt.impratio,
        },
        "versions": {
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "mujoco-warp": importlib.metadata.version("mujoco-warp"),
            "warp-lang": importlib.metadata.version("warp-lang"),
        },
        "nominal_evaluation_foot_friction": 1.0,
        "simulation_config": asdict(cfg.sim.mujoco),
        "export_max_abs_error": checks,
    }
    profile["files"] = {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "joint_count": len(names),
                "anchor": profile["anchor_link"],
                "versions": profile["versions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args().output)
