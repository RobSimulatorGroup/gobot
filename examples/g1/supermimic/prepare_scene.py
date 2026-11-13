"""Import the official model into Gobot authoring data and require a static audit."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import gobot
import mujoco

from .model_audit import audit_model
from .reference import LOCK, sha256


def prepare(source, project):
    source, project = Path(source).resolve(), Path(project).resolve()
    profile = json.loads((source / "profile.json").read_text())
    lock = json.loads(LOCK.read_text())
    if (
        profile.get("protocol") != "gobot_supermimic_official_model_v1"
        or profile["reference_commit"] != lock["supermimic"]["commit"]
        or profile["mjlab_commit"] != lock["mjlab_research"]["commit"]
    ):
        raise ValueError("export does not match the official baseline")
    if profile["versions"]["mujoco"] != mujoco.__version__:
        raise ValueError("model export and import must use the same MuJoCo version")
    if project.exists():
        raise FileExistsError("use a new project directory")
    for name, expected in profile["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != expected:
            raise ValueError(f"export changed: {name}")
    target = project / "assets/supermimic"
    target.mkdir(parents=True)
    for name in profile["files"]:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    (target / "robot_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    gobot.set_project_path(str(project))
    gobot.import_mjcf_scene(
        str(target / "world.xml"), "res://g1_29dof_flat_scene.jscn", name="g1_29dof"
    )
    ctx = gobot.app.create_context()
    try:
        ctx.set_project_path(str(project))
        ctx.load_scene("res://g1_29dof_flat_scene.jscn")
        ctx.fixed_time_step = profile["physics_dt"]
        ctx.set_mujoco_solver_settings(profile["solver_settings"])
        artifact = ctx.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        report = audit_model(
            mujoco.MjModel.from_xml_path(str(target / "world.xml")),
            artifact,
            "g1_29dof_",
            compare_options=True,
        )
        report.update(
            protocol=profile["protocol"],
            profile_sha256=sha256(target / "robot_profile.json"),
        )
        (target / "world_physics_audit.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not report["passed"]:
            raise RuntimeError(
                f"official G1 model audit failed: {report['failed_checks']}"
            )
        print(
            json.dumps(
                {
                    "project": str(project),
                    "passed": True,
                    "checks": len(report["checks"]),
                }
            )
        )
    finally:
        ctx.clear_scene()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-export", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    prepare(a.reference_export, a.project)
