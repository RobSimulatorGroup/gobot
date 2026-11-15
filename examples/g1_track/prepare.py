"""Install one exported tracking policy and its prepared motion project locally."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from gobot.rl.policy import scene_bundle_digest
from gobot.rl.tracking.play import prepare as prepare_player


HERE = Path(__file__).resolve().parent


def prepare(checkpoint, project_path, *, output=HERE, device="cuda:0"):
    checkpoint, source, output = map(lambda path: Path(path).resolve(),
                                     (checkpoint, project_path, output))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    deployment = payload.get("infos", {}).get("deployment", {})
    if deployment.get("schema") != "gobot_tracking_deployment_v1":
        raise ValueError("Export on the training host with python -m gobot.rl.tracking.export first")
    cfg = json.loads(checkpoint.with_name("config.json").read_text())
    if scene_bundle_digest(source, cfg["scene_path"]) != deployment["scene_bundle_digest"]:
        raise ValueError("source scene does not match the exported policy")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "assets").exists():
        raise FileExistsError(f"{output / 'assets'} already exists; choose a fresh output directory")
    with tempfile.TemporaryDirectory(prefix="gobot-g1-track-") as temporary:
        stage = Path(temporary)
        assets = stage / "assets"
        shutil.copytree(source, assets / "project", ignore=shutil.ignore_patterns("__pycache__", "logs"))
        policy = assets / "policy"
        policy.mkdir()
        shutil.copy2(checkpoint, policy / "policy.pt")
        shutil.copy2(checkpoint.with_name("config.json"), policy / "config.json")
        player = prepare_player(policy / "policy.pt", stage / "player", assets / "project", device)
        scene = json.loads((player / "world.jscn").read_text())
        for resource in scene["__EXT_RESOURCES__"]:
            path = resource["__PATH__"]
            if Path(path).is_absolute():
                relative = Path(path).relative_to(assets / "project")
                resource["__PATH__"] = "res://assets/project/" + relative.as_posix()
        (player / "world.jscn").write_text(json.dumps(scene, indent=2) + "\n")
        settings = json.loads((player / "play.json").read_text())
        settings["runtime"].update(checkpoint="res://assets/policy/policy.pt", project_path="res://assets/project")
        (player / "play.json").write_text(json.dumps(settings, indent=2) + "\n")
        project = json.loads((player / "project.gobot").read_text())
        project["project_load_hook"] = "res://prepare.py"
        (player / "project.gobot").write_text(json.dumps(project, indent=2) + "\n")
        shutil.move(assets, output / "assets")
        for name in ("world.jscn", "play.json", "play.py", "project.gobot"):
            shutil.copy2(player / name, output / name)
        for name in ("prepare.py", "README.md"):
            if (HERE / name).resolve() != (output / name).resolve():
                shutil.copy2(HERE / name, output / name)
    return output


def main():
    if os.environ.get("GOBOT_PROJECT_HOOK") == "1":
        for name in ("assets/policy/policy.pt", "assets/policy/config.json", "assets/project/world.jscn", "play.json"):
            if not (HERE / name).is_file():
                raise FileNotFoundError(
                    "Prepare G1 tracking first: python examples/g1_track/prepare.py "
                    "<deployment/policy.pt> --project-path <prepared-motion-project>")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--project-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--device", default="cuda:0")
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
