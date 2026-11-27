"""Install one exported tracking policy and its prepared motion project locally."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
DEPLOYMENT_URL = ("https://github.com/RobSimulatorGroup/gobot-example-assets/releases/download/"
                  "g1-track-dance2-v1/g1-track-deployment.zip")
DEPLOYMENT_SHA256 = "01fef329f4d31707d6f0ad9f1740771b62444223fb6b7878924f9fac0a28321d"
PROJECT_FILES = ("world.jscn", "play.json", "play.py", "project.gobot", "prepare.py", "README.md")


def report_progress(current, total, message):
    if os.environ.get("GOBOT_PROJECT_HOOK") == "1":
        print("GOBOT_PROGRESS " + json.dumps(dict(current=current, total=total, message=message)),
              flush=True)


def ensure_assets(project, url, sha256=None):
    """Install just the assets referenced by the shipped G1 presentation."""
    scene = json.loads((project / "world.jscn").read_text())
    meshes = [r["__PATH__"].removeprefix("res://") for r in scene["__EXT_RESOURCES__"]
              if r["__PATH__"].startswith("res://assets/")]
    required = {"assets/policy/policy.pt", "assets/policy/config.json",
                "assets/project/world.jscn", *meshes}
    assets = project / "assets"
    if assets.exists():
        config = json.loads((assets / "policy/config.json").read_text())
        motion = assets / "project" / config["motion_file"]
        if not motion.is_file():
            raise FileNotFoundError(motion)
        for name in required:
            if not (project / name).is_file():
                raise FileNotFoundError(project / name)
        return
    if not url:
        raise FileNotFoundError("G1 Track assets are missing. Set GOBOT_G1_TRACK_URL to the "
                                "deployment ZIP URL and reopen the project.")

    # Extract only the files used by the packaged presentation. Publish the
    # directory after all members have been read and their ZIP CRCs checked.
    required.update(("assets/project/motion.npz", "assets/project/project.gobot"))
    with tempfile.TemporaryDirectory(dir=project, prefix=".g1-download-") as tmp:
        stage = Path(tmp)
        archive = stage / "deployment.zip"
        print("Downloading G1 tracking deployment...", flush=True)
        digest = hashlib.sha256()
        report_progress(0, 0, "Connecting to G1 tracking deployment")
        with urlopen(url, timeout=60) as response, archive.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                received += len(block)
                report_progress(received, total, "Downloading G1 tracking deployment")
        report_progress(received, received, "Verifying G1 tracking deployment")
        if sha256 and digest.hexdigest() != sha256.lower():
            raise ValueError("G1 tracking deployment SHA-256 mismatch")
        with ZipFile(archive) as bundle:
            unpacked = 0
            unpack_size = sum(bundle.getinfo(name).file_size for name in required)
            for name in sorted(required):
                target = stage / name
                if not target.resolve().is_relative_to(stage / "assets"):
                    raise ValueError(f"Invalid project asset path: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                unpacked += target.stat().st_size
                report_progress(unpacked, unpack_size, "Installing G1 tracking assets")
        config = json.loads((stage / "assets/policy/config.json").read_text())
        if config["motion_file"] != "motion.npz" or config["scene_path"] != "res://world.jscn":
            raise ValueError("Deployment does not match the packaged G1 tracking project")
        try:
            (stage / "assets").rename(assets)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            ensure_assets(project, None)


def prepare(checkpoint, project_path, *, output=HERE, device="cuda:0"):
    import torch
    from gobot.rl.policy import scene_bundle_digest
    from gobot.rl.tracking.play import prepare as prepare_player

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
    url = os.environ.get("GOBOT_G1_TRACK_URL", DEPLOYMENT_URL)
    sha256 = os.environ.get("GOBOT_G1_TRACK_SHA256", DEPLOYMENT_SHA256)
    if os.environ.get("GOBOT_PROJECT_HOOK") == "1":
        ensure_assets(HERE, url, sha256)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--project-path", type=Path)
    parser.add_argument("--download-url", default=url)
    parser.add_argument("--sha256", default=sha256)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.checkpoint or args.project_path:
        if not (args.checkpoint and args.project_path):
            parser.error("local preparation requires both a checkpoint and --project-path")
        print(prepare(args.checkpoint, args.project_path, output=args.output, device=args.device))
    else:
        output = args.output.expanduser().resolve()
        if output == HERE:
            ensure_assets(output, args.download_url, args.sha256)
        else:
            if output.exists():
                parser.error("--output must be a new directory")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
                stage = Path(tmp) / "g1_track"
                stage.mkdir()
                for name in PROJECT_FILES:
                    shutil.copy2(HERE / name, stage / name)
                ensure_assets(stage, args.download_url, args.sha256)
                stage.rename(output)
        print(output)


if __name__ == "__main__":
    main()
