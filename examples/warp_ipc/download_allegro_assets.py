"""Explicitly download the pinned, license-audited Allegro Hand asset set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "allegro_assets.json"


def _validate_file(path: Path, *, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == sha256


def download_assets(output: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    revision = str(manifest["revision"])
    asset_root = str(manifest["asset_root"])
    base_url = (
        "https://raw.githubusercontent.com/newton-physics/newton-assets/"
        f"{revision}/{asset_root}/"
    )
    output.mkdir(parents=True, exist_ok=True)

    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"asset manifest contains unsafe path {relative}")
        destination = output / relative
        size = int(entry["size"])
        sha256 = str(entry["sha256"])
        if _validate_file(destination, size=size, sha256=sha256):
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        request = urllib.request.Request(
            base_url + relative.as_posix(),
            headers={"User-Agent": "gobot-warp-ipc-asset-downloader/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
                raise RuntimeError(f"downloaded asset failed validation: {relative}")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    invalid = [
        entry["path"]
        for entry in manifest["files"]
        if not _validate_file(
            output / entry["path"],
            size=int(entry["size"]),
            sha256=str(entry["sha256"]),
        )
    ]
    if invalid:
        raise RuntimeError(f"Allegro asset validation failed: {invalid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "assets" / "wonik_allegro",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    download_assets(output)
    print(output)


if __name__ == "__main__":
    main()
