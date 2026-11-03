"""Prepare the full SuperMimic G1 library from a checksum-pinned ModelScope mirror."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from urllib.parse import quote, urlencode
import xml.etree.ElementTree as ET

DATASET = "AI-ModelScope/LAFAN1_Retargeting_Dataset"
REVISION = "cd6548b049528d19012d92bc36d23cb27dec0f85"
HF_DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
HF_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
BASE = "https://www.modelscope.cn"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_url(url: str) -> bytes:
    return subprocess.check_output([
        "curl", "-L", "--fail", "--silent", "--show-error", "--connect-timeout", "15",
        "--max-time", "90", "--retry", "2", url,
    ], timeout=300)


def tree(root: str, *, recursive: bool = False) -> list[dict]:
    query = urlencode({"Revision": REVISION, "Root": root, "Recursive": str(recursive).lower()})
    response = json.loads(read_url(f"{BASE}/api/v1/datasets/{DATASET}/repo/tree?{query}"))
    if response.get("Code") != 200:
        raise ValueError(f"ModelScope tree request failed: {response}")
    rows = response["Data"]["Files"]
    for row in rows:
        if row["Revision"] != REVISION:
            raise ValueError(f"Unpinned asset metadata: {row['Path']}")
    return rows


def prepare(reference: Path) -> dict:
    reference = reference.resolve()
    meshes = reference / "src/unisupermimic/assets/robots/g1/assets"
    motions = reference / "src/unisupermimic/assets/motions/g1/lafan1_supermimic"
    workspace = Path(__file__).resolve().parents[2]
    csvs = [row for row in tree("g1") if row["Path"].endswith(".csv")]
    if len(csvs) != 40:
        raise ValueError(f"Expected 40 reference clips, found {len(csvs)}")
    robots = {row["Path"]: row for row in tree("robot_description/g1", recursive=True)
              if row["Type"] == "blob"}

    def fetch(row: dict, target: Path, local: Path | None = None) -> dict:
        remote, expected = row["Path"], row["Sha256"]
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Missing SHA256 for {remote}")
        content = target.read_bytes() if target.exists() else b""
        origin = "existing"
        normalized_local = False
        if sha256(content) != expected:
            if local is not None and local.exists():
                candidate = local.read_bytes()
                if sha256(candidate) != expected and remote.endswith(".csv"):
                    candidate = candidate.replace(b"\r\n", b"\n")
                    normalized_local = True
                if sha256(candidate) == expected:
                    content, origin = candidate, "verified_local_copy"
            if sha256(content) != expected:
                content = read_url(f"{BASE}/datasets/{DATASET}/resolve/{REVISION}/{quote(remote)}")
                origin = "download"
            if sha256(content) != expected or len(content) != row["Size"]:
                raise ValueError(f"Asset checksum/size mismatch: {remote}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        record = {"remote": remote, "local": str(target.relative_to(reference)),
                  "sha256": expected, "bytes": len(content), "origin": origin,
                  "local_crlf_normalized": normalized_local}
        if remote.endswith(".csv"):
            frames = 0
            for values in csv.reader(io.StringIO(content.decode("utf-8"))):
                if len(values) != 36 or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError(f"Invalid frame {frames} in {remote}")
                frames += 1
            record["frames"] = frames
            if local is not None and local.exists():
                local_content = local.read_bytes()
                record["local_original_sha256"] = sha256(local_content)
                record["local_equivalent_after_crlf_normalization"] = (
                    local_content.replace(b"\r\n", b"\n") == content.replace(b"\r\n", b"\n")
                )
                if not record["local_equivalent_after_crlf_normalization"]:
                    raise ValueError(f"Existing motion has different values: {local}")
        print(f"ASSET {remote}: {len(content)} bytes ({origin})", flush=True)
        return record

    urdf_remote = "robot_description/g1/g1_29dof_rev_1_0.urdf"
    urdf = meshes / "supermimic/g1_29dof_rev_1_0.urdf"
    records = [fetch(robots[urdf_remote], urdf)]
    mesh_names = sorted({Path(mesh.attrib["filename"]).name
                         for mesh in ET.parse(urdf).getroot().iter("mesh")})
    jobs = [(row, motions / Path(row["Path"]).name,
             workspace / "examples/g1/assets/motions/raw" / Path(row["Path"]).name)
            for row in csvs]
    jobs += [(robots[f"robot_description/g1/meshes/{name}"], meshes / name,
              workspace / "examples/g1/assets/meshes" / name) for name in mesh_names]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(lambda job: fetch(*job), jobs))
    frame_count = sum(row.get("frames", 0) for row in records)
    if frame_count != 264705:
        raise ValueError(f"Expected 264705 frames, found {frame_count}")
    manifest = {
        "dataset": DATASET, "revision": REVISION, "source": BASE,
        "original_dataset": HF_DATASET, "original_revision": HF_REVISION,
        "clips": len(csvs), "frames": frame_count,
        "local_motion_equivalence_count": sum(
            row.get("local_equivalent_after_crlf_normalization", False) for row in records),
        "files": sorted(records, key=lambda row: row["remote"]),
        "evaluation_protocol": "all 40 full clips from first frame; not held-out velocity/terrain admission",
    }
    (reference / "reference-assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(records)} assets, {len(csvs)} clips, {frame_count} frames", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    prepare(parser.parse_args().reference)


if __name__ == "__main__":
    main()
