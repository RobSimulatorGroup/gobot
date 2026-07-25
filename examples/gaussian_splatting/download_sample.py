#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


REVISION = "ed0588b29edea35e36dad784f73c1f502cc8a0d2"
FILE_PATH = "FO_dataset/playroom/point_cloud/iteration_7000/point_cloud.ply"
EXPECTED_SIZE = 370_875_860
EXPECTED_SHA256 = "201bc92b65594727a3ecfbe7e658c09ac3f8be753e2e2024047cd3ea1fe31d8c"
DEFAULT_URLS = (
    f"https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/{REVISION}/{FILE_PATH}",
    f"https://hf-mirror.com/datasets/Voxel51/gaussian_splatting/resolve/{REVISION}/{FILE_PATH}",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == EXPECTED_SIZE
        and sha256(path) == EXPECTED_SHA256
    )


def download(url: str, destination: Path) -> None:
    existing_size = destination.stat().st_size if destination.is_file() else 0
    if existing_size > EXPECTED_SIZE:
        destination.unlink()
        existing_size = 0

    headers = {"User-Agent": "Gobot sample downloader"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        append = existing_size > 0 and getattr(response, "status", None) == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing_size}-"):
                raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
        else:
            existing_size = 0

        received = existing_size
        mode = "ab" if append else "wb"
        with destination.open(mode) as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
                received += len(block)
                print(
                    f"\rDownloading playroom 3DGS: {received / (1024 * 1024):.1f} / "
                    f"{EXPECTED_SIZE / (1024 * 1024):.1f} MiB",
                    end="",
                    flush=True,
                )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Gobot's optional 3DGS environment sample."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "assets" / "playroom-7000.ply",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if not args.force and is_valid(output):
        print(f"Sample is already verified: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if args.force:
        temporary.unlink(missing_ok=True)
    elif is_valid(temporary):
        temporary.replace(output)
        print(f"Recovered verified sample: {output}")
        return 0
    elif temporary.is_file() and temporary.stat().st_size >= EXPECTED_SIZE:
        temporary.unlink()

    override = os.environ.get("GOBOT_GSPLAT_SAMPLE_URL")
    urls = (override,) if override else DEFAULT_URLS
    for url in urls:
        try:
            print(f"Source: {url}")
            download(url, temporary)
            if temporary.stat().st_size != EXPECTED_SIZE:
                raise RuntimeError(
                    f"size mismatch: expected {EXPECTED_SIZE}, got {temporary.stat().st_size}"
                )
            actual_hash = sha256(temporary)
            if actual_hash != EXPECTED_SHA256:
                temporary.unlink()
                raise RuntimeError(
                    f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
                )
            temporary.replace(output)
            print(f"Verified sample: {output}")
            return 0
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            print(f"Download failed from {url}: {error}", file=sys.stderr)

    if temporary.is_file():
        print(
            f"Partial download retained for retry: {temporary} "
            f"({temporary.stat().st_size / (1024 * 1024):.1f} MiB)",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
