#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


REVISION = "65884107860281bfcde5b58904c327a923da7cc6"
FILE_PATH = "train/point_cloud/iteration_7000/point_cloud.ply"
EXPECTED_SIZE = 165_633_787
EXPECTED_SHA256 = "e1bc6c22fa74db350a783385f578be0eb5465c1df0daaedb33fa10c99e10c380"
DEFAULT_URLS = (
    f"https://huggingface.co/datadude/gaussian_splatting/resolve/{REVISION}/{FILE_PATH}",
    f"https://hf-mirror.com/datadude/gaussian_splatting/resolve/{REVISION}/{FILE_PATH}",
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
    request = urllib.request.Request(url, headers={"User-Agent": "Gobot sample downloader"})
    received = 0
    with urllib.request.urlopen(request, timeout=30) as response, destination.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            received += len(block)
            print(
                f"\rDownloading train 3DGS: {received / (1024 * 1024):.1f} / "
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
        default=Path(__file__).resolve().parent / "assets" / "train-7000.ply",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if not args.force and is_valid(output):
        print(f"Sample is already verified: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    override = os.environ.get("GOBOT_GSPLAT_SAMPLE_URL")
    urls = (override,) if override else DEFAULT_URLS
    for url in urls:
        try:
            temporary.unlink(missing_ok=True)
            print(f"Source: {url}")
            download(url, temporary)
            if temporary.stat().st_size != EXPECTED_SIZE:
                raise RuntimeError(
                    f"size mismatch: expected {EXPECTED_SIZE}, got {temporary.stat().st_size}"
                )
            actual_hash = sha256(temporary)
            if actual_hash != EXPECTED_SHA256:
                raise RuntimeError(
                    f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
                )
            temporary.replace(output)
            print(f"Verified sample: {output}")
            return 0
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            print(f"Download failed from {url}: {error}", file=sys.stderr)

    temporary.unlink(missing_ok=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
