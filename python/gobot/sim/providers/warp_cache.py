"""Recover incomplete Warp disk cache entries before provider initialization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import uuid
from typing import Any


_LOGGER = logging.getLogger(__name__)
_BINARY_SUFFIXES = frozenset({".ptx", ".cubin", ".o", ".obj"})


def _quarantine(cache: Path, path: Path, reason: str) -> Path:
    quarantine = cache / ".gobot-quarantine"
    quarantine.mkdir(exist_ok=True)
    destination = quarantine / f"{path.name}-{uuid.uuid4().hex}"
    path.rename(destination)
    _LOGGER.warning(
        "Quarantined damaged Warp cache (%s) at %s; the kernel will be rebuilt.",
        reason, destination,
    )
    return destination


def _invalid_entry(entry: Path) -> str | None:
    for path in entry.iterdir():
        # Do not inspect external targets or temporary compiler subdirectories.
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.suffix in _BINARY_SUFFIXES and path.stat().st_size == 0:
                return f"empty kernel binary: {path.name}"
            if path.suffix == ".meta":
                with path.open(encoding="utf-8") as stream:
                    metadata = json.load(stream)
                if not isinstance(metadata, dict):
                    return f"invalid metadata object: {path.name}"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return f"invalid JSON metadata: {path.name}"
        except FileNotFoundError:
            # Another process may have just repaired this entry.
            continue
    return None


def prepare_warp_kernel_cache(warp: Any) -> tuple[Path, ...]:
    """Quarantine corrupt published modules; Warp rebuilds them on demand.

    Call after warp.init(), before importing model data or warming up kernels.
    This never retries physics steps or changes Warp's process-global options.
    Warp 1.17 accepts a cached module based only on file existence, so a zero
    length metadata/binary file otherwise poisons every subsequent startup.
    """
    configured = getattr(getattr(warp, "config", None), "kernel_cache_dir", None)
    if not configured:
        return ()
    cache = Path(configured)
    if not cache.is_dir():
        return ()
    recovered = []
    # Tile solvers also consume shared LTO/fatbin files. Warp treats empty
    # bytes as a cache hit, then nvJitLink rejects them as invalid input.
    lto = cache / "lto"
    if not lto.is_symlink() and lto.is_dir():
        for path in lto.glob("*.lto"):
            try:
                if not path.is_symlink() and path.is_file() and path.stat().st_size == 0:
                    recovered.append(_quarantine(cache, path, "empty LTO binary"))
            except FileNotFoundError:
                continue
    for entry in cache.iterdir():
        if not entry.name.startswith("wp_") or entry.is_symlink() or not entry.is_dir():
            continue
        # Warp writes into module_p<PID>_t<TID> before publishing atomically.
        # Only inspect published module directories, ending in a hex hash.
        suffix = entry.name.rsplit("_", 1)[-1]
        if not suffix or any(char not in "0123456789abcdef" for char in suffix):
            continue
        try:
            reason = _invalid_entry(entry)
            if reason is None:
                continue
            # Move the whole entry: retaining an empty binary beside a missing
            # .meta file can prevent Warp from publishing its rebuilt binary.
            destination = _quarantine(cache, entry, reason)
        except FileNotFoundError:
            continue
        recovered.append(destination)
    return tuple(recovered)
