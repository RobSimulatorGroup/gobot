"""Small content-addressed cache for prepared physics runtime artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Gobot's supported wheel is Linux today.
    fcntl = None


_CACHE_MANIFEST = "manifest.json"
_CACHE_SCHEMA = 1


def physics_cache_root() -> Path:
    """Return Gobot's untracked per-user physics cache root."""

    override = os.environ.get("GOBOT_PHYSICS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "gobot" / "physics"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_component(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    result = result.strip("._")
    if not result:
        raise ValueError("cache namespace must contain a portable path character")
    return result


class ContentAddressedCache:
    """Validate and atomically publish immutable cache directories."""

    def __init__(self, namespace: str, *, root: Path | None = None) -> None:
        self._namespace = _safe_component(str(namespace))
        self._root = (root or physics_cache_root()) / self._namespace

    @property
    def root(self) -> Path:
        return self._root

    @contextmanager
    def _locked(self, cache_id: str) -> Iterator[None]:
        lock_directory = self._root / ".locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        lock_path = lock_directory / f"{cache_id}.lock"
        with lock_path.open("a+b") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _cache_id(key: str) -> str:
        value = str(key)
        if value.startswith("sha256:") and len(value) == 71:
            try:
                int(value[7:], 16)
            except ValueError:
                pass
            else:
                return value[7:]
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _remove_entry(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)

    @staticmethod
    def _manifest_for(
        directory: Path,
        *,
        key: str,
        metadata: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        files = []
        manifest_path = directory / _CACHE_MANIFEST
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path == manifest_path:
                continue
            relative = path.relative_to(directory).as_posix()
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "digest": _file_digest(path),
                }
            )
        return {
            "schema_version": _CACHE_SCHEMA,
            "key": str(key),
            "files": files,
            "metadata": dict(metadata or {}),
        }

    @staticmethod
    def _valid(directory: Path, key: str) -> bool:
        try:
            if directory.is_symlink() or not directory.is_dir():
                return False
            manifest_path = directory / _CACHE_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != _CACHE_SCHEMA:
                return False
            if manifest.get("key") != str(key):
                return False
            files = manifest.get("files")
            if not isinstance(files, list):
                return False
            listed: set[str] = set()
            for record in files:
                if not isinstance(record, Mapping):
                    return False
                relative = str(record.get("path", ""))
                candidate = directory / relative
                if (
                    not relative
                    or relative.startswith("/")
                    or ".." in Path(relative).parts
                    or not candidate.is_file()
                    or relative in listed
                ):
                    return False
                listed.add(relative)
                if candidate.stat().st_size != int(record.get("size", -1)):
                    return False
                if _file_digest(candidate) != str(record.get("digest", "")):
                    return False
            actual = {
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and path != manifest_path
            }
            return actual == listed
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def get_or_create(
        self,
        key: str,
        build: Callable[[Path], Mapping[str, Any] | None],
    ) -> Path:
        """Return a verified immutable entry, rebuilding corruption atomically."""

        cache_id = self._cache_id(key)
        self._root.mkdir(parents=True, exist_ok=True)
        entry = self._root / cache_id
        with self._locked(cache_id):
            if self._valid(entry, key):
                return entry
            if entry.exists() or entry.is_symlink():
                self._remove_entry(entry)

            temporary = Path(
                tempfile.mkdtemp(prefix=f".{cache_id}.", dir=str(self._root))
            )
            try:
                metadata = build(temporary)
                manifest = self._manifest_for(
                    temporary,
                    key=key,
                    metadata=metadata,
                )
                (temporary / _CACHE_MANIFEST).write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                if not self._valid(temporary, key):
                    raise RuntimeError("physics cache builder produced an invalid entry")
                temporary.rename(entry)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return entry


__all__ = ["ContentAddressedCache", "physics_cache_root"]
