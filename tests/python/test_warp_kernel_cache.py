from pathlib import Path
from types import SimpleNamespace

import pytest

from gobot.sim.providers.warp_cache import prepare_warp_kernel_cache


def _entry(root: Path, name: str, *, metadata: bytes = b'{}', binary: bytes = b'ptx') -> Path:
    entry = root / name
    entry.mkdir()
    (entry / f"{name}.meta").write_bytes(metadata)
    (entry / f"{name}.sm89.ptx").write_bytes(binary)
    return entry


@pytest.mark.parametrize("metadata,binary", [
    (b"", b"ptx"),
    (b'{"kernels":', b"ptx"),
    (b"\xff", b"ptx"),
    (b"null", b"ptx"),
    (b"{}", b""),
])
def test_damaged_modules_are_backed_up_without_touching_valid_cache(tmp_path, metadata, binary):
    warp = SimpleNamespace(config=SimpleNamespace(kernel_cache_dir=str(tmp_path)))
    damaged = _entry(tmp_path, "wp_solver_abc1234", metadata=metadata, binary=binary)
    valid = _entry(tmp_path, "wp_solver_def5678")
    original = {file.name: file.read_bytes() for file in damaged.iterdir()}

    backups = prepare_warp_kernel_cache(warp)

    assert len(backups) == 1
    assert not damaged.exists()  # Warp must see a complete cache miss.
    assert {file.name: file.read_bytes() for file in backups[0].iterdir()} == original
    assert (valid / f"{valid.name}.meta").read_bytes() == b"{}"
    assert (valid / f"{valid.name}.sm89.ptx").read_bytes() == b"ptx"
    assert prepare_warp_kernel_cache(warp) == ()
    _entry(tmp_path, damaged.name)  # Simulate Warp's next on-demand compilation.
    assert prepare_warp_kernel_cache(warp) == ()


def test_unpublished_builds_unrelated_files_and_symlink_targets_are_preserved(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    external = _entry(tmp_path, "external", metadata=b"")
    build = _entry(cache, "wp_solver_abc1234_p123_t456", metadata=b"")
    unrelated = _entry(cache, "other_abc1234", metadata=b"")
    (cache / "wp_solver_def5678").symlink_to(external, target_is_directory=True)
    warp = SimpleNamespace(config=SimpleNamespace(kernel_cache_dir=cache))

    assert prepare_warp_kernel_cache(warp) == ()
    assert (external / "external.meta").read_bytes() == b""
    assert build.exists() and unrelated.exists()


def test_uninitialized_warp_does_not_create_a_cache(tmp_path):
    assert prepare_warp_kernel_cache(SimpleNamespace()) == ()
    cache = tmp_path / "missing"
    assert prepare_warp_kernel_cache(SimpleNamespace(config=SimpleNamespace(kernel_cache_dir=cache))) == ()
    assert not cache.exists()


def test_empty_lto_pair_is_backed_up_and_valid_link_input_is_preserved(tmp_path):
    lto = tmp_path / "lto"
    lto.mkdir()
    (lto / "abc123.lto").write_bytes(b"")
    (lto / "abc123_fatbin.lto").write_bytes(b"")
    valid = lto / "def456.lto"
    valid.write_bytes(b"compiled LTO")
    warp = SimpleNamespace(config=SimpleNamespace(kernel_cache_dir=tmp_path))

    backups = prepare_warp_kernel_cache(warp)

    assert len(backups) == 2
    assert all(path.read_bytes() == b"" for path in backups)
    assert tuple(lto.iterdir()) == (valid,)
    assert valid.read_bytes() == b"compiled LTO"
    assert prepare_warp_kernel_cache(warp) == ()
