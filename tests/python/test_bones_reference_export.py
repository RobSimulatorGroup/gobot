"""Reference exports are content-addressed and do not consume edited worktrees."""
import hashlib
import subprocess

import pytest

from examples.g1.reproduction.export_bones_reference import git_bytes, write_verified


def test_export_reads_committed_bytes_instead_of_working_tree(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()

    git('init', '-q')
    path = tmp_path / 'filter.py'
    path.write_text('fixed source\n')
    git('add', 'filter.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
    revision = git('rev-parse', 'HEAD')
    path.write_text('uncommitted edit\n')
    assert git_bytes(tmp_path, revision, 'filter.py') == b'fixed source\n'
    assert path.read_text() == 'uncommitted edit\n'


def test_hash_mismatch_does_not_create_output(tmp_path):
    path = tmp_path / 'export' / 'filter.py'
    with pytest.raises(ValueError, match='hash differs'):
        write_verified(path, b'changed source', hashlib.sha256(b'fixed source').hexdigest())
    assert not path.exists()


def test_idempotent_export_refuses_to_overwrite_changed_input(tmp_path):
    path = tmp_path / 'filter.py'
    sha = write_verified(path, b'fixed source')
    assert write_verified(path, b'fixed source', sha) == sha
    path.write_bytes(b'local edit')
    with pytest.raises(ValueError, match='refusing to replace'):
        write_verified(path, b'fixed source', sha)
    assert path.read_bytes() == b'local edit'
