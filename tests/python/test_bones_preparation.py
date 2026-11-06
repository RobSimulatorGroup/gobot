"""Data integrity, selection and compact layout for resumable BONES preparation."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.g1.reproduction.prepare_bones import (
    compact_reference, digest, inventory, pack_library, validate_shard, write_manifests,
)


def make_npz(path, frames=3, value=1.):
    pos = np.full((frames, 2, 3), value, dtype=np.float32)
    pos[:, 0] = 99.  # The first link must not be mistaken for the floating base.
    quat = np.zeros((frames, 2, 4), dtype=np.float32)
    quat[..., 0] = 1.
    np.savez(path, body_names=np.array(['torso_link', 'pelvis']), body_pos_w=pos,
             body_quat_w=quat, joint_pos=np.full((frames, 29), 2., dtype=np.float32),
             joint_vel=np.full((frames, 29), 3., dtype=np.float32),
             body_lin_vel_w=np.full((frames, 2, 3), 4., dtype=np.float32),
             body_ang_vel_w=np.full((frames, 2, 3), 5., dtype=np.float32))


def test_compact_layout_uses_named_pelvis_and_reference_field_order(tmp_path):
    path = tmp_path / 'motion.npz'
    make_npz(path)
    actual = compact_reference(path)
    expected = np.array([1., 1., 1., 1., 0., 0., 0.] + [2.] * 29 + [4.] * 3 + [5.] * 3 + [3.] * 29,
                        dtype=np.float16)
    assert actual.dtype == np.float16 and actual.shape == (3, 71)
    np.testing.assert_array_equal(actual, np.tile(expected, (3, 1)))


def test_compact_overflow_is_not_silently_packed(tmp_path):
    path = tmp_path / 'motion.npz'
    make_npz(path, value=1e10)
    with pytest.warns(RuntimeWarning, match='overflow'), pytest.raises(ValueError, match='nonfinite'):
        compact_reference(path)


def test_duplicate_csv_stems_in_date_directories_are_rejected(tmp_path):
    for date in ('day1', 'day2'):
        folder = tmp_path / date
        folder.mkdir()
        (folder / 'walk.csv').write_text('same name')
    with pytest.raises(ValueError, match='unique'):
        inventory(tmp_path)


def test_training_keeps_acrobatics_and_evaluation_excludes_them(tmp_path):
    part = tmp_path / 'part'
    part.mkdir()
    rows = []
    for name, keep, eval_keep, reason, eval_reason in (
        ('cartwheel', 1, 0, '', 'name:cartwheel'),
        ('sit_motion', 0, 0, 'name:sit_', 'name:sit_'),
        ('walk', 1, 1, '', ''),
    ):
        make_npz(part / (name + '.npz'))
        rows.append(dict(name=name, frames=3, keep=keep, eval_keep=eval_keep,
                         reason=reason, eval_reason=eval_reason))
    (part / 'result.json').write_text(json.dumps(dict(rows=rows, kinematics_digest='rig',
                                      reference_comparisons=[], filter_disagreements=[])))
    output = tmp_path / 'output'
    output.mkdir()
    result = write_manifests([part], output, 'rig')
    assert result['train_clips'] == 2 and result['eval_clips'] == 1
    assert json.loads((output / 'train_clips.json').read_text()) == ['cartwheel', 'walk']
    assert json.loads((output / 'eval_clips.json').read_text()) == ['walk']
    assert {p.stem for p in (output / 'npz').glob('*.npz')} == {'cartwheel', 'walk'}
    assert (output / 'npz/cartwheel.npz').samefile(part / 'cartwheel.npz')


def test_packing_preserves_sorted_clip_boundaries_and_all_frames(tmp_path):
    parts = []
    expected = []
    for index, (name, count) in enumerate([('a_walk', 3), ('z_run', 5)]):
        part = tmp_path / str(index)
        part.mkdir()
        values = np.full((count, 71), index + .25, dtype=np.float16)
        np.save(part / 'references.npy', values)
        (part / 'result.json').write_text(json.dumps({'rows': [
            {'name': name, 'frames': count, 'keep': 1},
            {'name': name + '_rejected', 'frames': 30, 'keep': 0}]}))
        parts.append(part)
        expected.append(values)
    output = tmp_path / 'library.pt'
    report = pack_library(parts, output)
    actual = torch.load(output, weights_only=True)
    assert actual['names'] == ['a_walk', 'z_run']
    assert actual['starts'] == [0, 3] and actual['lengths'] == [3, 5]
    np.testing.assert_array_equal(actual['ref'].numpy(), np.concatenate(expected))
    assert report['frames'] == 8 and report['clips'] == 2


def test_completed_shard_cannot_resume_after_raw_input_changes(tmp_path):
    source = tmp_path / 'walk.csv'
    source.write_text('original input')
    part = tmp_path / 'part'
    part.mkdir()
    make_npz(part / 'walk.npz')
    np.save(part / 'references.npy', compact_reference(part / 'walk.npz'))
    (part / 'result.json').write_text(json.dumps({
        'plan_sha256': 'plan', 'rows': [{'name': 'walk', 'source_sha256': digest(source),
                                       'npz_sha256': digest(part / 'walk.npz')}],
        'references_sha256': digest(part / 'references.npy')}))
    assert validate_shard(part, [source], 'plan') is not None
    source.write_text('changed input')
    with pytest.raises(ValueError, match='content changed'):
        validate_shard(part, [source], 'plan')
