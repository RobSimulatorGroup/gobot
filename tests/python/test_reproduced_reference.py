"""Baseline evidence must bind actual bytes and preserve benchmark denominators."""
import hashlib
import json
from pathlib import Path

import pytest

from examples.g1.reproduction.reference import benchmark_summary, filtered_evaluation, verify_release


def test_benchmark_keeps_executed_step_weighting_and_failure_count():
    rows = [{'name': 'long', 'steps': 100, 'norm_return': .8, 'failed': False},
            {'name': 'short_failure', 'steps': 10, 'norm_return': .9, 'failed': True}]
    report = benchmark_summary(rows)
    assert report['mean_norm_return'] == pytest.approx(.85)
    assert report['step_weighted_norm_return'] == pytest.approx(89/110)
    assert report['failed'] == 1
    with pytest.raises(ValueError, match='unique'):
        benchmark_summary([rows[0], rows[0]])
    with pytest.raises(ValueError, match='invalid'):
        benchmark_summary([dict(rows[0], norm_return=float('nan'))])


def test_release_verifies_size_and_digest_before_checkpoint_can_be_used(tmp_path):
    content = b'verified fixture'
    path = tmp_path/'checkpoint.bin'
    path.write_bytes(content)
    expected = 'sha256:'+hashlib.sha256(content).hexdigest()
    records = [{'name': path.name, 'size': len(content), 'sha256': expected}]
    assert verify_release(tmp_path, records) == {path.name: expected}
    path.write_bytes(b'x'*len(content))
    with pytest.raises(ValueError, match='pinned bytes'):
        verify_release(tmp_path, records)


def test_evaluation_filter_does_not_claim_to_reconstruct_training_set(tmp_path):
    path = tmp_path/'scripts/data/filter_bones_seed.py'
    path.parent.mkdir(parents=True)
    path.write_text("SONIC_KEYWORDS=['chair']\nSONIC_ACROBATICS=['cartwheel']\nPAPER_EXTRA=['_box_']\n")
    rows = [{'name': name} for name in ('walk', 'CartWheel_fast', 'chair_sit', 'kick')]
    kept, identity = filtered_evaluation(rows, tmp_path)
    assert [row['name'] for row in kept] == ['walk', 'kick']
    assert 'evaluation' in identity['scope']
    assert identity['filter_sha256'].startswith('sha256:')


def test_pinned_release_records_resumed_configuration_and_version_bridge():
    lock = json.loads((Path(__file__).resolve().parents[2]/'examples/g1/reproduction/reference.lock.json').read_text())
    assert lock['physics_baselines']['published']['mujoco'] == '3.16.0'
    assert lock['physics_baselines']['paired']['mujoco'] == '3.17.0'
    assert lock['physics_baselines']['paired']['version_effect_evaluation_required']
    assert len(lock['release']['assets']) == 3


def test_static_audit_separates_plane_display_size_from_physical_shape():
    import mujoco
    from examples.g1.prepare_tracking import audit_model
    template = '''<mujoco><worldbody>
      <geom name="ground" type="plane" size="0 0 {grid}"/>
      <body name="base" pos="0 0 1"><freejoint name="root"/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom name="body" type="sphere" size="{radius}"/>
      </body></worldbody></mujoco>'''
    reference = mujoco.MjModel.from_xml_string(template.format(grid=.01, radius=.1))
    def audit(radius):
        return audit_model(reference, {'content': template.format(grid=.05, radius=radius),
                                      'content_digest': 'test'}, '')
    report = audit(.1)
    assert report['passed']
    assert report['display_differences'][0]['field'] == 'ground/plane_display_size'
    changed = audit(.2)
    assert not changed['passed']
    assert any(row['field'] == 'body/size' for row in changed['failed_checks'])
