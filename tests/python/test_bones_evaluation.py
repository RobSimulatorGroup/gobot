"""Integrity gates for complete, independently sharded BONES evaluation."""
from copy import deepcopy
import pytest
from examples.g1.reproduction.evaluate_bones import partition, summarize, validate_rows, validate_report


def result(name, frames=5, failed=False):
    steps = 3 if failed else frames
    return {'name': name, 'frames': frames, 'steps': steps, 'failed': failed,
            'fail_step': steps if failed else -1, 'norm_return': .8,
            'terms': {'motion_body_pos': .01}}


def test_length_buckets_preserve_every_clip_with_bounded_memory():
    rows = [{'name': f'clip{i}', 'frames': length} for i, length in enumerate([3, 4, 5, 6, 7, 8, 30])]
    parts = partition(rows, 3, 15)
    flattened = [r for part in parts for r in part]
    assert sorted(r['name'] for r in flattened) == sorted(r['name'] for r in rows)
    assert all(len(p) <= 3 and (len(p) == 1 or sum(r['frames'] for r in p) <= 15) for p in parts)
    assert parts[0] == [{'name': 'clip6', 'frames': 30}]


@pytest.mark.parametrize('mutation', [
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(steps=4),
    lambda rows: rows[0].update(frames=6),
    lambda rows: rows[0].update(norm_return=float('nan')),
    lambda rows: rows[1].update(fail_step=2),
    lambda rows: rows[0]['terms'].update(motion_body_pos=float('inf')),
])
def test_incomplete_or_corrupt_results_are_rejected(mutation):
    rows = [result('walk'), result('fall', failed=True)]
    mutation(rows)
    with pytest.raises(ValueError):
        validate_rows(rows, {'walk': 5, 'fall': 5})


def test_actual_steps_and_expected_length_returns_are_distinct():
    rows = [result('walk'), result('fall', failed=True)]
    summary = summarize(rows)
    assert summary['mean_norm_return'] == pytest.approx(.8)
    assert summary['step_weighted_norm_return'] == pytest.approx(.8)
    assert summary['expected_length_norm_return'] == pytest.approx(.64)
    assert summary['failure_rate'] == .5 and summary['frame_coverage'] == .8


def test_shards_must_belong_to_the_same_plan_and_expected_membership():
    rows = [result('walk')]
    plan = {'shards': [[{'name': 'walk', 'frames': 5}]]}
    report = {'complete': True, 'plan_sha256': 'pinned', 'shard_index': 0,
              'clips': rows, 'summary': summarize(rows)}
    validate_report(report, plan, 'pinned', 0)
    with pytest.raises(ValueError):
        validate_report(report, plan, 'other', 0)
    report['summary']['failed'] = 1
    with pytest.raises(ValueError):
        validate_report(report, plan, 'pinned', 0)
