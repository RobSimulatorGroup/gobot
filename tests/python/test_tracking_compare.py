from copy import deepcopy
import pytest
from examples.locomotion.tracking_compare import compare


def result():
    return {'per_clip':{'walk':{'steps':100,'expected_steps':100,'failed':False,'norm_return':.8},
                        'dance':{'steps':70,'expected_steps':100,'failed':True,'norm_return':.5}}}


def test_better_mean_cannot_hide_a_new_failed_clip():
    a=result();b=deepcopy(a)
    b['per_clip']['walk'].update(steps=99,failed=True)
    b['per_clip']['dance'].update(steps=100,failed=False,norm_return=.9)
    report=compare(a,b)
    assert report['mean_return_delta']>0
    assert report['new_failed_clips']==['walk']
    assert not report['passed']


def test_truncated_and_shortened_evaluation_cannot_pass():
    a=result();b=deepcopy(a)
    b['per_clip']['walk']['steps']=80
    assert not compare(a,b)['passed']
    b['per_clip']['walk']['expected_steps']=80
    with pytest.raises(ValueError,match='denominator'):compare(a,b)
    with pytest.raises(ValueError,match='screening'):compare(a,{'screening':True,**a})
