from copy import deepcopy
import pytest
from examples.g1.reproduction.compare_benchmarks import compare,validate


def payload():
    return {'summary':{'clips':1,'failed':0,'mean_norm_return':.8,'step_weighted_norm_return':.8},
            'clips':[{'name':'walk','frames':10,'steps':10,'norm_return':.8,'failed':False,'fail_step':-1}]}


def test_shortened_success_is_rejected():
    data=payload();data['clips'][0]['steps']=9
    with pytest.raises(ValueError,match='not completed'):validate(data)


def test_comparison_preserves_disagreeing_failures():
    a=payload();b=deepcopy(a)
    b['clips'][0].update(steps=5,failed=True,fail_step=5,norm_return=.9)
    b['summary'].update(failed=1,mean_norm_return=.9,step_weighted_norm_return=.9)
    report=compare(a,b)
    assert report['gobot_only_failures']==['walk']
    assert report['failed_count_difference']==1
    assert report['mean_return_difference']==pytest.approx(.1)
    assert not report['fresh_training_reproduction']


def test_inconsistent_aggregate_is_rejected():
    data=payload();data['summary']['mean_norm_return']=.9
    with pytest.raises(ValueError,match='summary'):validate(data)
