from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import patch
import pytest

from examples.g1.tracking_experiment import TrackingExperiment, require_tracking_admission
from gobot.rl.locomotion.budget import TrainingBudget, bind_budget_window


def test_cluster_accounting_keeps_wall_limit_and_existing_authorizations(tmp_path):
    ledger=tmp_path/'cluster.json'
    with patch('gobot.rl.locomotion.budget.time.monotonic',return_value=0.) as clock:
        with TrainingBudget(ledger,total_hours=None,run_hours=3) as budget:
            assert budget.remaining==10800 and budget.total_hours is None
            clock.return_value=60
        assert json.loads(ledger.read_text())['seconds']==60
    window=tmp_path/'window.json'
    window.write_text(json.dumps({'baseline_seconds':0,'additional_gpu_hours':1,'original_total_cap_hours':24}))
    bind_budget_window(ledger,window)
    with TrainingBudget(ledger,total_hours=None,run_hours=3) as budget:
        assert budget.remaining<=3540 and budget.total_hours==1


def test_formal_recipe_keeps_reference_network_and_training_split():
    recipe=TrackingExperiment()
    assert recipe.learner_config().horizon==32
    assert recipe.learner_config().test_hidden_dims is None
    with pytest.raises(ValueError,match='full networks'):
        replace(recipe,algorithm={'test_hidden_dims':[8]})
    with pytest.raises(ValueError,match='validation'):
        replace(recipe,split='validation')
    with pytest.raises(ValueError,match='10 updates'):
        replace(recipe,purpose='pipeline_smoke')


def test_admission_requires_full_coverage_and_matching_physics():
    report={'protocol':'paired_tracking_fidelity_v1','passed':True,'new_failed_clips':[],
            'candidate_physics':{'robot_signature':'robot'},'per_clip':[
                {'clip':'a','reference_complete':True,'candidate_complete':True,'return_delta':-.001},
                {'clip':'b','reference_complete':False,'candidate_complete':False,'return_delta':0.}]}
    require_tracking_admission(report,robot_signature='robot',clip_names=['a','b'])
    with pytest.raises(ValueError,match='full pinned'):
        require_tracking_admission(report,robot_signature='robot',clip_names=['a','b','c'])
    with pytest.raises(ValueError,match='different'):
        require_tracking_admission(report,robot_signature='other',clip_names=['a','b'])
    changed=deepcopy(report);changed['per_clip'][0]['candidate_complete']=False
    with pytest.raises(ValueError,match='new failed'):
        require_tracking_admission(changed,robot_signature='robot',clip_names=['a','b'])
