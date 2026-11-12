"""Complete evaluation shard contracts and bounded queue membership."""
import hashlib
import json
from pathlib import Path

import pytest

from gobot.rl.supermimic.official.evaluate import merge_reports, summarize, validate_row


def row(name,failed=False):
    return {'name':name,'frames':11,'expected_steps':10,'executed_steps':5 if failed else 10,
            'return':.7,'coverage':.5 if failed else 1.,'failed':failed,'finished':not failed,
            'completed':not failed,'capped':False}


def reports(tmp_path):
    names=['a','b','c','d','e']
    paths=[]
    for index in range(2):
        rows=[row(n,n=='b') for n in names[index::2]]
        payload={'protocol':'gobot_supermimic_official_evaluation_v1','checkpoint_sha256':'checkpoint',
            'data_manifest_sha256':'data','selection_sha256':hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest(),
            'selection_count':len(names),'shard_index':index,'shard_count':2,
            'evaluation_options':{'num_envs':2},'evaluation_sources':{'evaluate.py':'abc'},
            'complete':True,'clips':rows,'summary':summarize(rows)}
        p=tmp_path/f'{index}.json';p.write_text(json.dumps(payload));paths.append(p)
    return names,paths


def test_complete_shards_merge_in_original_manifest_order(tmp_path):
    names,paths=reports(tmp_path)
    result=merge_reports(paths[::-1],names,tmp_path/'merged.json')
    assert [r['name'] for r in result['clips']]==names
    assert result['summary']['failures']==1 and result['summary']['failure_rate']==.2
    assert result['summary']['frame_coverage']==.9
    assert result['complete']


@pytest.mark.parametrize('damage',['missing','duplicate','checkpoint','membership','capped','summary'])
def test_incomplete_or_inconsistent_shards_cannot_be_accepted(tmp_path,damage):
    names,paths=reports(tmp_path)
    if damage=='missing':paths=paths[:1]
    elif damage=='duplicate':paths=[paths[0],paths[0]]
    else:
        r=json.loads(paths[1].read_text())
        if damage=='checkpoint':r['checkpoint_sha256']='other'
        if damage=='membership':r['clips'][0]['name']='unknown'
        if damage=='capped':r['complete']=False
        if damage=='summary':r['summary']['mean_return']+=.1
        paths[1].write_text(json.dumps(r))
    with pytest.raises(ValueError):merge_reports(paths,names,tmp_path/'merged.json')
    assert not (tmp_path/'merged.json').exists()


@pytest.mark.parametrize('key,value',[('return',float('nan')),('executed_steps',12),('coverage',.5),('completed',False),('failed',1)])
def test_invalid_clip_result_is_rejected(key,value):
    r=row('a');r[key]=value
    with pytest.raises(ValueError):validate_row(r)
