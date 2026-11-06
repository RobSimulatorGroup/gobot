from gobot.rl.locomotion.capabilities import assess_capabilities


def row(command, success=1., **kwargs):
    return {'command': command, 'admission': success, 'natural_admission': success,
            'cells': [{'admission': success, 'natural_admission': success}], **kwargs}


def test_forward_stage_does_not_claim_missing_or_failed_skills():
    report = assess_capabilities([row([0, 0, 0]), row([.8, 0, 0]), row([-.3, 0, 0], 0.)], 'flat')
    assert report['stage_passed']
    assert report['capabilities']['backward']['status'] == 'failed'
    assert report['capabilities']['forward_run']['status'] == 'untested'
    assert not assess_capabilities([row([.8, 0, 0])], 'flat')['stage_passed']


def test_natural_and_route_evidence_cannot_be_inferred_from_survival():
    walk = row([.8, 0, 0]); walk['natural_admission'] = 0.
    report = assess_capabilities([row([0, 0, 0]), walk, row([2., 0, 0])], 'flat', natural=True)
    assert not report['stage_passed']
    report = assess_capabilities([row([.8, 0, 0], route=True)], 'rough')
    assert report['stage_passed']
    assert report['capabilities']['stairs_up']['status'] == 'untested'
