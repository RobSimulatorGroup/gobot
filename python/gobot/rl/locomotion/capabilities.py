"""Explicit stage evidence; absent measurements never mean a capability passed."""
from __future__ import annotations

CAPABILITIES = ('stand', 'forward_walk', 'forward_run', 'backward', 'lateral',
                'yaw', 'stop_start', 'terrain_routes', 'stairs_up', 'stairs_down',
                'low_obstacle', 'waypoints', 'skill_transition')
STAGE_REQUIREMENTS = {
    'flat_control': ('stand', 'forward_walk'),
    'flat_gait': ('stand', 'forward_walk', 'forward_run'),
    'rough': ('terrain_routes',),
    'mixed': ('stand', 'forward_walk', 'forward_run', 'backward', 'lateral', 'yaw', 'terrain_routes'),
}


def scenario_capability(scenario, *, run_speed=1.6):
    if scenario.get('sequence'):
        return 'stop_start'
    x, y, yaw = scenario['command']
    if abs(x) + abs(y) + abs(yaw) < .05:
        return 'stand'
    if abs(x) + abs(y) < .05:
        return 'yaw'
    if abs(y) > abs(x):
        return 'lateral'
    if x < 0:
        return 'backward'
    return 'forward_run' if x >= run_speed else 'forward_walk'


def assess_capabilities(scenarios, stage, *, final=False, natural=False, run_speed=1.6):
    if stage not in STAGE_REQUIREMENTS:
        raise ValueError(f'unknown capability stage {stage!r}')
    threshold = .95 if final else .9
    evidence = {key: [] for key in CAPABILITIES}
    for index, scenario in enumerate(scenarios):
        name = scenario_capability(scenario, run_speed=run_speed)
        key = 'natural_admission' if natural and name in ('forward_walk', 'forward_run') else 'admission'
        cells = scenario.get('cells', [])
        passed = bool(cells) and scenario.get(key, 0.) >= threshold and all(c.get(key, 0.) >= .9 for c in cells)
        evidence[name].append({'scenario': index, 'passed': passed})
        if scenario.get('route'):
            evidence['terrain_routes'].append({'scenario': index, 'passed': passed})
        # Legacy tile crossing does not establish stairs/obstacle route success.
        route_capability = scenario.get('route_capability')
        if route_capability in ('stairs_up', 'stairs_down', 'low_obstacle', 'waypoints') and scenario.get('route_protocol'):
            evidence[route_capability].append({'scenario': index, 'passed': passed})
    capabilities = {name: {'status': 'untested' if not rows else 'passed' if all(row['passed'] for row in rows) else 'failed',
                           'evidence': rows} for name, rows in evidence.items()}
    required = STAGE_REQUIREMENTS[stage]
    return {'capability_protocol': 'locomotion_capabilities_v1', 'capabilities': capabilities,
            'required_capabilities': list(required),
            'stage_passed': all(capabilities[key]['status'] == 'passed' for key in required)}
