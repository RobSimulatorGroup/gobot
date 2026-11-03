"""Compute independent stride/cadence ranges from held-out human motion clips."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from gobot.rl.locomotion.gait_metrics import BipedGaitMetrics


def build_reference(path):
    windows = []
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        dt = metadata['dt']
        for i, clip in enumerate(metadata['clips']):
            if clip['split'] != 'holdout': continue
            root = data[f'root_{i}']; features = data[f'features_{i}']
            local = features[:, -12:].reshape(-1, 4, 3)[:, :2]
            rotation = Rotation.from_quat(root[:, [4, 5, 6, 3]]).as_matrix()
            feet = np.einsum('tij,tfj->tfi', rotation, local) + root[:, None, :3]
            velocity = np.gradient(feet, dt, axis=0)
            floor = np.percentile(feet[:, :, 2], 5, axis=0)
            contact = (feet[:, :, 2] < floor + .03) & (np.linalg.norm(velocity[:, :, :2], axis=2) < .4)
            contact &= np.abs(velocity[:, :, 2]) < .3
            for start in range(0, len(root) - 300, 150):
                end = start + 300
                if not data[f'valid_{i}'][start:end].all(): continue
                speed = float(np.linalg.norm(root[start:end, 7:9], axis=1).mean())
                if speed < .2: continue
                accumulator = BipedGaitMetrics(1, dt, warmup_s=0.)
                for t in range(start, end): accumulator.update(contact[t:t+1], feet[t:t+1], [True])
                values = accumulator.arrays()
                if values['landings'][0] < 4: continue
                windows.append({'clip': clip['name'], 'speed': speed, 'gait': clip['gait'],
                                **{key: value[0].tolist() for key, value in values.items()}})
    ranges = []
    for gait in (0, 1):
        for speed in np.arange(.4, 2.41, .2):
            chosen = [w for w in windows if w['gait'] == gait and abs(w['speed'] - speed) <= .25]
            if len(chosen) < 5: continue
            entry = {'gait': gait, 'speed': round(float(speed), 2), 'windows': len(chosen)}
            for key in ('cadence_hz', 'stride_length_m', 'flight_fraction'):
                lo, hi = np.percentile([w[key] for w in chosen], [5, 95])
                entry[key] = [float(lo * .8), float(hi * 1.2)]
            ranges.append(entry)
    return {'source': metadata['source'], 'protocol': 'heldout_subject5_6s_windows_3s_stride_kinematic_contacts',
            'contact_height_tolerance_m': .03, 'contact_speed_threshold_m_s': .4, 'ranges': ranges, 'windows': windows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('examples/g1/assets/motions/locomotion.npz'))
    p.add_argument('--output', type=Path, default=Path('examples/g1/assets/motions/heldout_gaits.json'))
    args = p.parse_args()
    result = build_reference(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(result["windows"])} held-out gait windows, {len(result["ranges"])} speed/gait ranges')

if __name__ == '__main__': main()


def check_natural_gait(result, reference):
    """Validate flat forward gait against held-out data at a comparable speed."""
    speed = float(result['command'][0])
    gait = int(speed >= 1.6)
    candidates = [row for row in reference['ranges'] if row['gait'] == gait and abs(row['speed'] - speed) <= .25]
    if not candidates:
        return {'passed': False, 'supported': False, 'reason': 'no held-out reference speed coverage'}
    target = min(candidates, key=lambda row: abs(row['speed'] - speed))
    checks = {}
    for key in ('cadence_hz', 'stride_length_m'):
        value = float(result['gait'][key])
        lower, upper = target[key]
        checks[key] = {'value': value, 'range': [lower, upper], 'passed': lower <= value <= upper}
    return {'passed': all(check['passed'] for check in checks.values()), 'supported': True,
            'reference_speed': target['speed'], 'reference_windows': target['windows'], 'checks': checks}
