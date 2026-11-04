"""Compare the published actor in two separate processes, on identical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PosixPath

from .reference import LOCK, digest


def load_release(path):
    import torch
    lock = json.loads(LOCK.read_text())
    record = next(row for row in lock['release']['assets'] if row['name'].endswith('.pt'))
    if digest(path) != record['sha256']:
        raise ValueError('checkpoint does not match the pinned release')
    # The upstream config contains Paths. No arbitrary pickle constructors are enabled.
    with torch.serialization.safe_globals([PosixPath, (PosixPath, 'pathlib.PosixPath')]):
        payload = torch.load(path, map_location='cpu', weights_only=True)
    return payload, record['sha256']


def run(mode, checkpoint, output, *, reference=None, mjlab=None, fixture=None):
    import torch
    torch.set_num_threads(2)
    payload, sha256 = load_release(checkpoint)
    if mode == 'reference':
        from .math_fixture import numerical_packages
        numerical_packages(reference, mjlab)
        from supermimic.policy.nets import Policy, PolicyCfg
    else:
        from gobot.rl.supermimic.policy import Policy, PolicyCfg
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in payload['config']['policy'].items()}
    model = Policy(PolicyCfg(**options)).eval()
    model.load_state_dict(payload['policy'], strict=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if mode == 'reference':
            torch.manual_seed(1701)
            obs = torch.randn(16, 930)
            goals = torch.randn(16, options['goal_dim'])
            values = model(obs, goals)
            torch.save({'observations': obs, 'goals': goals, 'actions': values,
                        'checkpoint_sha256': sha256}, output)
            result = {'protocol': 'e8_actor_reference_v1', 'checkpoint_sha256': sha256,
                      'iteration': payload['iteration'], 'num_updates': payload['num_updates'],
                      'torch': torch.__version__, 'scope': 'fixed-input actor only'}
            output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
        else:
            expected = torch.load(fixture, weights_only=True)
            if expected['checkpoint_sha256'] != sha256:
                raise ValueError('reference output uses a different checkpoint')
            actual = model(expected['observations'], expected['goals'])
            torch.testing.assert_close(actual, expected['actions'], atol=1e-6, rtol=1e-5)
            result = {'protocol': 'e8_actor_parity_v1', 'checkpoint_sha256': sha256,
                      'max_abs_action_error': float((actual-expected['actions']).abs().max()),
                      'passed': True, 'torch': torch.__version__,
                      'scope': 'fixed-input actor only; not rollout or training parity'}
            output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'candidate'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--mjlab', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'reference' and (args.reference is None or args.mjlab is None):
        parser.error('reference mode requires both checkouts')
    if args.mode == 'candidate' and args.fixture is None:
        parser.error('candidate mode requires the independent reference fixture')
    run(args.mode, args.checkpoint, args.output, reference=args.reference, mjlab=args.mjlab, fixture=args.fixture)
