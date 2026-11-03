"""Prepare an isolated, explicitly patched reference environment for 3.17 parity.

Uses already installed pinned packages. The original 3.11 environment and source
checkout remain intact; every compatibility source change is recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import venv

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'examples/locomotion/logs/supermimic_reference'


def prepare(destination=BASE/'venv_317'):
    destination=Path(destination).resolve()
    venv.EnvBuilder(with_pip=False,symlinks=True).create(destination)
    python_version=f'python{sys.version_info.major}.{sys.version_info.minor}'
    target=destination/'lib'/python_version/'site-packages'
    original=BASE/'.venv/lib'/python_version/'site-packages'
    shared=ROOT/'.venv/lib'/python_version/'site-packages'
    source=BASE/'Uni-SuperMimic/src'
    package=target/'unisim'
    if package.exists():
        raise FileExistsError('compatibility overlay already exists; use a new isolated directory')
    shutil.copytree(original/'unisim',package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    dependency=package/'backend/mjwarp/dependencies.py'
    before=dependency.read_text()
    anchor='_REQUIRED_MUJOCO_WARP_LINE = "3.11"'
    if before.count(anchor)!=1:
        raise ValueError('reference dependency boundary changed; review compatibility patch')
    after=before.replace(anchor,'_REQUIRED_MUJOCO_WARP_LINE = "3.17"')
    dependency.write_text(after)
    (target/'reference_dependencies.pth').write_text(''.join(str(p)+'\n' for p in (shared,original,source)))
    report={'source_environment':str(original),'shared_environment':str(shared),'expected_physics_version':'3.17.0',
            'patches':[{'path':'unisim/backend/mjwarp/dependencies.py','reason':'explicitly test 3.17 API compatibility',
            'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),
            'replacement':{'before':anchor,'after':'_REQUIRED_MUJOCO_WARP_LINE = "3.17"'}}]}
    (destination/'compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def patch_installed(destination):
    """Patch a freshly installed cluster reference environment, not the 3.11 baseline."""
    destination=Path(destination).resolve()
    if destination==(BASE/'.venv').resolve():
        raise ValueError('do not patch the frozen 3.11 reference environment')
    candidates=list(destination.glob('lib/python*/site-packages/unisim/backend/mjwarp/dependencies.py'))
    if len(candidates)!=1:raise ValueError('expected one installed unisim-core package in the isolated environment')
    path=candidates[0]
    if not path.resolve().is_relative_to(destination):
        raise ValueError('reference dependency must be installed inside the isolated environment')
    before=path.read_text()
    expected='cac0811335f0d39583aa192beefaad97a3a094934919e544cd2fcb0be7a910f3'
    if hashlib.sha256(before.encode()).hexdigest()!=expected:
        raise ValueError('unisim-core dependency source differs from the pinned compatibility baseline')
    after=before.replace('_REQUIRED_MUJOCO_WARP_LINE = "3.11"','_REQUIRED_MUJOCO_WARP_LINE = "3.17"')
    temporary=path.with_suffix('.compat-tmp')
    temporary.write_text(after);temporary.replace(path)
    report={'expected_physics_version':'3.17.0','patches':[{'path':'unisim/backend/mjwarp/dependencies.py',
        'before_sha256':expected,'after_sha256':hashlib.sha256(after.encode()).hexdigest(),
        'reason':'explicit 3.17 compatibility evaluation'}]}
    (destination/'compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'venv_317')
    parser.add_argument('--patch-installed',action='store_true')
    args=parser.parse_args()
    print(json.dumps((patch_installed if args.patch_installed else prepare)(args.output),indent=2))
