"""Create a checksummed source overlay and runtime assets for a GPU training host."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile
from .experiment import Experiment


SOURCE_SUFFIXES = {'.py', '.pyi', '.cpp', '.hpp', '.c', '.h', '.cmake', '.in', '.txt',
                   '.jscn', '.jres', '.gobot', '.stl', '.obj', '.dae', '.png', '.mtl',
                   '.ply', '.json', '.xml', '.urdf', '.yaml', '.md', '.sh', '.sbatch'}


def source_files(root):
    """Explicit project roots, including uncommitted native/provider changes.

    Never sweep the user's home or caches; generated binaries and logs stay out.
    Large data/checkpoints are selected separately by the requested experiment.
    """
    files=set()
    for name in ('python/gobot', 'python/gobot_cli', 'src/gobot', 'include/gobot', 'cmake',
                 'tests', 'examples/g1', 'examples/go1', 'examples/locomotion'):
        for path in (root/name).rglob('*'):
            if not path.is_file() or any(p in {'logs','__pycache__','policies','evaluation','.venv'}
                                         for p in path.relative_to(root).parts):
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            if (path.suffix.lower() in SOURCE_SUFFIXES or path.name in {'LICENSE','NOTICE'}
                    or path.name.startswith('LICENSE-')):
                files.add(path)
    for name in ('.gitignore','CMakeLists.txt','gobot_build_backend.py','pyproject.toml','uv.lock',
                 'doc/g1_amp_training_status.md',
                 'doc/gobot_training.md',
                 'doc/supermimic_reproduced_alignment_plan.md','doc/supermimic_reproduced_progress.md'):
        files.add(root/name)
    if (root/'LICENSE').exists():files.add(root/'LICENSE')
    for fixture in (root/'tests/fixtures').glob('supermimic_reproduced_*.pt'):
        files.add(fixture)
    return files


def build(output, configs, checkpoint=None, *, reports=()):
    root=Path(__file__).resolve().parents[2]
    files=source_files(root)
    robots=set()
    for config in configs:
        config=Path(config).resolve();files.add(config)
        exp=Experiment.load(config);robots.add(exp.robot)
        if exp.robot=='g1' and exp.recipe=='natural':
            dataset=(root/exp.motion_dataset).resolve();files.add(dataset);files.add(dataset.parent/'source.json')
    if checkpoint:
        files.add(Path(checkpoint).resolve())
        sidecar=Path(str(checkpoint)+'.manifest.json').resolve()
        if sidecar.exists():files.add(sidecar)
    for report in reports:
        report=Path(report).resolve()
        if report.suffix!='.json':raise ValueError('diagnostic reports must be JSON files')
        files.add(report)
    paths={}
    for path in sorted(files):
        if not path.exists():raise FileNotFoundError(path)
        if not path.resolve().is_relative_to(root.resolve()):raise ValueError(f'bundle file resolves outside project: {path}')
        relative=str(path.relative_to(root)) # Prevent accidental inclusion of arbitrary outside files.
        paths[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    versions={p:importlib.metadata.version(p) for p in ('torch','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','numpy','tensordict')}
    metadata={'schema_version':2,'purpose':'training_experiment',
              'reports':[str(Path(p).resolve().relative_to(root)) for p in reports],'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'python':platform.python_version(),'packages':versions,'robots':sorted(robots),'sha256':paths,
              'configs':[str(Path(p).resolve().relative_to(root)) for p in configs],
              'checkpoint':str(Path(checkpoint).resolve().relative_to(root)) if checkpoint else None,
              'execution':'one independent process per visible GPU; not synchronized data parallel AMP'}
    readme='''Gobot locomotion training overlay

1. Check out RobSimulatorGroup/gobot at the base_commit in bundle.json.
2. Extract this archive over that checkout (it contains the exact working source and assets).
3. Install/build Gobot using uv sync --frozen and the repository build guide.
4. Verify files: python -m examples.locomotion.package --verify bundle.json
5. Start one independent process per GPU. Example:
   CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m examples.locomotion.train --config CONFIG --log-dir logs/RUN --budget-file logs/RUN-budget.json
7. Use --resume CHECKPOINT for the same recipe, --initialize CHECKPOINT for an admitted next stage.
8. A cluster budget must be explicit in the experiment config. For concurrent independent jobs use separate ledgers; sum their GPU-hours in the campaign report.

Logs/status.json includes PID identity, heartbeat and exit reason. Inspect via:
   uv run --no-sync python -m examples.locomotion.status logs

Native Gobot must be built for the destination Python/platform; local build binaries are not copied.
'''
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz',dereference=True) as archive:
        for name in paths:archive.add(root/name,arcname=name,recursive=False)
        for name,content in [('bundle.json',json.dumps(metadata,indent=2)+'\n'),('BUNDLE_README.txt',readme)]:
            content=content.encode();info=tarfile.TarInfo(name);info.size=len(content);archive.addfile(info,io.BytesIO(content))
    with tarfile.open(output) as archive:
        for name,digest in paths.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
    print(json.dumps({'output':str(output),'files':len(paths),'bytes':output.stat().st_size,'verified':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--config',type=Path,action='append');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--report',type=Path,action='append',default=[])
    p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:
        metadata=json.loads(a.verify.read_text());root=a.verify.parent
        for name,digest in metadata['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'checksum mismatch: {name}')
        print('All bundle checksums verified')
    elif a.output and a.config:build(a.output,a.config,a.checkpoint,reports=a.report)
    else:p.error('provide --output and --config, or --verify')

if __name__=='__main__':main()
