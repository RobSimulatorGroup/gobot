"""Convert an existing audited BONES-SEED compact library to official packed motions.

No MJCF, FK or interpolation is performed. Copies the exact 71D fp16 source
values into six named fields with clip boundaries and provenance, usable by both
Gobot and the pinned official mjlab_research MotionLoader.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from gobot.rl.supermimic.official.motion import FIELDS, SLICES, PROTOCOL, digest, inspect_library


def prepare(source, profile, output, *, clips=None, limit=None, chunk_frames=65536):
    source, profile, output = Path(source).resolve(), Path(profile).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'output already exists: {output}; use verify to reuse it')
    if chunk_frames < 1 or (limit is not None and limit < 1):
        raise ValueError('chunk_frames and limit must be positive')
    metadata = json.loads(profile.read_text())
    joints = metadata['joint_names']
    if len(joints) != 29 or len(set(joints)) != 29:
        raise ValueError('source profile must name all 29 joints in library order')
    pack = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    state, names = pack['ref'], list(pack['names'])
    starts, lengths = np.asarray(pack['starts'], dtype=np.int64), np.asarray(pack['lengths'], dtype=np.int64)
    if (state.ndim != 2 or state.shape[1] != 71 or state.dtype not in (torch.float16,torch.float32)
            or len(names) != len(set(names)) or len(starts) != len(names) or not len(names)
            or lengths.shape != starts.shape or np.any(lengths < 3)
            or not np.array_equal(starts, np.r_[0,np.cumsum(lengths)[:-1]])
            or int(lengths.sum()) != state.shape[0]):
        raise ValueError('source compact library has invalid state/clip layout')
    if clips is not None:
        if len(clips) != len(set(clips)) or not set(clips) <= set(names):
            raise ValueError('requested clips contain duplicates or unknown names')
        selected = [i for i,name in enumerate(names) if name in set(clips)]
    else:
        selected = list(range(len(names)))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError('clip selection is empty')
    frames = int(lengths[selected].sum())
    # Fingerprint before and after streaming so a changing source is never published.
    source_hash, profile_hash = digest(source), digest(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'.partial-', dir=output.parent))
    try:
        dtype = np.float16 if state.dtype == torch.float16 else np.float32
        fields = {name: np.lib.format.open_memmap(stage/f'{name}.npy', mode='w+',
                  dtype=dtype, shape=(frames,width)) for name,width in FIELDS.items()}
        cursor = 0
        for i in selected:
            start, length = int(starts[i]), int(lengths[i])
            for offset in range(0,length,chunk_frames):
                block = state[start+offset:start+min(length,offset+chunk_frames)].numpy()
                if not np.isfinite(block).all() or np.any(np.linalg.norm(block[:,3:7].astype(np.float32),axis=-1)<1e-8):
                    raise ValueError(f'nonfinite or invalid quaternion in source clip {names[i]}')
                for name,(lo,hi) in SLICES.items():
                    fields[name][cursor+offset:cursor+offset+len(block)] = block[:,lo:hi]
            cursor += length
        for array in fields.values():
            array.flush()
        fields.clear()
        np.save(stage/'motion_ends.npy', np.cumsum(lengths[selected], dtype=np.int64))
        (stage/'motion_names.json').write_text(json.dumps([names[i] for i in selected])+'\n')
        (stage/'joint_names.json').write_text(json.dumps(joints)+'\n')
        if digest(source) != source_hash or digest(profile) != profile_hash:
            raise ValueError('source changed while preparing motions')
        manifest = {'protocol':PROTOCOL, 'complete':True, 'fps':50., 'quaternion_order':'wxyz',
            'root_velocity_frame':'world', 'frames':frames,'clips':len(selected),'joint_names':joints,
            'source':{'path':str(source),'sha256':source_hash,'dtype':str(state.dtype),
                      'profile_sha256':profile_hash,'profile':str(profile),
                      'reference_commit':metadata.get('reference_commit')},
            'conversion':'exact field split; source quantization preserved; no FK/resampling',
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(stage.iterdir())}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        inspect_library(stage,verify_hashes=True)
        os.rename(stage,output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    create = sub.add_parser('prepare')
    create.add_argument('--source',type=Path,required=True)
    create.add_argument('--profile',type=Path,required=True)
    create.add_argument('--output',type=Path,required=True)
    create.add_argument('--clips',nargs='+')
    create.add_argument('--limit',type=int)
    verify = sub.add_parser('verify')
    verify.add_argument('--data',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        result = inspect_library(args.data,verify_hashes=True)
    else:
        result = prepare(args.source,args.profile,args.output,clips=args.clips,limit=args.limit)
    print(json.dumps({k:result[k] for k in ('protocol','complete','clips','frames','source')},indent=2))

if __name__ == '__main__':
    main()
