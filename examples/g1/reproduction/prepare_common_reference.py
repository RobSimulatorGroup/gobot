"""Freeze original dense FP32 BONES states for a shared policy benchmark.

Each NPZ is checked against the historical audited manifest before extraction.
No fp16 roundtrip, resampling or FK is performed. Reference body transforms are
computed later by the same scene FK for both policies, from these exact states.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from gobot.rl.supermimic.official.motion import FIELDS, PROTOCOL, digest, inspect_library


def prepare(data, profile, output, *, workers=4, limit=None):
    data, profile, output = map(lambda p: Path(p).resolve(), (data, profile, output))
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError('workers/limit must be positive')
    source_manifest = data/'manifest.json'
    source_hash = digest(source_manifest)
    profile_hash = digest(profile)
    source = json.loads(source_manifest.read_text())
    if not source['complete'] or not source['ready_for_paired_training']:
        raise ValueError('source motion conversion is not audited')
    for name in ('manifest_all.jsonl', 'train_clips.json', 'eval_clips.json'):
        if digest(data/name) != source['artifacts'][name]:
            raise ValueError(f'source manifest changed: {name}')
    rows = [row for row in map(json.loads, (data/'manifest_all.jsonl').read_text().splitlines()) if row['keep']]
    if [r['name'] for r in rows] != json.loads((data/'train_clips.json').read_text()):
        raise ValueError('source clip membership differs')
    if limit is not None:
        rows = rows[:limit]
    joints = json.loads(profile.read_text())['joint_names']
    if len(joints) != 29 or len(set(joints)) != 29:
        raise ValueError('profile needs 29 unique joint names')
    names = [r['name'] for r in rows]
    if not names or len(names) != len(set(names)) or any(Path(n).name != n for n in names):
        raise ValueError('invalid source clip names')
    lengths = np.asarray([r['frames'] for r in rows], dtype=np.int64)
    if np.any(lengths < 3):
        raise ValueError('clip has fewer than three frames')
    total = int(lengths.sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+'.partial-', dir=output.parent))
    try:
        arrays = {k: np.lib.format.open_memmap(stage/f'{k}.npy', mode='w+', dtype=np.float32,
                  shape=(total, width)) for k, width in FIELDS.items()}
        def load(row):
            import hashlib
            # Verify the very bytes being decoded, avoiding hash/read races.
            raw = (data/'npz'/(row['name']+'.npz')).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['npz_sha256']:
                raise ValueError(f'NPZ hash changed: {row["name"]}')
            with np.load(io.BytesIO(raw), allow_pickle=False) as record:
                if not np.allclose(record['fps'], 50.):
                    raise ValueError('source must already be 50 Hz')
                bodies = [str(n) for n in record['body_names']]
                root = bodies.index('pelvis')
                if len(bodies) != len(set(bodies)):
                    raise ValueError('duplicate source bodies')
                if 'joint_names' in record and list(record['joint_names']) != joints:
                    raise ValueError('source joint order differs')
                values = {'root_pos_w': record['body_pos_w'][:, root],
                          'root_quat_w': record['body_quat_w'][:, root],
                          'root_lin_vel_w': record['body_lin_vel_w'][:, root],
                          'root_ang_vel_w': record['body_ang_vel_w'][:, root],
                          'joint_pos': record['joint_pos'], 'joint_vel': record['joint_vel']}
                for key, value in values.items():
                    if value.dtype != np.float32 or value.shape != (row['frames'], FIELDS[key]) or not np.isfinite(value).all():
                        raise ValueError(f'invalid FP32 field {row["name"]}/{key}')
                if np.any(np.linalg.norm(values['root_quat_w'], axis=-1) < 1e-8):
                    raise ValueError('invalid source quaternion')
                return values
        cursor = 0
        start = time.monotonic()
        # Bounded windows: Executor.map on all 130k records can retain huge decoded results.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for first in range(0, len(rows), 128):
                batch = rows[first:first+128]
                for row, values in zip(batch, pool.map(load, batch), strict=True):
                    for key in FIELDS:
                        arrays[key][cursor:cursor+row['frames']] = values[key]
                    cursor += row['frames']
                if first % 1024 == 0 or first+len(batch) == len(rows):
                    print(json.dumps({'stage':'packing_fp32','clips':first+len(batch),
                        'total_clips':len(rows),'frames':cursor,'elapsed_seconds':time.monotonic()-start}), flush=True)
        for value in arrays.values():value.flush()
        arrays.clear()
        np.save(stage/'motion_ends.npy', np.cumsum(lengths))
        (stage/'motion_names.json').write_text(json.dumps(names)+'\n')
        (stage/'joint_names.json').write_text(json.dumps(joints)+'\n')
        if digest(source_manifest) != source_hash or digest(profile) != profile_hash:
            raise ValueError('source identity changed during packing')
        manifest = {'protocol':PROTOCOL,'complete':True,'fps':50.,'quaternion_order':'wxyz',
            'root_velocity_frame':'world','frames':total,'clips':len(rows),'joint_names':joints,
            'source':{'path':str(data),'manifest_sha256':source_hash,
                      'artifacts':source['artifacts'],'profile_sha256':profile_hash,
                      'dtype':'float32','every_npz_hash_verified':True},
            'conversion':'original dense FP32 root and joints; no fp16, no resampling, no FK',
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(stage.iterdir())}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        inspect_library(stage, verify_hashes=True)
        os.rename(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({k:result[k] for k in ('clips','frames','source','conversion')},indent=2))


if __name__ == '__main__':main()
