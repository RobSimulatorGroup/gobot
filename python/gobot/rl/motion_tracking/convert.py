"""Convert G1 retarget CSVs using reference resampling and scene-owned FK."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import gobot
from gobot.rl.kinematics import SceneKinematics
from gobot.rl.supermimic.motion_convert import G1_JOINT_NAMES, load_csv, resample_motion
from gobot.rl.supermimic.rotations import quat_apply_inverse
from .kinematics import SceneStateKinematics


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def convert(project, sources, output, *, input_fps=30., scene='res://g1_29dof_flat_scene.jscn', compact=False):
    project, output = Path(project), Path(output)
    sources = list(sources)
    if not sources or len({Path(p).stem for p in sources}) != len(sources):
        raise ValueError('conversion requires unique, nonempty clip names')
    output.mkdir(parents=True, exist_ok=True)
    profile = json.loads((project/'assets/supermimic/robot_profile.json').read_text())
    if tuple(profile['joint_names']) != G1_JOINT_NAMES:
        raise ValueError('CSV and scene joint orders differ')
    ctx = gobot.app.create_context()
    try:
        ctx.set_project_path(str(project.resolve()))
        ctx.load_scene(scene)
        kin = SceneKinematics(ctx.compile_physics_scene_snapshot(), robot_name=profile['robot_name'],
            base_link=profile['base_link'], joint_names=G1_JOINT_NAMES, device='cpu', dtype=torch.float64)
        fk = SceneStateKinematics(kin)
        manifest = {'protocol':'supermimic_reproduced_scene_motion_v1', 'kinematics_digest':kin.digest,
                    'reference_commit':profile['reference_commit'], 'input_fps':input_fps, 'output_fps':50., 'clips':[]}
        refs, starts, lengths = [], [], []
        for source in sources:
            source = Path(source)
            raw = load_csv(source)
            if len(raw) < 3 or not np.isfinite(raw).all() or not np.all(np.linalg.norm(raw[:,3:7],axis=-1)>1e-8):
                raise ValueError(f'invalid motion {source}')
            m = resample_motion(raw, input_fps, 50.)
            q = torch.from_numpy(np.concatenate((m.root_pos,m.root_quat,m.joint_pos),-1))
            angular = quat_apply_inverse(q[:,3:7],torch.from_numpy(m.root_ang_vel_w))
            v = torch.cat((torch.from_numpy(m.root_lin_vel_w), angular, torch.from_numpy(m.joint_vel)),-1)
            target = output/(source.stem+'.npz')
            with torch.no_grad():
                chunks = [fk(q[sl:sl+4096],v[sl:sl+4096]) for sl in range(0,len(q),4096)]
                body = {key:torch.cat([chunk[key] for chunk in chunks]).float().numpy() for key in chunks[0]}
            np.savez(target, fps=np.array([50.],dtype=np.float32), joint_pos=m.joint_pos.astype('float32'),
                joint_vel=m.joint_vel.astype('float32'), body_pos_w=body['pos'],body_quat_w=body['quat'],
                body_lin_vel_w=body['lin_vel'],body_ang_vel_w=body['ang_vel'], joint_names=np.array(G1_JOINT_NAMES),
                body_names=np.array(kin.body_names),source=np.array(str(source)),input_fps=np.array([input_fps],dtype=np.float32))
            manifest['clips'].append({'name':source.stem,'source':str(source.resolve()),'source_sha256':digest(source),
                'npz_sha256':digest(target),'frames':m.num_frames})
            if compact:
                starts.append(sum(lengths)); lengths.append(m.num_frames)
                root_index = kin.body_names.index(profile['base_link'])
                refs.append(torch.from_numpy(np.concatenate((body['pos'][:,root_index],body['quat'][:,root_index],
                    m.joint_pos.astype('float32'),body['lin_vel'][:,root_index],body['ang_vel'][:,root_index],
                    m.joint_vel.astype('float32')),-1)).half())
        if compact:
            pack = output/'library.pt'
            torch.save({'ref':torch.cat(refs), 'names':[c['name'] for c in manifest['clips']],
                        'starts':starts, 'lengths':lengths}, pack)
            manifest['library_sha256'] = digest(pack)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
    finally:
        ctx.clear_scene()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-fps',type=float,default=30.)
    p.add_argument('--compact',action='store_true')
    a = p.parse_args()
    print(json.dumps(convert(a.project,a.input,a.output,input_fps=a.input_fps,compact=a.compact),indent=2))

if __name__=='__main__':
    main()
