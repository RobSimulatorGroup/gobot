"""Portable training terrain compiled from a Gobot scene; no engine imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def digest(data):
    content = {key: value for key, value in data.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def from_environment(env):
    source = ET.fromstring(env.compiled_artifact.content)
    root = ET.Element('mujoco')
    assets = ET.SubElement(root, 'asset')
    body = ET.SubElement(ET.SubElement(root, 'worldbody'), 'body', name='terrain')
    for field in source.findall('./asset/hfield'):
        assets.append(deepcopy(field))
    for geometry in source.findall('./worldbody/geom'):
        geometry = deepcopy(geometry)
        geometry.set('group', '0')
        body.append(geometry)
    if not len(body):
        raise ValueError('training terrain has no collision geometry')
    config = env._terrain_config
    rows, cols = env.terrain_grid_shape
    origins = env.terrain_origins
    if hasattr(origins, 'detach'):
        origins = origins.detach().cpu().numpy()
    origins = np.asarray(origins, dtype=np.float32).reshape(rows, cols, 3)
    data = {'seed': int(config['seed']), 'grid_shape': [rows, cols],
            'size': list(map(float, config['patch_size'])), 'border_width': float(config['border_width']),
            'names': list(env.terrain_type_names),
            'proportions': [float(t['proportion']) for t in config['sub_terrains']],
            'max_init_level': env.cfg_obj.max_init_terrain_level,
            'origins': origins.tolist(), 'xml': ET.tostring(root, encoding='unicode')}
    data['sha256'] = digest(data)
    return data


def load(path):
    data = json.loads(Path(path).read_text())
    if data.get('sha256') != digest(data):
        raise ValueError('training terrain checksum does not match its contents')
    rows, cols = data['grid_shape']
    origins = np.asarray(data['origins'])
    proportions = np.asarray(data['proportions'])
    if (rows < 1 or cols < 1 or origins.shape != (rows, cols, 3) or not np.isfinite(origins).all()
            or len(data['names']) != cols or len(proportions) != cols
            or not np.isfinite(proportions).all() or (proportions < 0).any() or proportions.sum() <= 0):
        raise ValueError('invalid training terrain grid or proportions')
    return data


def assignment_digest(levels, types):
    return hashlib.sha256(np.stack((levels, types), 1).astype('<i8').tobytes()).hexdigest()


def verify_environment(env, data):
    actual = from_environment(env)
    if actual['sha256'] != data['sha256']:
        raise ValueError('scene terrain differs from --terrain-reference; export the current authored terrain again')


def add_to_spec(spec, data):
    """Install only the exported terrain geometry in a MuJoCo terrain entity."""
    import mujoco
    source = mujoco.MjSpec.from_string(data['xml'])
    body = spec.worldbody.add_body(name='terrain')
    for field in source.hfields:
        spec.add_hfield(name=field.name, nrow=field.nrow, ncol=field.ncol, size=field.size, userdata=field.userdata)
    fields = ('name', 'type', 'size', 'pos', 'quat', 'contype', 'conaffinity', 'condim',
              'priority', 'friction', 'solref', 'solimp', 'solmix', 'margin', 'gap', 'hfieldname')
    for geometry in source.geoms:
        body.add_geom(**{key: getattr(geometry, key) for key in fields}, group=0, mass=0)
