"""Package the pinned OpenArm v2 MJCF arms with the existing Sharpa hands.

This example-local conversion preserves arm frames, axes, masses and limits.
It is not a general MJCF importer. Rebuild from vendored inputs without network.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

HERE = Path(__file__).resolve().parent
SHARPA = HERE.parent / 'sharpa'
sys.path.insert(0, str(SHARPA))
from prepare_assets import collision_hull

MOUNT_HEIGHT = 1.29
PEDESTAL_HEIGHT = .697
MOUNT_FORWARD = .26


def numbers(text):
    return np.fromstring(text, sep=' ')


def encoded(values):
    return ' '.join(f'{v:.12g}' for v in values)


def origin(owner, position=(0., 0., 0.), rotation=None):
    ET.SubElement(owner, 'origin', xyz=encoded(position), rpy=encoded(
        np.zeros(3) if rotation is None else Rotation.from_matrix(rotation).as_euler('xyz')))


def mj_rotation(element):
    if element.get('quat') is not None:
        return Rotation.from_quat(numbers(element.get('quat'))[[1, 2, 3, 0]]).as_matrix()
    return Rotation.from_euler('xyz', numbers(element.get('euler', '0 0 0'))).as_matrix()


def descendants(root, parent):
    names = {parent}
    for _ in root.findall('joint'):
        for joint in root.findall('joint'):
            if joint.find('parent').get('link') in names:
                names.add(joint.find('child').get('link'))
    return names


def write_xml(root, path):
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def prepare(source=None):
    source_dir = HERE / 'source'
    if source is not None:
        shutil.copytree(source/'v2/assets', source_dir/'assets', dirs_exist_ok=True)
        shutil.copy2(source/'v2/openarm_bimanual.xml', source_dir/'openarm_bimanual.xml')
        shutil.copy2(source/'v2/pedestal/openarm_pedestal.xml', source_dir/'openarm_pedestal.xml')
        for name in ('LICENSE', 'README.md'):
            (source_dir/name).write_bytes(subprocess.check_output(['git', '-C', str(source), 'show', 'HEAD:'+name]))
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        hashes = {str(p.relative_to(source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source_dir.rglob('*')) if p.is_file() and p.name != 'hashes.json'}
        (source_dir/'hashes.json').write_text(json.dumps(dict(revision=revision, files=hashes), indent=2)+'\n')
    mjcf = ET.parse(source_dir/'openarm_bimanual.xml').getroot()
    root = ET.Element('robot', name='openarm_sharpa')
    root.append(ET.Comment('Derived from enactic/openarm_mujoco v2; see SOURCE.md and source/LICENSE.'))
    assets = {a.get('name'): a for a in mjcf.find('asset')}
    materials = {a.get('name'): a.get('rgba') for a in mjcf.findall('asset/material')}
    for folder in ('visual', 'collision'):
        (HERE/folder).mkdir(exist_ok=True)

    def mesh_geometry(owner, mesh_spec, is_collision):
        mesh = trimesh.load(source_dir/'assets'/mesh_spec.get('file'), force='mesh', process=False)
        scale = numbers(mesh_spec.get('scale', '1 1 1'))
        matrix = np.diag(np.r_[scale, 1.])
        mesh.apply_transform(matrix)  # Also reverses winding for a reflection.
        target = HERE/('collision' if is_collision else 'visual')/(mesh_spec.get('name')+'.obj')
        if is_collision:
            mesh = collision_hull(np.unique(np.round(mesh.vertices, 5), axis=0))
        mesh.export(target, include_color=False, include_texture=False)
        geometry = ET.SubElement(owner, 'geometry')
        ET.SubElement(geometry, 'mesh', filename=str(target.relative_to(HERE)))

    # Keep the upstream pedestal at its original scale, on a fixed riser. The
    # arm mount height is authored for the existing 85 cm conveyor worktop.
    base = ET.SubElement(root, 'link', name='base_link')
    riser_height = MOUNT_HEIGHT-PEDESTAL_HEIGHT
    for kind in ('visual', 'collision'):
        element = ET.SubElement(base, kind)
        origin(element, (0., 0., riser_height/2))
        ET.SubElement(ET.SubElement(element, 'geometry'), 'box', size=encoded((.32, .32, riser_height)))
        if kind == 'visual':
            ET.SubElement(ET.SubElement(element, 'material', name='riser'), 'color', rgba='.18 .20 .22 1')
        spec = ET.Element('mesh', name='pedestal', file='collision/body_link0_symp.stl' if kind == 'collision'
                          else 'visual/body/body_link0.stl', scale='.001 .001 .001')
        element = ET.SubElement(base, kind)
        origin(element, (0., 0., riser_height))
        mesh_geometry(element, spec, kind == 'collision')
        if kind == 'visual':
            ET.SubElement(ET.SubElement(element, 'material', name='pedestal'), 'color', rgba='.247 .247 .247 1')
    # A fixed bracket brings the compact arms over the rear edge of the table.
    mount = ET.SubElement(root, 'link', name='openarm_mount')
    inertia = ET.SubElement(mount, 'inertial')
    origin(inertia, (-MOUNT_FORWARD/2, 0., -.04))
    ET.SubElement(inertia, 'mass', value='1.2')
    ET.SubElement(inertia, 'inertia', ixx='.00164', iyy='.0122', izz='.01256', ixy='0', ixz='0', iyz='0')
    for kind in ('visual', 'collision'):
        element = ET.SubElement(mount, kind)
        origin(element, (-MOUNT_FORWARD/2, 0., -.04))
        ET.SubElement(ET.SubElement(element, 'geometry'), 'box', size=encoded((MOUNT_FORWARD+.08, .10, .08)))
        if kind == 'visual':
            ET.SubElement(ET.SubElement(element, 'material', name='bracket'), 'color', rgba='.18 .20 .22 1')
    joint = ET.SubElement(root, 'joint', name='openarm_mount_joint', type='fixed')
    origin(joint, (MOUNT_FORWARD, 0., MOUNT_HEIGHT))
    ET.SubElement(joint, 'parent', link='base_link'); ET.SubElement(joint, 'child', link='openarm_mount')

    def body_link(body, parent):
        name = body.get('name')
        if 'finger' in name:
            return  # Replace the stock fingers, retaining the seventh wrist motor.
        link = ET.SubElement(root, 'link', name=name)
        inertial = body.find('inertial')
        if inertial is not None:
            out = ET.SubElement(link, 'inertial')
            origin(out, numbers(inertial.get('pos')), mj_rotation(inertial))
            ET.SubElement(out, 'mass', value=inertial.get('mass'))
            inertia = numbers(inertial.get('diaginertia'))
            ET.SubElement(out, 'inertia', ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]),
                          ixy='0', ixz='0', iyz='0')
        for geom in body.findall('geom'):
            collision = geom.get('class') == 'collision'
            out = ET.SubElement(link, 'collision' if collision else 'visual')
            origin(out, numbers(geom.get('pos', '0 0 0')), mj_rotation(geom))
            mesh_geometry(out, assets[geom.get('mesh')], collision)
            if not collision:
                ET.SubElement(ET.SubElement(out, 'material', name=geom.get('material')),
                              'color', rgba=materials[geom.get('material')])
        source_joint = body.find('joint')
        joint = ET.SubElement(root, 'joint', name=source_joint.get('name') if source_joint is not None else name+'_mount',
                              type='revolute' if source_joint is not None else 'fixed')
        origin(joint, numbers(body.get('pos', '0 0 0')), mj_rotation(body))
        ET.SubElement(joint, 'parent', link=parent); ET.SubElement(joint, 'child', link=name)
        if source_joint is not None:
            assert np.all(numbers(source_joint.get('pos', '0 0 0')) == 0.)
            lower, upper = numbers(source_joint.get('range'))
            effort = max(abs(numbers(source_joint.get('actuatorfrcrange'))))
            ET.SubElement(joint, 'axis', xyz=source_joint.get('axis'))
            ET.SubElement(joint, 'limit', lower=str(lower), upper=str(upper), effort=str(effort), velocity='1.8')
        for child in body.findall('body'):
            body_link(child, name)

    for body in mjcf.findall('worldbody/body'):
        body_link(body, 'openarm_mount')
    for side in ('left', 'right'):
        hand = ET.parse(SHARPA/f'{side}_arm.urdf').getroot()
        keep = descendants(hand, 'sharpa_mount')
        for entry in hand:
            if not ((entry.tag == 'link' and entry.get('name') in keep) or
                    (entry.tag == 'joint' and entry.find('parent').get('link') in keep)):
                continue
            entry = deepcopy(entry)
            for element in entry.iter():
                for key in ('name', 'link'):
                    if element.get(key, '').startswith('sharpa_mount'):
                        element.set(key, side+'_'+element.get(key))
                if element.tag == 'mesh':
                    element.set('filename', '../sharpa/'+element.get('filename'))
            root.append(entry)
        bridge = ET.SubElement(root, 'joint', name=side+'_sharpa_adapter', type='fixed')
        origin(bridge, (0., 0., -.105), Rotation.from_euler('y', 180, degrees=True).as_matrix()
               @ Rotation.from_euler('z', 30 if side == 'left' else -30, degrees=True).as_matrix())
        ET.SubElement(bridge, 'parent', link=f'openarm_{side}_ee_base_link')
        ET.SubElement(bridge, 'child', link=f'{side}_sharpa_mount')
    write_xml(root, HERE/'openarm_sharpa.urdf')
    for side in ('left', 'right'):
        keep = descendants(root, f'openarm_{side}_base_link') | {'openarm_mount'}
        arm = ET.Element('robot', name=f'openarm_{side}_sharpa')
        for entry in root:
            if ((entry.tag == 'link' and entry.get('name') in keep) or
                (entry.tag == 'joint' and entry.find('child').get('link') in keep and entry.find('parent').get('link') in keep)):
                arm.append(deepcopy(entry))
        write_xml(arm, HERE/f'{side}_arm.urdf')
    print(HERE/'openarm_sharpa.urdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path)
    prepare(parser.parse_args().source)
