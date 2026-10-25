"""Derive the authored G1 training scene from the existing Unitree asset.

The input MJCF is only an import artifact. Training always loads g1_scene.jscn.
Collision primitives are adapted from mjlab 8bb2bc9e, Apache-2.0.
"""
from __future__ import annotations
import argparse
import json
import shutil
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import gobot
from gobot.rl.locomotion.scene import configure_robot_drives
from gobot.rl.locomotion.math import find_node_by_name
from .config import g1_velocity_cfg


def prepare(asset_root=None, project_path=None):
    cfg = g1_velocity_cfg(project_path=project_path)
    project = Path(cfg.project_path).resolve()
    asset = Path(asset_root or project.parent / 'newton_g1/assets/unitree_g1').resolve()
    source = asset / 'mjcf/g1_29dof_with_hand_rev_1_0.xml'
    if not source.is_file():
        raise FileNotFoundError(f"G1 source asset is missing: {source}; pass --asset-root")
    tree = ET.parse(source)
    root = tree.getroot()
    meshes = project / 'assets/meshes'
    meshes.mkdir(parents=True, exist_ok=True)
    for mesh in root.findall('./asset/mesh'):
        name = mesh.get('file')
        shutil.copy2(asset / 'meshes' / name, meshes / name)
    root.find('compiler').set('meshdir', str(meshes))
    preset = json.loads((Path(__file__).parent / 'collision_preset.json').read_text())['bodies']
    # Keep original visual meshes and inertias; use simple locomotion contacts.
    for body in root.findall('.//body'):
        for geom in list(body.findall('geom')):
            if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
                body.remove(geom)
        for attrs in preset.get(body.get('name'), ()):
            ET.SubElement(body, 'geom', attrs)
    for tag in ('sensor', 'keyframe'):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    gobot.set_project_path(str(project))
    with tempfile.TemporaryDirectory(prefix='gobot-g1-import-') as directory:
        xml = Path(directory) / 'g1.xml'
        tree.write(xml, encoding='unicode')
        gobot.import_mjcf_scene(str(xml), 'res://g1.jscn', name='g1')
    scene = gobot.load_scene('res://g1.jscn')
    robot = scene.root
    configure_robot_drives(robot, cfg)
    for name, value in zip(cfg.joint_names, cfg.default_joint_pos, strict=True):
        find_node_by_name(robot, name).initial_position = value
    def add(parent, kind, name):
        sensor = gobot.create_node(kind, name)
        sensor.enabled = True
        sensor.sensor_period = 0.
        sensor.noise_stddev = 0.
        sensor.visualize_debug = True
        parent.add_child(sensor)
        return sensor
    base = find_node_by_name(robot, cfg.base_link)
    add(base, 'IMUSensor3D', 'imu')
    add(base, 'AngularMomentumSensor3D', 'root_angmom')
    scan = add(base, 'HeightScanner3D', 'terrain_scan')
    scan.pattern_mode = gobot.RayPatternMode.Grid
    scan.grid_size = (1.6, 1.0)
    scan.grid_resolution = .1
    scan.ray_alignment = gobot.RayAlignmentMode.Yaw
    scan.ray_direction = (0., 0., -1.)
    scan.ray_direction_world_space = True
    scan.max_distance = 5.
    scan.reduction_mode = gobot.RayReductionMode.None_
    for foot in cfg.robot.feet:
        link = find_node_by_name(robot, foot.link)
        height = add(link, 'TerrainHeightSensor3D', foot.height_sensor)
        height.position = (.04, 0., -.035)
        height.pattern_mode = gobot.RayPatternMode.Custom
        height.sample_offsets = list(foot.sample_offsets)
        height.ray_alignment = gobot.RayAlignmentMode.Yaw
        height.ray_direction = (0., 0., -1.)
        height.ray_direction_world_space = True
        height.max_distance = 1.
        height.reduction_mode = gobot.RayReductionMode.Min
        contact = add(link, 'ContactSensor3D', foot.contact_sensor)
        contact.position = (.04, 0., -.035)
        contact.radius = .15
        contact.min_threshold = 0.
        for name in foot.collision_shapes:
            collision = find_node_by_name(robot, name)
            collision.physics_material = {'sliding_friction': .6, 'torsional_friction': 0., 'rolling_friction': 0.,
                                          'contact_compliance': 1.e-4, 'contact_damping': 1.}
    gobot.save_scene(robot, 'res://g1.jscn')
    scene_file = project / 'g1.jscn'
    data = json.loads(scene_file.read_text())
    for node in data['__NODES__']:
        if 'source_path' in node.get('properties', {}):
            node['properties']['source_path'] = 'res://' + Path(os.path.relpath(source, project)).as_posix()
    for resource in data['__EXT_RESOURCES__']:
        if resource['__TYPE__'] == 'ArrayMesh':
            resource['__PATH__'] = 'res://assets/meshes/' + Path(resource['__PATH__']).name
    scene_file.write_text(json.dumps(data, indent=4) + '\n')
    print(f"Prepared {project / 'g1.jscn'} (29 policy joints, 14 held joints)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root')
    parser.add_argument('--project-path')
    args = parser.parse_args()
    prepare(args.asset_root, args.project_path)
