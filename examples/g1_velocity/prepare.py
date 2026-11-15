"""Author the pinned mjlab G1 as a self-contained Gobot locomotion scene.

MJCF is an offline input; training and evaluation load only g1.jscn and its meshes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import mujoco
import numpy as np
import gobot
from gobot.rl.locomotion.scene import configure_robot_drives
from gobot.rl.locomotion.math import find_node_by_name
from .config import g1_velocity_cfg, REFERENCE_REVISION


SOURCE_SHA256 = "febdcbeffbbf84051556ae41a5ac1b43fb479a5d76bdb3f54824dbc2721c20aa"


def prepare(source=None, project_path=None):
    cfg = g1_velocity_cfg(project_path=project_path)
    project = Path(cfg.project_path).resolve()
    source = Path(source or Path(__file__).resolve().parents[2] /
                  'mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml').resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Pass --source pointing to mjlab {REFERENCE_REVISION}'s G1 XML: {source}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("G1 source differs from the pinned mjlab model; review the source before regenerating")
    spec = mujoco.MjSpec.from_file(str(source))
    model = spec.compile()
    if (model.nq, model.nv) != (36, 35):
        raise ValueError('locomotion requires the 29-joint G1 with rigid palms')
    # Store visuals beside the scene; training never needs the reference checkout.
    mesh_directory = project / 'g1.meshes'
    mesh_directory.mkdir(parents=True, exist_ok=True)
    for mesh in spec.meshes:
        i = model.mesh(mesh.name).id
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.mesh_quat[i])
        first, count = model.mesh_vertadr[i], model.mesh_vertnum[i]
        vertices = (model.mesh_vert[first:first+count] @ rotation.reshape(3, 3).T + model.mesh_pos[i]).astype('<f4')
        first, count = model.mesh_faceadr[i], model.mesh_facenum[i]
        faces = model.mesh_face[first:first+count]
        records = np.empty(count, dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
        records['count'] = 3
        records['indices'] = faces
        header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n'
                  f'property float x\nproperty float y\nproperty float z\nelement face {len(faces)}\n'
                  'property list uchar int vertex_indices\nend_header\n')
        with (mesh_directory / f'{mesh.name}.ply').open('wb') as stream:
            stream.write(header.encode())
            stream.write(vertices.tobytes())
            stream.write(records.tobytes())
        # Inline input meshes keep the temporary MJCF independent of mesh directories.
        mesh.uservert = vertices.ravel()
        mesh.userface = faces.ravel()
        mesh.file = ''
    # FULL_COLLISION in the pinned mjlab G1 preset: foot friction, priorities and dimensions.
    for geom in spec.geoms:
        if geom.name.endswith('_collision'):
            foot = '_foot' in geom.name
            geom.condim = 3 if foot else 1
            geom.priority = int(foot)
            if foot:
                geom.friction[0] = .6
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for key in list(spec.keys):
        spec.delete(key)
    gobot.set_project_path(str(project))
    with tempfile.TemporaryDirectory(prefix='gobot-g1-import-') as directory:
        xml = Path(directory) / 'g1.xml'
        xml.write_text(spec.to_xml())
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
    imu = add(base, 'IMUSensor3D', 'imu')
    imu.position = tuple(model.site('imu_in_pelvis').pos)
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
        height.position = (.04, 0., -.037)
        height.pattern_mode = gobot.RayPatternMode.Custom
        height.sample_offsets = list(foot.sample_offsets)
        height.ray_alignment = gobot.RayAlignmentMode.Yaw
        height.ray_direction = (0., 0., -1.)
        height.ray_direction_world_space = True
        height.max_distance = 1.
        height.reduction_mode = gobot.RayReductionMode.Min
        contact = add(link, 'ContactSensor3D', foot.contact_sensor)
        contact.position = (.04, 0., -.037)
        contact.radius = .15
        contact.min_threshold = 0.
    gobot.save_scene(robot, 'res://g1.jscn')
    # The source identifier is provenance only, not a runtime dependency.
    scene_file = project / 'g1.jscn'
    data = json.loads(scene_file.read_text())
    visuals = {}
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = model.mesh(int(model.geom_dataid[i])).name
            name = model.geom(i).name or f'{mesh_name}_visual_{i}'
            visuals[name] = mesh_name
    for mesh in spec.meshes:
        data['__EXT_RESOURCES__'].append({'__ID__': f'mesh_{mesh.name}', '__TYPE__': 'ArrayMesh',
                                         '__PATH__': f'res://g1.meshes/{mesh.name}.ply'})
    for node in data['__NODES__']:
        if node['type'] == 'MeshInstance3D':
            node['properties']['mesh'] = f'ExtResource(mesh_{visuals[node["name"]]})'
        if 'source_path' in node.get('properties', {}):
            node['properties']['source_path'] = ''
    scene_file.write_text(json.dumps(data, indent=4) + '\n')
    print(f"Prepared {scene_file} (29 joints, rigid palms; mjlab {REFERENCE_REVISION[:8]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--project-path', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.project_path)
