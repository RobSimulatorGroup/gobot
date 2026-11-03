"""Author the reference 29DoF G1 in Gobot and audit its compiled physics.

The reference checkout is used only by this asset preparation command. Runtime
environments load the resulting .jscn and never import UniLab or source MJCF.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
from copy import deepcopy
import xml.etree.ElementTree as ET

import gobot
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / 'examples/locomotion/logs/supermimic_reference/Uni-SuperMimic'


def audit_model(reference, artifact, prefix, *, compare_options=False):
    actual = mujoco.MjModel.from_xml_string(artifact['content'])
    before, after = mujoco.MjData(reference), mujoco.MjData(actual)
    mujoco.mj_forward(reference, before); mujoco.mj_forward(actual, after)
    checks = []

    # MuJoCo's XML writer emits six significant digits. Account for that
    # representation error explicitly; the report retains every absolute delta.
    def compare(name, left, right, *, atol=1.e-5, rtol=1.e-5):
        left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        passed = left.shape == right.shape and bool(np.allclose(left, right, atol=atol, rtol=rtol))
        checks.append({'field': name, 'passed': passed,
                       'max_error': float(np.max(np.abs(left-right))) if left.shape == right.shape and left.size else None})

    for name in ('nq', 'nv', 'nu', 'nbody'):
        compare(name, getattr(reference, name), getattr(actual, name), atol=0, rtol=0)
    if compare_options:
        for name in ('timestep', 'gravity', 'integrator', 'cone', 'solver', 'jacobian', 'iterations',
                     'ls_iterations', 'noslip_iterations', 'ccd_iterations', 'tolerance', 'ls_tolerance',
                     'noslip_tolerance', 'ccd_tolerance', 'impratio'):
            compare('option/'+name, getattr(reference.opt, name), getattr(actual.opt, name))
    for i in range(1, reference.nbody):
        name = reference.body(i).name
        j = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_BODY, prefix+name)
        if j < 0:
            checks.append({'field': f'body/{name}', 'passed': False, 'reason': 'missing'}); continue
        for field in ('body_mass', 'body_ipos', 'body_inertia'):
            compare(f'{name}/{field}', getattr(reference, field)[i], getattr(actual, field)[j])
        for field in ('xpos', 'xmat', 'ximat'):
            compare(f'{name}/{field}', getattr(before, field)[i], getattr(after, field)[j])
        expected_parent = reference.body(int(reference.body_parentid[i])).name
        actual_parent = actual.body(int(actual.body_parentid[j])).name.removeprefix(prefix)
        checks.append({'field': f'{name}/parent', 'passed': actual_parent == expected_parent,
                       'expected': expected_parent, 'actual': actual_parent})
    for i in range(reference.njnt):
        name = reference.joint(i).name
        j = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_JOINT, prefix+name)
        if j < 0:
            checks.append({'field': f'joint/{name}', 'passed': False, 'reason': 'missing'}); continue
        for field in ('jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited'):
            compare(f'{name}/{field}', getattr(reference, field)[i], getattr(actual, field)[j])
        di, dj = reference.jnt_dofadr[i], actual.jnt_dofadr[j]
        count = 6 if reference.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else 1
        for field in ('dof_armature', 'dof_damping', 'dof_frictionloss'):
            compare(f'{name}/{field}', getattr(reference, field)[di:di+count], getattr(actual, field)[dj:dj+count])
    for i in range(reference.nu):
        name = reference.joint(int(reference.actuator_trnid[i, 0])).name
        jid = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_JOINT, prefix+name)
        matches = np.flatnonzero(actual.actuator_trnid[:, 0] == jid)
        if len(matches) != 1:
            checks.append({'field': f'{name}/actuator', 'passed': False}); continue
        j = matches[0]
        for field in ('actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'actuator_forcerange', 'actuator_forcelimited', 'actuator_ctrllimited'):
            compare(f'{name}/{field}', getattr(reference, field)[i], getattr(actual, field)[j])
    colliders = np.flatnonzero(reference.geom_contype | reference.geom_conaffinity)
    compare('collision_count', len(colliders), actual.ngeom, atol=0, rtol=0)
    directions = np.random.default_rng(71).normal(size=(32, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def support(model, data, geom):
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        return (world @ directions.T).max(0)

    for i in colliders:
        name = reference.geom(i).name
        j = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_GEOM, prefix+name)
        if j < 0 and reference.geom_bodyid[i] == 0:
            candidates = np.flatnonzero((actual.geom_bodyid == 0) & (actual.geom_type == reference.geom_type[i]))
            if len(candidates) == 1:
                j = candidates[0]
        if j < 0:
            checks.append({'field': f'geom/{name}', 'passed': False, 'reason': 'missing'}); continue
        for field in ('geom_type', 'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap', 'geom_condim'):
            compare(f'{name}/{field}', getattr(reference, field)[i], getattr(actual, field)[j])
        if reference.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and actual.geom_type[j] == mujoco.mjtGeom.mjGEOM_MESH:
            compare(f'{name}/convex_support', support(reference, before, i), support(actual, after, j))
        else:
            compare(f'{name}/size', reference.geom_size[i], actual.geom_size[j])
            compare(f'{name}/position', before.geom_xpos[i], after.geom_xpos[j])
            compare(f'{name}/rotation', before.geom_xmat[i], after.geom_xmat[j])
    return {'schema_version': 1, 'passed': all(row['passed'] for row in checks),
            'numeric_tolerance': {'atol': 1.e-5, 'rtol': 1.e-5, 'reason': 'float32 scene and MuJoCo XML precision'},
            'reference_version': mujoco.__version__, 'artifact_digest': artifact['content_digest'],
            'checks': checks, 'failed_checks': [row for row in checks if not row['passed']]}


def prepare(reference_root=DEFAULT_REFERENCE, project_path=None):
    project = Path(project_path or Path(__file__).parent).resolve()
    reference_root = Path(reference_root).resolve()
    records = json.loads((reference_root/'reference-assets.json').read_text())
    for item in records['files']:
        asset = reference_root/item['local']
        if hashlib.sha256(asset.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'pinned source asset changed: {asset}')
    motion_directory = project/'assets/motions/tracking'
    motion_directory.mkdir(parents=True, exist_ok=True)
    motion_records = []
    for item in records['files']:
        if not item['remote'].startswith('g1/') or not item['remote'].endswith('.csv'):
            continue
        asset = reference_root/item['local']
        shutil.copy2(asset, motion_directory/asset.name)
        subject = int(re.search(r'subject(\d+)', asset.stem).group(1))
        motion_records.append({'name':asset.stem,'file':asset.name,'sha256':item['sha256'],
            'subject':subject,'split':'train' if subject<=3 else 'validation' if subject==4 else 'holdout'})
    if len(motion_records) != 40:
        raise ValueError('reference preparation requires all 40 pinned motions')
    motion_manifest = {'schema_version':1,'source':records['dataset'],'revision':records['revision'],
                       'source_fps':30,'clips':sorted(motion_records,key=lambda row:row['name'])}
    (motion_directory/'manifest.json').write_text(json.dumps(motion_manifest,indent=2)+'\n')
    source = reference_root / 'src/unisupermimic/assets/robots/g1'
    target = project / 'assets/supermimic'
    meshes = target / 'meshes'; meshes.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source / 'supermimic.xml')
    manifest = {'reference_commit': 'faf7818e8abac3ec6abb8f00e6e904d3577f800b',
                'source_files': {}, 'robot_name': 'g1_29dof', 'joint_names': []}
    for mesh in tree.getroot().findall('./asset/mesh'):
        path = source / 'assets' / mesh.get('file')
        shutil.copy2(path, meshes / path.name)
        manifest['source_files'][str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['source_files']['supermimic.xml'] = hashlib.sha256((source/'supermimic.xml').read_bytes()).hexdigest()
    tree.getroot().set('model', 'g1_29dof')
    tree.getroot().find('compiler').set('meshdir', 'meshes')
    tree.write(target/'robot.xml', encoding='unicode')
    gobot.set_project_path(str(project))
    gobot.import_mjcf_scene(str(target/'robot.xml'), 'res://g1_29dof.jscn', name='g1_29dof')
    reference = mujoco.MjModel.from_xml_path(str(source/'supermimic.xml'))
    scene_model = mujoco.MjModel.from_xml_path(str(source/'scene_supermimic.xml'))
    world = deepcopy(tree.getroot())
    scene_xml = ET.parse(source/'scene_supermimic.xml').getroot()
    world.append(deepcopy(scene_xml.find('option')))
    for geometry in scene_xml.findall('./worldbody/geom'):
        geometry = deepcopy(geometry); geometry.attrib.pop('material', None)
        world.find('worldbody').append(geometry)
    # Keep zero-pose geometry; motion resets provide the root and joint state.
    ET.ElementTree(world).write(target/'world.xml', encoding='unicode')
    gobot.import_mjcf_scene(str(target/'world.xml'), 'res://g1_29dof_flat_scene.jscn', name='g1_29dof')
    manifest['joint_names'] = [reference.joint(i).name for i in range(reference.njnt) if reference.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    manifest['default_joint_position'] = scene_model.key('stand').qpos[7:].tolist()
    manifest['kp'] = reference.actuator_gainprm[:, 0].tolist()
    manifest['kd'] = (-reference.actuator_biasprm[:, 2]).tolist()
    manifest['action_scale'] = (.25 * reference.actuator_forcerange[:, 1] / reference.actuator_gainprm[:, 0]).tolist()
    manifest['physics_dt'] = .005; manifest['decimation'] = 4
    manifest['solver_settings'] = {'solver': 'Newton', 'integrator': 'ImplicitFast', 'cone': 'Pyramidal',
        'iterations': int(scene_model.opt.iterations), 'line_search_iterations': int(scene_model.opt.ls_iterations),
        'convex_collision_iterations': int(scene_model.opt.ccd_iterations), 'tolerance': float(scene_model.opt.tolerance),
        'line_search_tolerance': float(scene_model.opt.ls_tolerance),
        'convex_collision_tolerance': float(scene_model.opt.ccd_tolerance), 'impedance_ratio': float(scene_model.opt.impratio)}
    (target/'robot_profile.json').write_text(json.dumps(manifest, indent=2)+'\n')
    context = gobot.app.context(); context.set_project_path(str(project))
    context.load_scene('res://g1_29dof.jscn'); context.fixed_time_step = .005
    artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
    report = audit_model(reference, artifact, 'g1_29dof_')
    (target/'physics_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    context.clear_scene(); context.load_scene('res://g1_29dof_flat_scene.jscn')
    context.set_mujoco_solver_settings(manifest['solver_settings'])
    world_artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
    world_report = audit_model(scene_model, world_artifact, 'g1_29dof_', compare_options=True)
    (target/'world_physics_audit.json').write_text(json.dumps(world_report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    if not report['passed'] or not world_report['passed']:
        print(json.dumps(world_report['failed_checks'], indent=2))
        raise RuntimeError('29DoF physics audit failed; inspect assets/supermimic/physics_audit.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--project-path', type=Path)
    args = parser.parse_args()
    prepare(args.reference_root, args.project_path)
