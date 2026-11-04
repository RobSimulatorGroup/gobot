"""Author the pinned mjlab G1 export as Gobot scenes and audit compiled physics.

Reference export is an offline input. Training reads the resulting .jscn/profile,
never the reference checkout or a second independently loaded robot model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import gobot
import mujoco
import numpy as np

from .reproduction.reference import digest, REFERENCE_COMMIT, MJLAB_COMMIT


def audit_model(reference, artifact, prefix, *, compare_options=False):
    actual = mujoco.MjModel.from_xml_string(artifact['content'])
    before, after = mujoco.MjData(reference), mujoco.MjData(actual)
    mujoco.mj_forward(reference, before); mujoco.mj_forward(actual, after)
    checks = []
    display_differences = []

    # MuJoCo's XML writer emits six significant digits. Account for that
    # representation error explicitly; the report retains every absolute delta.
    def compare(name, left, right, *, atol=1.e-5, rtol=1.e-5):
        left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        passed = left.shape == right.shape and bool(np.allclose(left, right, atol=atol, rtol=rtol))
        checks.append({'field': name, 'passed': passed,
                       'max_error': float(np.max(np.abs(left-right))) if left.shape == right.shape and left.size else None})

    for name in ('nq', 'nv', 'nu', 'nbody', 'nexclude'):
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
        for field in ('geom_type', 'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap', 'geom_condim', 'geom_priority', 'geom_solmix', 'geom_solimp'):
            compare(f'{name}/{field}', getattr(reference, field)[i], getattr(actual, field)[j])
        if reference.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and actual.geom_type[j] == mujoco.mjtGeom.mjGEOM_MESH:
            compare(f'{name}/convex_support', support(reference, before, i), support(actual, after, j))
        else:
            if reference.geom_type[i] == actual.geom_type[j] == mujoco.mjtGeom.mjGEOM_PLANE:
                # MuJoCo planes collide as infinite planes. All three size
                # components control rendering (the third is grid spacing).
                if not np.array_equal(reference.geom_size[i], actual.geom_size[j]):
                    display_differences.append({'field': f'{name}/plane_display_size',
                        'reference': reference.geom_size[i].tolist(),
                        'candidate': actual.geom_size[j].tolist(),
                        'reason': 'MJCF plane size controls rendering, not collision geometry'})
            else:
                compare(f'{name}/size', reference.geom_size[i], actual.geom_size[j])
            compare(f'{name}/position', before.geom_xpos[i], after.geom_xpos[j])
            compare(f'{name}/rotation', before.geom_xmat[i], after.geom_xmat[j])
    return {'schema_version': 1, 'passed': all(row['passed'] for row in checks),
            'numeric_tolerance': {'atol': 1.e-5, 'rtol': 1.e-5, 'reason': 'float32 scene and MuJoCo XML precision'},
            'reference_version': mujoco.__version__, 'artifact_digest': artifact['content_digest'],
            'checks': checks, 'display_differences': display_differences, 'failed_checks': [row for row in checks if not row['passed']]}



def prepare(reference_export, project_path):
    source, project = Path(reference_export).resolve(), Path(project_path).resolve()
    profile = json.loads((source/'profile.json').read_text())
    if (profile.get('schema_version') != 2 or profile.get('reference_commit') != REFERENCE_COMMIT
            or profile.get('mjlab_commit') != MJLAB_COMMIT):
        raise ValueError('model export does not match the pinned reproduction')
    if profile['versions']['mujoco'] != mujoco.__version__:
        raise ValueError('reference export and Gobot importer must use the same MuJoCo version')
    for name, expected in profile['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError(f'reference model file changed: {name}')
    project.mkdir(parents=True, exist_ok=True)
    target = project/'assets/supermimic'
    target.mkdir(parents=True, exist_ok=True)
    for name in profile['files']:
        destination = target/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, destination)
    (target/'robot_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    gobot.set_project_path(str(project))
    gobot.import_mjcf_scene(str(target/'world.xml'), 'res://g1_29dof_flat_scene.jscn', name='g1_29dof')
    reference = mujoco.MjModel.from_xml_path(str(target/'world.xml'))
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(project))
        context.load_scene('res://g1_29dof_flat_scene.jscn')
        context.fixed_time_step = profile['physics_dt']
        context.set_mujoco_solver_settings(profile['solver_settings'])
        artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        report = audit_model(reference, artifact, 'g1_29dof_', compare_options=True)
        report.update(reference_commit=REFERENCE_COMMIT, mjlab_commit=MJLAB_COMMIT,
                      profile_sha256=digest(target/'robot_profile.json'),
                      scope='static model equivalence; rollout and policy parity are separate checks')
        (target/'world_physics_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        (target/'compiled_world.xml').write_text(artifact['content'])
        print(json.dumps({key:value for key,value in report.items() if key!='checks'}, indent=2))
        if not report['passed']:
            raise RuntimeError('G1 static physics audit failed; inspect world_physics_audit.json')
        return report
    finally:
        context.clear_scene()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-export', type=Path, required=True)
    parser.add_argument('--project-path', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference_export, args.project_path)
