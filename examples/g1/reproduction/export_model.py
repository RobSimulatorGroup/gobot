"""Export the effective pinned mjlab G1 model without creating a GPU environment.

Run with the independent reference interpreter. Only the resulting XML/profile
cross the reference-to-Gobot boundary; Gobot's runtime never imports mjlab.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import types
import xml.etree.ElementTree as ET

from .reference import digest, verify_checkout, MJLAB_COMMIT, REFERENCE_COMMIT


def export(output):
    import mujoco
    import mjlab
    import supermimic
    import torch
    # Config factories are called directly; unrelated registered PPO tasks are
    # not needed to author this model and can otherwise import training plugins.
    if 'mjlab.tasks' not in sys.modules:
        tasks = types.ModuleType('mjlab.tasks')
        tasks.__path__ = [str(Path(mjlab.__file__).parent/'tasks')]
        sys.modules['mjlab.tasks'] = tasks
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE
    from mjlab.scene import Scene
    from supermimic.data.motion_convert import G1_JOINT_NAMES
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg, TRACKED_BODIES

    mjlab_root = Path(mjlab.__file__).resolve().parents[2]
    reference_root = Path(supermimic.__file__).resolve().parents[2]
    verify_checkout(mjlab_root, MJLAB_COMMIT)
    verify_checkout(reference_root, REFERENCE_COMMIT)
    output = Path(output).resolve()
    if (output/'profile.json').exists():
        raise FileExistsError('model export already exists; use a new directory')
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], num_envs=1))
    scene = Scene(cfg.scene, device='cpu')
    compiled = scene.compile()
    cfg.sim.mujoco.apply(compiled)
    fields = ('timestep', 'integrator', 'gravity', 'impratio', 'cone', 'jacobian',
              'solver', 'iterations', 'tolerance', 'ls_iterations', 'ls_tolerance',
              'ccd_iterations', 'disableflags', 'enableflags')
    for name in fields:
        setattr(scene.spec.option, name, getattr(compiled.opt, name))
    scene.write(output)
    # Remove only the entity namespace, consistently for names and references.
    # Preserve the original export for comparison and leave mesh paths untouched.
    tree = ET.parse(output/'scene.xml')
    for node in tree.iter():
        for key, value in list(node.attrib.items()):
            if key not in ('file', 'meshdir', 'texturedir') and value.startswith('robot/'):
                node.set(key, value.replace('robot/', 'robot_', 1) if key in ('class', 'childclass')
                         else value.removeprefix('robot/'))
    tree.getroot().set('model', 'g1_29dof')
    tree.write(output/'world.xml', encoding='unicode')
    model = mujoco.MjModel.from_xml_path(str(output/'world.xml'))
    import numpy as np
    # Verify that XML/name normalization preserves the effective compiled model.
    fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
              'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'dof_armature', 'dof_damping',
              'geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_priority',
              'geom_condim', 'geom_contype', 'geom_conaffinity', 'geom_solref', 'geom_solimp',
              'actuator_trnid', 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange')
    export_checks = {}
    for name in fields:
        before, after = np.asarray(getattr(compiled, name)), np.asarray(getattr(model, name))
        if before.shape != after.shape or not np.allclose(before, after, atol=1e-5, rtol=1e-5):
            raise ValueError(f'XML export changed effective model field: {name}')
        export_checks[name] = float(np.max(np.abs(before-after))) if before.size else 0.
    names = tuple(model.joint(i).name for i in range(model.njnt)
                  if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE)
    if names != G1_JOINT_NAMES or (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError('exported G1 topology differs from the pinned reference task')
    actuator_ids = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        matches = [i for i in range(model.nu) if model.actuator_trnid[i, 0] == joint_id]
        if len(matches) != 1:
            raise ValueError(f'expected one actuator for {name}')
        actuator_ids.append(matches[0])
    scales = []
    for name in names:
        matches = [value for pattern, value in G1_ACTION_SCALE.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f'action scale patterns are ambiguous for {name}')
        scales.append(matches[0])
    foot = [model.geom(i).name for i in range(model.ngeom)
            if re.fullmatch(r'(left|right)_foot[1-7]_collision', model.geom(i).name or '')]
    profile = {
        'schema_version': 2, 'reference_commit': REFERENCE_COMMIT, 'mjlab_commit': MJLAB_COMMIT,
        'robot_name': 'g1_29dof', 'base_link': 'pelvis', 'anchor_link': 'torso_link',
        'joint_names': names, 'tracked_links': TRACKED_BODIES, 'foot_shapes': foot,
        'default_joint_position': model.key('init_state').qpos[7:].tolist(),
        'kp': model.actuator_gainprm[actuator_ids, 0].tolist(),
        'kd': (-model.actuator_biasprm[actuator_ids, 2]).tolist(),
        'effort_limit': model.actuator_forcerange[actuator_ids, 1].tolist(),
        'action_scale': scales, 'physics_dt': cfg.sim.mujoco.timestep, 'decimation': cfg.decimation,
        'soft_joint_limit_factor': cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor,
        'solver_settings': {'solver': 'Newton', 'integrator': 'ImplicitFast', 'cone': 'Pyramidal',
            'iterations': model.opt.iterations, 'line_search_iterations': model.opt.ls_iterations,
            'convex_collision_iterations': model.opt.ccd_iterations, 'tolerance': model.opt.tolerance,
            'line_search_tolerance': model.opt.ls_tolerance,
            'convex_collision_tolerance': model.opt.ccd_tolerance, 'impedance_ratio': model.opt.impratio},
        'versions': {'mujoco': mujoco.__version__, 'torch': torch.__version__,
                     'mujoco-warp': importlib.metadata.version('mujoco-warp'),
                     'warp-lang': importlib.metadata.version('warp-lang')},
        'nominal_evaluation_foot_friction': 1.,
        'export_scope': 'static authored model; per-episode randomization is not baked into XML',
        'task_autodiscovery': False, 'export_max_abs_error': export_checks,
        'simulation_config': asdict(cfg.sim.mujoco),
    }
    profile['files'] = {str(path.relative_to(output)): digest(path)
                        for path in sorted(output.rglob('*')) if path.is_file()}
    (output/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
                      'collision_geoms': int(((model.geom_contype | model.geom_conaffinity) != 0).sum()),
                      'foot_shapes': len(foot), 'versions': profile['versions']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    export(parser.parse_args().output)
