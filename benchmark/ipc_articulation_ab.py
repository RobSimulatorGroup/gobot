"""Experimental joint-space IPC vs staggered proxy coupling on a Gobot scene.

Two serial sliders give a non-diagonal inertia matrix without quaternion or
joint-limit ambiguity. This is not a LEAP-hand provider or a throughput test.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import time

import numpy as np
import torch
import mujoco

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcConfig
from gobot.sim.providers import CompiledMuJoCoIpcArtifact, MuJoCoIpcConfig, MuJoCoIpcProvider, MuJoCoWarpProvider

ROOT = Path(__file__).resolve().parents[1]
DT = .002


def create_scene(*, contact=True):
    spec = importlib.util.spec_from_file_location("press_scene_builder", ROOT / "examples/mujoco_libuipc/build_scene.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = gobot.create_node("Node3D", "serial_press_probe")
    robot = gobot.create_node("Robot3D", "press")
    robot.mode = gobot.RobotMode.Motion
    root.add_child(robot)
    parent = None
    for name, mass, size, height in (("frame", 1., (.06, .06, .02), .45),
                                     ("carrier", 1., (.08, .08, .02), -.09),
                                     ("pad", 2., (.26, .26, .04), -.14)):
        link = gobot.create_node("Link3D", name)
        builder._set_box_inertia(link, mass, size)
        collision = gobot.create_box_collision(name + "_shape", size)
        collision.physics_material = {"sliding_friction": 1.}
        link.add_child(collision)
        if parent is None:
            link.position = (0., 0., height)
            robot.add_child(link)
        else:
            joint = gobot.create_node("Joint3D", name + "_slide")
            joint.joint_type = gobot.JointType.Prismatic
            joint.parent_link, joint.child_link = parent.name, name
            joint.position, joint.axis = (0., 0., height), (0., 0., 1.)
            joint.lower_limit, joint.upper_limit = -.12, .01
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness, joint.drive_damping = 5000., 100.
            joint.effort_limit, joint.velocity_limit = 1000., 1.
            joint.control_lower_limit, joint.control_upper_limit = -.12, .01
            parent.add_child(joint)
            joint.add_child(link)
        parent = link
    support = gobot.create_node("Robot3D", "support")
    support.mode = gobot.RobotMode.Assembly
    ground = gobot.create_node("Link3D", "ground")
    ground.position = (0., 0., -.025)
    builder._set_box_inertia(ground, 100., (.8, .8, .05))
    shape = gobot.create_box_collision("ground_shape", (.8, .8, .05))
    shape.physics_material = {"sliding_friction": 1.}
    ground.add_child(shape)
    support.add_child(ground)
    root.add_child(support)
    soft = gobot.create_node("DeformableBody3D", "soft")
    soft.mesh = builder._box_tetrahedral_mesh((.22, .22, .16), (3, 3, 3))
    soft.position = (0., 0. if contact else .28, .0808)
    soft.density, soft.young_modulus, soft.poisson_ratio = 650., 2.5e4, .38
    soft.self_collision_enabled = False
    root.add_child(soft)

    def walk(node, path=""):
        yield node, path
        for child in node.children:
            yield from walk(child, path + "/" + child.name)

    for link, path in [(node, path) for node, path in walk(root) if node.type_name == "Link3D"]:
        coupling = gobot.create_node("PhysicsCoupling", "couple_" + link.name)
        coupling.target_body_path = ".." + path
        coupling.mode = (gobot.PhysicsCouplingMode.TwoWay if link.name in ("carrier", "pad")
                         else gobot.PhysicsCouplingMode.OneWay)
        root.add_child(coupling)
    return root


def _transform_points(points, transform):
    matrix = np.asarray(transform["matrix_row_major"]).reshape(4, 4)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def probe_job(artifact, workspace):
    manifest = json.loads(artifact.ipc.manifest)
    soft = manifest["deformable_bodies"][0]
    blob = artifact.ipc.blobs[soft["mesh_blob"]]
    _, nv, nt, _ = struct.unpack_from("<4I", blob, 8)
    vertices = np.frombuffer(blob, "<f8", nv * 3, 24).reshape(-1, 3)
    tetrahedra = np.frombuffer(blob, "<u4", nt * 4, 24 + nv * 24).reshape(-1, 4)
    soft["vertices"] = _transform_points(vertices, soft["transform"]).tolist()
    soft["tetrahedra"] = tetrahedra.tolist()
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    triangles = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                 [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
    links, joints = [], []
    for robot in manifest["robots"]:
        joints.extend(robot["joints"])
        for link in robot["links"]:
            shapes = [shape for shape in link["collision_shapes"] if not shape["disabled"]]
            if len(shapes) != 1 or shapes[0]["shape_type"] != "box":
                raise ValueError("probe requires one authored box per link")
            shape = shapes[0]
            link["vertices"] = _transform_points(corners * np.asarray(shape["size"]) / 2.,
                                                shape["link_transform"]).tolist()
            link["triangles"] = triangles
            link["fixed"] = link["path"] in robot["root_link_paths"]
            links.append(link)
    return {"workspace": str(workspace), "dt": DT, "contact": True, "linear_tolerance": 1.e-3,
            "soft": soft, "links": links, "joints": joints}


def joint_layout(model, job):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "press_" + joint["name"])
           for joint in job["joints"]]
    if len(ids) != 2 or min(ids) < 0 or len(set(ids)) != 2:
        raise ValueError("joint experiment requires a bijective named joint mapping")
    if np.any(model.jnt_type[ids] != int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError("only scalar slide coordinates may use this probe")
    return np.asarray(model.jnt_qposadr[ids]), np.asarray(model.jnt_dofadr[ids])


class ArticulationProbe:
    def __init__(self, module, job):
        self.library = ctypes.CDLL(str(Path(module).resolve()))
        lib = self.library
        lib.gobot_probe_error.restype = ctypes.c_char_p
        lib.gobot_probe_create.argtypes, lib.gobot_probe_create.restype = [ctypes.c_char_p], ctypes.c_void_p
        pointer = ctypes.POINTER(ctypes.c_double)
        lib.gobot_probe_step.argtypes = [ctypes.c_void_p, pointer, pointer]
        lib.gobot_probe_step.restype = ctypes.c_char_p
        lib.gobot_probe_destroy.argtypes = [ctypes.c_void_p]
        lib.gobot_probe_destroy.restype = None
        self.handle = lib.gobot_probe_create(json.dumps(job, allow_nan=False).encode())
        if not self.handle:
            raise RuntimeError(lib.gobot_probe_error().decode())

    def step(self, mass, prediction):
        mass = np.ascontiguousarray(mass, dtype=np.float64)
        prediction = np.ascontiguousarray(prediction, dtype=np.float64)
        if mass.shape != (2, 2) or prediction.shape != (2,):
            raise ValueError("probe inputs require a 2x2 mass and 2-vector prediction")
        pointer = ctypes.POINTER(ctypes.c_double)
        result = self.library.gobot_probe_step(self.handle, mass.ctypes.data_as(pointer),
                                               prediction.ctypes.data_as(pointer))
        if result is None:
            raise RuntimeError(self.library.gobot_probe_error().decode())
        return json.loads(result)

    def close(self):
        if self.handle:
            self.library.gobot_probe_destroy(self.handle)
            self.handle = None


def run(args):
    if args.steps < 1 or not np.isfinite(args.linear_tolerance) or args.linear_tolerance <= 0.:
        raise ValueError("steps and linear tolerance must be positive")
    report = {"schema_version": 1, "experiment": "two_serial_prismatic_soft_press",
              "mode": args.mode, "contact": not args.no_contact, "error": None,
              "scope": "Host-staged scalar-joint experiment, not a LEAP provider or batch speed comparison.",
              "linear_tolerance": args.linear_tolerance, "samples": []}
    with tempfile.TemporaryDirectory(prefix="gobot-articulation-probe-") as directory:
        context = gobot.app.create_context()
        provider = probe = None
        try:
            scene_path = Path(directory) / "probe.jscn"
            gobot.save_scene(create_scene(contact=not args.no_contact), str(scene_path))
            context.load_scene(str(scene_path))
            context.fixed_time_step = DT
            artifact = CompiledMuJoCoIpcArtifact.from_context(context)
            report["artifact_digest"] = artifact.digest
            model = mujoco.MjModel.from_xml_string(artifact.mujoco.content)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            mass = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, data, mass)
            if mass.shape != (2, 2) or abs(mass[0, 1]) < .1:
                raise ValueError("fixture must have two coupled scalar inertial DOFs")
            report["mass_matrix"] = mass.tolist()
            job = probe_job(artifact, directory)
            job["linear_tolerance"] = args.linear_tolerance
            q_ids, dof_ids = joint_layout(model, job)
            report["ipc_joint_names"] = [joint["name"] for joint in job["joints"]]
            report["ipc_to_mujoco_qpos"] = q_ids.tolist()
            report["ipc_to_mujoco_dof"] = dof_ids.tolist()
            if args.mode == "joint":
                probe = ArticulationProbe(args.probe_module, job)
                provider = MuJoCoWarpProvider(artifact.mujoco, num_envs=1, nconmax=32, njmax=64,
                                              overflow_check_interval=1)
            else:
                provider = MuJoCoIpcProvider(artifact, config=MuJoCoIpcConfig(
                    num_envs=1, environments_per_shard=1, coupling_iterations=2),
                    libuipc_config=LibuipcBatchConfig(solver=LibuipcConfig(
                        fixed_time_step=DT, workspace=directory,
                        module_path=args.module_path, contact_activation_distance=.0008,
                        friction_coefficient=1., kinematic_strength=100.),
                        environments_per_shard=1, newton_max_iterations=48,
                        line_search_max_iterations=16, strict_convergence=True,
                        linear_system_tolerance_rate=args.linear_tolerance),
                    mujoco_options={"nconmax": 32, "njmax": 64, "overflow_check_interval": 1})
            arrays = provider.arrays
            command = torch.zeros_like(arrays["ctrl"])
            residual_max = reaction_max = prediction_error_max = 0.
            times = []
            for tick in range(args.steps):
                progress = min(1., (tick + 1) / 300.)
                command.fill_(-.035 * progress * progress * (3. - 2. * progress))
                before = arrays["qpos"][0].cpu().numpy().copy()
                start = time.perf_counter()
                provider.step(command)
                provider.synchronize()
                if probe is not None:
                    predicted = (arrays["qpos"][0].cpu().numpy() - before)[q_ids]
                    ipc_mass = mass[np.ix_(dof_ids, dof_ids)]
                    result = probe.step(ipc_mass, predicted)
                    delta = np.asarray(result["delta"])
                    prediction_error_max = max(prediction_error_max, float(np.abs(delta - predicted).max()))
                    q = before.copy()
                    q[q_ids] += delta
                    velocity = np.zeros(model.nv)
                    velocity[dof_ids] = delta / DT
                    # IPC's solved increment, not the commanded target, owns the
                    # correction. Update both q and qdot, without a reset each tick.
                    arrays["qpos"].copy_(torch.as_tensor(q[None], device="cuda:0"))
                    arrays["qvel"].copy_(torch.as_tensor(velocity[None], device="cuda:0"))
                    provider.forward()
                    provider.synchronize()
                    positions = np.asarray(result["positions"])
                    reaction = float(np.linalg.norm(ipc_mass @ (delta - predicted) / DT ** 2))
                    transforms = np.asarray(result["transforms"]).reshape(-1, 4, 4)
                    data.qpos[:] = q
                    mujoco.mj_kinematics(model, data)
                    moving = [link for link in job["links"] if not link["fixed"]]
                    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "press_" + link["name"])
                           for link in moving]
                    residual = float(np.linalg.norm(transforms[:, :3, 3] - data.xpos[ids], axis=1).max())
                else:
                    positions = arrays["ipc_positions"][0].cpu().numpy()
                    residual = float(torch.linalg.vector_norm(
                        arrays["ipc_affine_targets"][0, :, :3, 3] -
                        arrays["ipc_affine_transforms"][0, :, :3, 3], dim=1).max())
                    data.qpos[:] = arrays["qpos"][0].cpu().numpy()
                    mujoco.mj_forward(model, data)
                    wrenches = arrays["ipc_affine_contact_wrenches"][0].cpu().numpy()
                    generalized = np.zeros(model.nv)
                    jacobian_p, jacobian_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    for mapping in artifact.coupled_bodies:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mapping.mujoco_body_name)
                        mujoco.mj_jacBodyCom(model, data, jacobian_p, jacobian_r, body_id)
                        force = wrenches[mapping.ipc_body_index]
                        generalized += jacobian_p.T @ force[:3] + jacobian_r.T @ force[3:]
                    reaction = float(np.linalg.norm(generalized))
                times.append(1000. * (time.perf_counter() - start))
                q = arrays["qpos"][0].cpu().numpy().copy()
                if not np.isfinite(positions).all() or not np.isfinite(q).all():
                    raise RuntimeError("non-finite coupled state")
                if np.any(q < -.12) or np.any(q > .01):
                    raise RuntimeError("joint correction escaped the authored limits")
                residual_max, reaction_max = max(residual_max, residual), max(reaction_max, reaction)
                if tick == 0 or (tick + 1) % 25 == 0 or tick + 1 == args.steps:
                    report["samples"].append({"step": tick + 1, "qpos": q.tolist(),
                        "soft_height": float(np.ptp(positions[:, 2])),
                        "minimum_z": float(positions[:, 2].min()),
                        "interface_translation_error_meters": residual, "reaction_norm": reaction})
                    if probe is not None:
                        report["samples"][-1].update(predicted_delta=predicted.tolist(), solved_delta=delta.tolist())
                        report["samples"][-1]["solver"] = {key: result[key] for key in (
                            "newton_iterations", "pcg_iterations", "pcg_relative_residual")}
            report.update(steps=args.steps, maximum_interface_translation_error_meters=residual_max,
                          maximum_reaction_norm=reaction_max,
                          maximum_prediction_correction_meters=prediction_error_max if probe else None,
                          step_ms={"median": float(np.median(times)), "p95": float(np.percentile(times, 95))})
            report["reaction_units"] = "generalized force N (two prismatic DOFs)"
            report["reaction_source"] = ("M * (solved_delta - predicted_delta) / dt^2; includes numerical residual"
                                         if probe else "J^T * native IPC contact wrenches")
            report["passed"] = (residual_max < .001 and (args.no_contact or reaction_max > .1)
                                and report["samples"][-1]["minimum_z"] >= -.001)
            if args.no_contact and probe:
                report["passed"] &= prediction_error_max < 1.e-5
        except Exception as error:
            report["error"], report["passed"] = str(error), False
        finally:
            if provider is not None:
                provider.close()
            if probe is not None:
                probe.close()
            context.clear_scene()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("joint", "proxy"), required=True)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--linear-tolerance", type=float, default=1.e-3)
    parser.add_argument("--module-path", default="")
    parser.add_argument("--probe-module", type=Path,
                        default=ROOT / "build/cp313-cp313-linux_x86_64/benchmark/libgobot_ipc_articulation_probe.so")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "error": report["error"], "report": str(args.report)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
