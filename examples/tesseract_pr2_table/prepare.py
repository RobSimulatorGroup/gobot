"""Solve Tesseract's upstream PR2 table case and export both FK tracks for Gobot."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from tesseract_robotics import tesseract_collision as collision
from tesseract_robotics import tesseract_common as common
from tesseract_robotics import tesseract_environment as environment
from tesseract_robotics import tesseract_motion_planners_trajopt as trajopt


def solve(env, group, case):
    basic = case["basic_info"]
    seed = np.asarray(case["init_info"]["data"])
    names = group.getJointNames()
    env.setState(names, seed[0])
    last = basic["n_steps"] - 1
    problem = trajopt.ProblemConstructionInfo(env)
    problem.kin = group
    problem.basic_info.n_steps = len(seed)
    problem.basic_info.manip = basic["manip"]
    problem.basic_info.fixed_timesteps = np.array(basic["fixed_timesteps"], dtype=np.int32)
    problem.basic_info.convex_solver = trajopt.ModelType(trajopt.ModelType.OSQP)
    problem.init_info.type = trajopt.InitInfo.GIVEN_TRAJ
    problem.init_info.data = seed

    vel, contact = (term["params"] for term in case["costs"])
    velocity = trajopt.createSmoothVelocityTermInfo(0, last, len(names), float(vel["coeffs"][0]))
    term = trajopt.CollisionTermInfo()
    term.name = "collision"
    term.first_step, term.last_step = 0, last
    term.term_type = trajopt.TermType_TT_COST
    term.evaluator_type = trajopt.CollisionEvaluatorType_CAST_CONTINUOUS
    term.longest_valid_segment_length = contact["longest_valid_segment_length"]
    term.fixed_steps = np.array(contact["fixed_steps"], dtype=np.int32)
    term.info = trajopt.createSafetyMarginDataVector(len(seed), contact["dist_pen"], contact["coeffs"])
    target = case["constraints"][0]["params"]
    end = trajopt.JointPosTermInfo()
    end.name = "joint0"
    end.first_step, end.last_step = target["first_step"], target["last_step"]
    end.coeffs, end.targets = np.array(target["coeffs"], dtype=float), np.array(target["targets"])
    end.term_type = trajopt.TermType_TT_CNT
    problem.cost_infos.push_back(velocity)
    problem.cost_infos.push_back(term)
    problem.cnt_infos.push_back(end)
    native = trajopt.ConstructProblem(problem)
    # In wheel 0.5.1 the Python shim passes an invalid default plotter. Select
    # the one-argument C++ overload directly.
    result = trajopt._tesseract_motion_planners_trajopt_python.OptimizeProblem(native)
    path = np.array(result.traj)
    np.testing.assert_allclose(path[[0, -1]], seed[[0, -1]], atol=1e-5, rtol=0)
    return seed, path


def colliding_segments(env, group, path):
    manager = env.getContinuousContactManager()
    active = group.getActiveLinkNames()
    manager.setActiveCollisionObjects(active)
    manager.setCollisionMarginData(common.CollisionMarginData(0.0))
    hits = []
    for index, (a, b) in enumerate(zip(path, path[1:])):
        first = env.getState(group.getJointNames(), a)
        second = env.getState(group.getJointNames(), b)
        for name in active:
            manager.setCollisionObjectsTransform(name, first.link_transforms[name], second.link_transforms[name])
        result = collision.ContactResultMap()
        manager.contactTest(result, collision.ContactRequest(collision.ContactTestType_FIRST))
        if not result.empty():
            hits.append(index)
    return hits


def display_model(urdf, checkout, output):
    # Display precisely the collision geometries used in the task. Drop the
    # unrelated Gazebo extensions, controllers and textured decorative meshes.
    original = ET.fromstring(urdf.read_text().replace("tesseract:make_convex", "make_convex"))
    robot = ET.Element("robot", name="pr2")
    for link in original.findall("link"):
        node = ET.SubElement(robot, "link", name=link.get("name"))
        for shape in link.findall("collision"):
            visual = deepcopy(shape)
            visual.tag = "visual"
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                source = Path(mesh.get("filename").removeprefix("package://"))
                dest = output / source
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkout / source, dest)
                mesh.set("filename", source.as_posix())
            material = ET.SubElement(visual, "material", name="table" if link.get("name") == "table_link" else "pr2")
            color = "0.27 0.43 0.48 1" if link.get("name") == "table_link" else "0.72 0.76 0.8 1"
            ET.SubElement(material, "color", rgba=color)
            node.append(visual)
    for joint in original.findall("joint"):
        robot.append(deepcopy(joint))
    ET.ElementTree(robot).write(output / "pr2.urdf", encoding="utf-8", xml_declaration=True)
    return [link.get("name") for link in original.findall("link") if link.find("collision") is not None]


def track(env, names, path, links):
    samples = np.concatenate([np.linspace(a, b, 32, endpoint=False) for a, b in zip(path, path[1:])] + [path[-1:]])
    frames = []
    for q in samples:
        state = env.getState(names, q)
        poses = []
        for name in links:
            transform = state.link_transforms[name]
            quat = common.Quaterniond(transform.rotation())
            poses.append([*transform.translation().ravel(), quat.x(), quat.y(), quat.z(), quat.w()])
        frames.append(poses)
    return np.asarray(frames, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="checkout of tesseract-robotics/trajopt")
    checkout = parser.parse_args().checkout.resolve()
    data = checkout / "trajopt_common" / "data"
    urdf = data / "arm_around_table.urdf"
    case = json.loads((data / "config" / "arm_around_table.json").read_text())
    os.environ["TESSERACT_RESOURCE_PATH"] = str(checkout)
    locator = common.GeneralResourceLocator()
    env = environment.Environment()
    if not env.init(common.FilesystemPath(str(urdf)), common.FilesystemPath(str(data / "pr2.srdf")), locator):
        raise RuntimeError("Tesseract could not load the PR2 environment")
    group = env.getJointGroup(case["basic_info"]["manip"])
    seed, path = solve(env, group, case)
    initial_hits = colliding_segments(env, group, seed)
    final_hits = colliding_segments(env, group, path)
    if not initial_hits or final_hits:
        raise RuntimeError(f"Upstream collision acceptance failed. Seed {initial_hits}, optimized {final_hits}")
    output = Path(__file__).resolve().parent / "generated"
    output.mkdir(exist_ok=True)
    links = display_model(urdf, checkout, output)
    tracks = np.stack([track(env, group.getJointNames(), q, links) for q in (seed, path)])
    np.savez_compressed(output / "replay.npz", links=links, poses=tracks, seed=seed, optimized=path,
                        seed_hits=initial_hits)
    print(f"PR2 continuous collision check (zero-based segments). Seed {initial_hits}, optimized {final_hits}.")


if __name__ == "__main__":
    main()
