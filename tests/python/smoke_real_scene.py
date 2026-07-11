import argparse
import json
import math
import pathlib

import gobot
from gobot.rl.tasks.go1 import (
    GO1_DEFAULT_BASE_POSITION,
    GO1_DEFAULT_JOINT_POS,
    GO1_JOINT_NAMES,
)


def _find_node_by_name(node, name):
    if node is None:
        return None
    if node.name == name:
        return node
    for child in node.children:
        found = _find_node_by_name(child, name)
        if found is not None:
            return found
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--scene", default="res://world.jscn")
    parser.add_argument("--backend", default="mujoco", choices=["null", "mujoco"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--expect-go1-stand", action="store_true")
    parser.add_argument("--expect-go1-sensors", action="store_true")
    parser.add_argument("--expect-empty-robot-source-path", action="store_true")
    args = parser.parse_args()

    context = gobot.app.context()
    context.set_project_path(args.project)
    scene_path = pathlib.Path(args.scene.replace("res://", args.project + "/", 1))

    if args.expect_empty_robot_source_path:
        scene_json = json.loads(scene_path.read_text())
        robot_nodes = [
            node for node in scene_json.get("__NODES__", [])
            if node.get("type") == "Robot3D"
        ]
        if not robot_nodes:
            raise AssertionError(f"No Robot3D nodes found in {scene_path}")
        for node in robot_nodes:
            source_path = node.get("properties", {}).get("source_path", "")
            if source_path:
                raise AssertionError(
                    f"Robot3D '{node.get('name', '<unnamed>')}' still depends on source_path={source_path!r}"
                )

    root = context.load_scene(args.scene)
    print(f"loaded scene root={root.name} type={root.type} children={root.child_count}")
    context.build_world(gobot.PhysicsBackendType.MuJoCoCpu if args.backend == "mujoco"
                        else gobot.PhysicsBackendType.Null)
    context.reset_simulation()
    go1 = root.find("go1")
    if go1 is None:
        go1 = root if root.name == "go1" else None

    if args.expect_go1_stand:
        if go1 is None:
            raise AssertionError("Loaded scene has no Robot3D node 'go1'")
        base_link = _find_node_by_name(go1, "trunk")
        if base_link is None:
            raise AssertionError("Go1 scene has no trunk link")
        joints = {}
        for name, target in zip(GO1_JOINT_NAMES, GO1_DEFAULT_JOINT_POS):
            joint = _find_node_by_name(go1, name)
            if joint is None:
                raise AssertionError(f"Go1 scene has no joint {name!r}")
            joints[name] = joint

        base_link.reset_runtime_state(GO1_DEFAULT_BASE_POSITION,
                                      [1.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0])
        for name, target in zip(GO1_JOINT_NAMES, GO1_DEFAULT_JOINT_POS):
            joints[name].reset_runtime_state(target, 0.0)
            joints[name].set_position_target(target)

        context.step_once()

        for index in range(args.steps):
            for name, target in zip(GO1_JOINT_NAMES, GO1_DEFAULT_JOINT_POS):
                joints[name].set_position_target(target)
            context.step_once()
            print(f"step={index + 1} time={context.simulation_time:.6f} frame={context.frame_count}")

        base = base_link.get_runtime_state()
        base_z = base["global_transform"]["position"][2]
        print(f"go1_stand_base_z={base_z:.6f}")
        if base_z <= 0.15:
            raise AssertionError(f"Go1 default stand base height is too low: {base_z:.6f}")
    else:
        for index in range(args.steps):
            context.step_once()
            print(f"step={index + 1} time={context.simulation_time:.6f} frame={context.frame_count}")

    if args.expect_go1_sensors:
        if go1 is None:
            raise AssertionError("Loaded scene has no Robot3D node 'go1'")
        sensors = {}
        for name in ("imu", "root_angmom", "terrain_scan"):
            sensor_node = _find_node_by_name(go1, name)
            if sensor_node is not None:
                state = sensor_node.get_runtime_state()
                sensors[state["sensor_name"]] = state
        for foot in ("FR", "FL", "RR", "RL"):
            for suffix in ("foot_height_scan", "foot_contact"):
                sensor_node = _find_node_by_name(go1, f"{foot}_{suffix}")
                if sensor_node is not None:
                    state = sensor_node.get_runtime_state()
                    sensors[state["sensor_name"]] = state
        print(f"go1_sensor_names={sorted(sensors)}")

        imu = sensors.get("imu")
        if imu is None:
            raise AssertionError("Go1 runtime state is missing IMUSensor3D 'imu'")
        if imu["type"] != "imu":
            raise AssertionError(f"Go1 imu sensor has unexpected type {imu['type']!r}")
        if len(imu["values"]) != 13:
            raise AssertionError(f"Go1 imu sensor expected 13 values, got {len(imu['values'])}")
        if imu["channel_names"][7:10] != [
            "linear_velocity_x",
            "linear_velocity_y",
            "linear_velocity_z",
        ]:
            raise AssertionError(f"Go1 imu channel names missing linear velocity: {imu['channel_names']!r}")

        angular_momentum = sensors.get("root_angmom")
        if angular_momentum is None:
            raise AssertionError("Go1 runtime state is missing AngularMomentumSensor3D 'root_angmom'")
        if angular_momentum["type"] != "angular_momentum":
            raise AssertionError(
                f"Go1 angular momentum sensor has unexpected type {angular_momentum['type']!r}"
            )
        if len(angular_momentum["values"]) != 3:
            raise AssertionError(
                f"Go1 angular momentum sensor expected 3 values, got {len(angular_momentum['values'])}"
            )

        terrain_scan = sensors.get("terrain_scan")
        if terrain_scan is None:
            raise AssertionError("Go1 runtime state is missing HeightScanner3D 'terrain_scan'")
        if terrain_scan["type"] != "height_scanner":
            raise AssertionError(f"Go1 terrain scan sensor has unexpected type {terrain_scan['type']!r}")
        if len(terrain_scan["values"]) != 187:
            raise AssertionError(f"Go1 terrain scan expected 187 values, got {len(terrain_scan['values'])}")
        if terrain_scan["channel_names"][:3] != ["distance_0", "distance_1", "distance_2"]:
            raise AssertionError(f"Go1 terrain scan channel names are unexpected: {terrain_scan['channel_names']!r}")
        if "global_transform" not in terrain_scan:
            raise AssertionError("Go1 terrain scan runtime state is missing global_transform")
        if len(terrain_scan.get("hits", [])) != 187:
            raise AssertionError("Go1 terrain scan runtime state is missing raycast hits")

        for foot in ("FR", "FL", "RR", "RL"):
            height_sensor = sensors.get(f"{foot}_foot_height_scan")
            if height_sensor is None:
                raise AssertionError(f"Go1 runtime state is missing {foot}_foot_height_scan")
            if height_sensor["type"] != "terrain_height" or len(height_sensor["values"]) != 1:
                raise AssertionError(f"Go1 {foot} foot height sensor is malformed: {height_sensor!r}")
            contact_sensor = sensors.get(f"{foot}_foot_contact")
            if contact_sensor is None:
                raise AssertionError(f"Go1 runtime state is missing {foot}_foot_contact")
            if contact_sensor["type"] != "contact" or len(contact_sensor["values"]) != 1:
                raise AssertionError(f"Go1 {foot} foot contact sensor is malformed: {contact_sensor!r}")

        checked_sensors = [imu, angular_momentum, terrain_scan]
        checked_sensors += [
            sensors[f"{foot}_foot_height_scan"]
            for foot in ("FR", "FL", "RR", "RL")
        ]
        checked_sensors += [
            sensors[f"{foot}_foot_contact"]
            for foot in ("FR", "FL", "RR", "RL")
        ]
        for sensor in checked_sensors:
            if args.backend == "mujoco" and sensor["timestamp"] <= 0.0:
                raise AssertionError(f"Go1 sensor {sensor['sensor_name']} timestamp did not advance")
            if not all(math.isfinite(float(value)) for value in sensor["values"]):
                raise AssertionError(f"Go1 sensor {sensor['sensor_name']} produced non-finite values")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
