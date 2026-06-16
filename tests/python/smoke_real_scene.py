import argparse
import importlib.util
import json
import math
import os
import pathlib
import gobot


GO1_RESET_MIN_CONTACT_DISTANCE = -0.025


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
        script_path = pathlib.Path(args.project) / "scripts" / "go1.py"
        spec = importlib.util.spec_from_file_location("go1_scene_script", script_path)
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)
        os.environ["GOBOT_GO1_POLICY"] = ""

        if go1 is None:
            raise AssertionError("Loaded scene has no Robot3D node 'go1'")
        go1.reset_link_state("trunk", script_module.RESET_BASE_POSITION,
                             [1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0])
        for name, target in zip(script_module.JOINT_NAMES, script_module.DEFAULT_POS):
            go1.reset_joint_state(name, target, 0.0)
            go1.set_joint_position_target(name, target)

        context.step_once()
        state = go1.get_runtime_state()
        contact_distances = [
            float(contact["distance"])
            for contact in state.get("contacts", [])
            if contact.get("robot_name") == "go1"
        ]
        if contact_distances:
            min_contact_distance = min(contact_distances)
            print(f"go1_reset_min_contact_distance={min_contact_distance:.6f}")
            if min_contact_distance < GO1_RESET_MIN_CONTACT_DISTANCE:
                raise AssertionError(
                    f"Go1 reset pose starts too far inside the ground: {min_contact_distance:.6f}"
                )

        for index in range(args.steps):
            for name, target in zip(script_module.JOINT_NAMES, script_module.DEFAULT_POS):
                go1.set_joint_position_target(name, target)
            context.step_once()
            print(f"step={index + 1} time={context.simulation_time:.6f} frame={context.frame_count}")

        robot = go1.get_runtime_state()
        base = next(link for link in robot["links"] if link["link_name"] == "trunk")
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
        robot = go1.get_runtime_state()
        sensors = {sensor["sensor_name"]: sensor for sensor in robot.get("sensors", [])}
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
