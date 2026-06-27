"""Regenerate the Go1 robot scene from the UniLab-aligned MJCF asset."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from _repo_imports import prefer_repo_gobot

prefer_repo_gobot()

import gobot


FOOT_LINKS: dict[str, str] = {
    "FR": "FR_calf",
    "FL": "FL_calf",
    "RR": "RR_calf",
    "RL": "RL_calf",
}

FOOT_SAMPLE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.04, 0.0, 0.0),
    (0.0, 0.04, 0.0),
    (-0.04, 0.0, 0.0),
    (0.0, -0.04, 0.0),
)


def _required_child(root: gobot.Node, path: str) -> gobot.Node:
    node = root.find(path)
    if node is None:
        raise RuntimeError(f"expected Go1 scene node '{path}' while refreshing robot sensors")
    return node


def _find_by_name(root: gobot.Node, name: str) -> gobot.Node | None:
    if root.name == name:
        return root
    for child in root.children:
        found = _find_by_name(child, name)
        if found is not None:
            return found
    return None


def _required_name(root: gobot.Node, name: str) -> gobot.Node:
    node = _find_by_name(root, name)
    if node is None:
        raise RuntimeError(f"expected Go1 scene node named '{name}' while refreshing robot sensors")
    return node


def _remove_if_present(root: gobot.Node, path: str) -> None:
    node = root.find(path)
    if node is not None:
        node.remove(delete=True)


def _configure_sensor(sensor: gobot.Sensor3D) -> None:
    sensor.enabled = True
    sensor.sensor_period = 0.0
    sensor.noise_stddev = 0.0
    sensor.visualize_debug = True


def _add_trunk_sensors(root: gobot.Node) -> None:
    trunk = _required_name(root, "trunk")
    for name in ("root_angmom", "imu", "gyro", "terrain_scan"):
        _remove_if_present(trunk, name)

    root_angmom = gobot.create_node("AngularMomentumSensor3D", "root_angmom")
    _configure_sensor(root_angmom)
    trunk.add_child(root_angmom)

    imu = gobot.create_node("IMUSensor3D", "imu")
    _configure_sensor(imu)
    imu.position = (0.0, 0.0, 0.0)
    trunk.add_child(imu)

    terrain_scan = gobot.create_node("HeightScanner3D", "terrain_scan")
    _configure_sensor(terrain_scan)
    terrain_scan.pattern_mode = gobot.RayPatternMode.Grid
    terrain_scan.grid_size = (1.6, 1.0)
    terrain_scan.grid_resolution = 0.1
    terrain_scan.ray_alignment = gobot.RayAlignmentMode.Yaw
    terrain_scan.ray_direction = (0.0, 0.0, -1.0)
    terrain_scan.ray_direction_world_space = True
    terrain_scan.max_distance = 5.0
    terrain_scan.reduction_mode = gobot.RayReductionMode.None_
    trunk.add_child(terrain_scan)


def _add_foot_sensors(root: gobot.Node) -> None:
    for foot_name, link_name in FOOT_LINKS.items():
        link = _required_name(root, link_name)
        for name in (f"{foot_name}_foot_height_scan", f"{foot_name}_foot_contact"):
            _remove_if_present(link, name)

        height_scan = gobot.create_node("TerrainHeightSensor3D", f"{foot_name}_foot_height_scan")
        _configure_sensor(height_scan)
        height_scan.position = (0.0, 0.0, -0.213)
        height_scan.sample_offsets = list(FOOT_SAMPLE_OFFSETS)
        height_scan.pattern_mode = gobot.RayPatternMode.Custom
        height_scan.grid_size = (1.0, 1.0)
        height_scan.grid_resolution = 0.1
        height_scan.ray_alignment = gobot.RayAlignmentMode.Yaw
        height_scan.ray_direction = (0.0, 0.0, -1.0)
        height_scan.ray_direction_world_space = True
        height_scan.max_distance = 1.0
        height_scan.reduction_mode = gobot.RayReductionMode.Min
        link.add_child(height_scan)

        contact = gobot.create_node("ContactSensor3D", f"{foot_name}_foot_contact")
        _configure_sensor(contact)
        contact.position = (0.0, 0.0, -0.213)
        contact.radius = 0.03
        contact.min_threshold = 0.0
        contact.max_threshold = 0.0
        link.add_child(contact)


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    gobot.set_project_path(str(project_path))
    gobot.import_mjcf_scene("res://assets/xml/go1.xml", "res://go1.jscn", name="go1")
    scene = gobot.load_scene("res://go1.jscn")
    root = scene.root
    _add_trunk_sensors(root)
    _add_foot_sensors(root)
    gobot.save_scene(root, "res://go1.jscn")


if __name__ == "__main__":
    main()
