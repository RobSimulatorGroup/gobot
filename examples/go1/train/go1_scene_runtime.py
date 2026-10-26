"""Compatibility imports for the Go1 scene preparation helpers."""
from gobot.rl.locomotion.scene import (
    configure_robot_drives, find_robot_node, find_terrain_node, joint_gain_array,
    prepare_velocity_scene as prepare_go1_scene, terrain_generator_config, terrain_spawn_origins,
)
