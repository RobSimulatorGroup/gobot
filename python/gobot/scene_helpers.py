"""High-level pure Python helpers for authoring common Gobot scenes."""

from __future__ import annotations

from . import _core


def create_cartpole_scene(
    name: str = "cartpole",
    cart_size: tuple[float, float, float] = (0.35, 0.22, 0.18),
    pole_size: tuple[float, float, float] = (0.05, 0.05, 1.0),
):
    """Create a Gobot-native CartPole scene using public scene bindings."""

    robot = _core.create_node("Robot3D", name)
    robot.mode = _core.RobotMode.Motion

    rail = _core.create_node("Link3D", "rail")
    rail.role = _core.LinkRole.VirtualRoot

    slider = _core.create_node("Joint3D", "slider")
    slider.joint_type = _core.JointType.Prismatic
    slider.parent_link = "rail"
    slider.child_link = "cart"
    slider.axis = (1.0, 0.0, 0.0)
    slider.lower_limit = -2.4
    slider.upper_limit = 2.4
    slider.effort_limit = 40.0
    slider.velocity_limit = 20.0

    cart = _core.create_node("Link3D", "cart")
    cart.has_inertial = True
    cart.mass = 1.0
    cart.inertia_diagonal = (0.02, 0.02, 0.02)
    cart.add_child(_core.create_box_visual("cart_visual", cart_size))
    cart.add_child(_core.create_box_collision("cart_collision", cart_size))

    hinge = _core.create_node("Joint3D", "hinge")
    hinge.joint_type = _core.JointType.Continuous
    hinge.parent_link = "cart"
    hinge.child_link = "pole"
    hinge.axis = (0.0, 1.0, 0.0)
    hinge.position = (0.0, 0.0, 0.12)
    hinge.velocity_limit = 20.0

    pole = _core.create_node("Link3D", "pole")
    pole.position = (0.0, 0.0, 0.5)
    pole.has_inertial = True
    pole.mass = 0.1
    pole.center_of_mass = (0.0, 0.0, 0.0)
    pole.inertia_diagonal = (0.003, 0.003, 0.0002)
    pole.add_child(_core.create_box_visual("pole_visual", pole_size))
    pole.add_child(_core.create_box_collision("pole_collision", pole_size))

    robot.add_child(rail)
    rail.add_child(slider)
    slider.add_child(cart)
    cart.add_child(hinge)
    hinge.add_child(pole)
    return robot


def save_cartpole_scene(path: str = "res://cartpole.jscn", name: str = "cartpole"):
    """Create and save a CartPole scene, returning the authored root node."""

    root = create_cartpole_scene(name=name)
    _core.save_scene(root, path)
    return root


__all__ = ["create_cartpole_scene", "save_cartpole_scene"]
