"""High-level pure Python helpers for authoring common Gobot scenes."""

from __future__ import annotations

from . import _core


def create_cartpole_scene(
    name: str = "cartpole",
    cart_mass: float = 1.0,
    cart_size: tuple[float, float, float] = (0.35, 0.22, 0.18),
    pole_mass: float = 0.3,
    pole_length: float = 0.3,
    pole_radius: float = 0.03,
    slider_range: float = 2.4,
    slider_effort: float = 20.0,
    slider_damping: float = 1.0,
    hinge_damping: float = 0.05,
):
    """Create a Gobot-native CartPole scene using public scene bindings.

    Default parameters are tuned for stable PPO training with a target-reaching
    task (move cart from 0 to 1 while balancing the pole).
    """

    robot = _core.create_node("Robot3D", name)
    robot.mode = _core.RobotMode.Motion

    rail = _core.create_node("Link3D", "rail")
    rail.role = _core.LinkRole.VirtualRoot

    slider = _core.create_node("Joint3D", "slider")
    slider.joint_type = _core.JointType.Prismatic
    slider.parent_link = "rail"
    slider.child_link = "cart"
    slider.axis = (1.0, 0.0, 0.0)
    slider.lower_limit = -slider_range
    slider.upper_limit = slider_range
    slider.effort_limit = slider_effort
    slider.velocity_limit = 20.0
    slider.damping = slider_damping

    cart = _core.create_node("Link3D", "cart")
    cart.has_inertial = True
    cart.mass = cart_mass
    cart.inertia_diagonal = (0.02, 0.02, 0.02)
    cart.add_child(_core.create_box_visual("cart_visual", cart_size))
    cart.add_child(_core.create_box_collision("cart_collision", cart_size))

    hinge = _core.create_node("Joint3D", "hinge")
    hinge.joint_type = _core.JointType.Continuous
    hinge.parent_link = "cart"
    hinge.child_link = "pole"
    hinge.axis = (0.0, 1.0, 0.0)
    hinge.position = (0.0, 0.0, cart_size[2] / 2.0 + pole_radius)
    hinge.velocity_limit = 20.0
    hinge.damping = hinge_damping

    pole_size = (pole_radius * 2, pole_radius * 2, pole_length)
    pole = _core.create_node("Link3D", "pole")
    pole.position = (0.0, 0.0, pole_length / 2.0)
    pole.has_inertial = True
    pole.mass = pole_mass
    pole.center_of_mass = (0.0, 0.0, 0.0)
    # Thin rod inertia: I_xx = I_yy = m*L^2/12, I_zz ≈ m*r^2/2
    rod_inertia = pole_mass * pole_length ** 2 / 12.0
    pole.inertia_diagonal = (rod_inertia, rod_inertia, pole_mass * pole_radius ** 2 / 2.0)
    pole.add_child(_core.create_box_visual("pole_visual", pole_size))
    pole.add_child(_core.create_box_collision("pole_collision", pole_size))

    robot.add_child(rail)
    rail.add_child(slider)
    slider.add_child(cart)
    cart.add_child(hinge)
    hinge.add_child(pole)
    return robot


def save_cartpole_scene(path: str = "res://cartpole.jscn", **kwargs):
    """Create and save a CartPole scene, returning the authored root node."""

    root = create_cartpole_scene(**kwargs)
    _core.save_scene(root, path)
    return root


__all__ = ["create_cartpole_scene", "save_cartpole_scene"]
