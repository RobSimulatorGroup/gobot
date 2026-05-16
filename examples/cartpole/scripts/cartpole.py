import math
import os

import gobot


ROBOT = "cartpole"
SLIDER_JOINT = "slider"
HINGE_JOINT = "hinge"
TARGET_CART_POSITION = 1.0
FORCE_LIMIT = 20.0
PRINT_EVERY_TICKS = 240
DEFAULT_POLICY_PATH = "res://policies/cartpole.pt"


class TorchPolicy:
    def __init__(self, path):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        self.policy = torch.jit.load(path, map_location=self.device)
        self.policy.eval()

    def action(self, observation):
        with self.torch.no_grad():
            obs = self.torch.as_tensor([observation], dtype=self.torch.float32, device=self.device)
            output = self.policy(obs)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return float(output.reshape(-1)[0].clamp(-1.0, 1.0).cpu().item())


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


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


class Script(gobot.NodeScript):
    def _ready(self):
        self.robot = self._find_robot()
        self.slider = self._find_joint(SLIDER_JOINT, ("rail/slider", "slider"))
        self.hinge = self._find_joint(HINGE_JOINT, ("rail/slider/cart/hinge", "cart/hinge", "hinge"))
        self.target_cart_position = TARGET_CART_POSITION
        self.previous_x = float(self.slider.joint_position)
        self.previous_theta = _wrap_angle(float(self.hinge.joint_position))
        self.observation = self._observation(0.0)
        self.policy = self._load_policy()
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False

        self.slider.effort_limit = FORCE_LIMIT
        self.slider.velocity_limit = max(float(getattr(self.slider, "velocity_limit", 0.0)), 20.0)
        self.hinge.effort_limit = 0.0
        print(
            "CartPole RL policy playback started. policy={} force_limit={:.1f}N".format(
                "loaded" if self.policy is not None else "missing",
                FORCE_LIMIT,
            )
        )

    def _process(self, delta):
        pass

    def _physics_process(self, delta):
        if not self.playing:
            return
        if not self._ensure_world_controls():
            return

        self.observation = self._observation(delta)
        action = 0.0
        if self.policy is not None:
            action = self.policy.action(self.observation)
        action = _clamp(action, -1.0, 1.0)
        effort = action * FORCE_LIMIT
        self.context.set_joint_effort_target(self.robot.name, SLIDER_JOINT, effort)

        self.ticks += 1
        if PRINT_EVERY_TICKS > 0 and self.ticks % PRINT_EVERY_TICKS == 0:
            x, x_dot, theta, theta_dot, target_error = self.observation
            print(
                "CartPole RL t={:.2f}s x={:.3f} x_dot={:.3f} theta={:.4f} theta_dot={:.3f} action={:.3f}".format(
                    self.context.simulation_time,
                    x,
                    x_dot,
                    theta,
                    theta_dot,
                    action,
                )
            )

    def reset(self):
        self.previous_x = float(self.slider.joint_position)
        self.previous_theta = _wrap_angle(float(self.hinge.joint_position))
        self.observation = self._observation(0.0)
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def set_target(self, position):
        self.target_cart_position = float(position)

    def _load_policy(self):
        path = os.environ.get("GOBOT_CARTPOLE_POLICY", DEFAULT_POLICY_PATH)
        if path.startswith("res://"):
            project_path = self.context.project_path if self.context is not None else ""
            path = os.path.join(project_path, path.removeprefix("res://"))
        if not path or not os.path.exists(path):
            print("CartPole RL policy not found; set GOBOT_CARTPOLE_POLICY to a TorchScript policy.")
            return None
        try:
            return TorchPolicy(path)
        except Exception as error:
            print(f"CartPole RL policy load failed: {error}")
            return None

    def _ensure_world_controls(self):
        if self.world_controls_ready:
            return True
        if not self.context.has_world:
            return False
        self.context.set_joint_passive(self.robot.name, HINGE_JOINT)
        self.world_controls_ready = True
        return True

    def _find_robot(self):
        root = self.get_root()
        if root is None:
            raise RuntimeError("CartPole script has no scene root")
        if root.name == ROBOT:
            return root
        robot = root.find(ROBOT) or _find_node_by_name(root, ROBOT)
        if robot is None:
            raise RuntimeError(f"scene has no robot node '{ROBOT}'")
        return robot

    def _find_joint(self, name, paths):
        for path in paths:
            node = self.robot.find(path)
            if node is not None:
                return node
        node = _find_node_by_name(self.robot, name)
        if node is None:
            raise RuntimeError(f"robot '{self.robot.name}' has no joint '{name}'")
        return node

    def _observation(self, delta):
        runtime_joints = self._runtime_joint_states()
        if runtime_joints:
            slider = runtime_joints.get(SLIDER_JOINT)
            hinge = runtime_joints.get(HINGE_JOINT)
            if slider is not None and hinge is not None:
                x = float(slider.get("position", 0.0))
                theta = _wrap_angle(float(hinge.get("position", 0.0)))
                x_dot = float(slider.get("velocity", 0.0))
                theta_dot = float(hinge.get("velocity", 0.0))
                self.previous_x = x
                self.previous_theta = theta
                return [x, x_dot, theta, theta_dot, self.target_cart_position - x]

        x = float(self.slider.joint_position)
        theta = _wrap_angle(float(self.hinge.joint_position))
        if delta > 0.0:
            x_dot = (x - self.previous_x) / float(delta)
            theta_dot = _wrap_angle(theta - self.previous_theta) / float(delta)
        else:
            x_dot = 0.0
            theta_dot = 0.0
        self.previous_x = x
        self.previous_theta = theta
        return [x, x_dot, theta, theta_dot, self.target_cart_position - x]

    def _runtime_joint_states(self):
        if not self.context.has_world:
            return None
        state = self.context.get_runtime_state()
        for robot in state.get("robots", []):
            if robot.get("name") == self.robot.name:
                return {joint.get("name"): joint for joint in robot.get("joints", [])}
        return None
