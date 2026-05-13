import math
import gobot


ROBOT = "cartpole"
SLIDER_JOINT = "slider"
HINGE_JOINT = "hinge"
TARGET_CART_POSITION = 1.0
GRAVITY = (0.0, 0.0, -9.81)
POLE_MASS = 0.1
POLE_LENGTH = 1.0
POLE_SIZE = (0.04, 0.04, POLE_LENGTH)
FORCE_LIMIT = 10.0
PRINT_EVERY_TICKS = 240

POLE_HALF_LENGTH = POLE_LENGTH * 0.5
POLE_INERTIA_ABOUT_CENTER = POLE_MASS * (POLE_LENGTH * POLE_LENGTH + POLE_SIZE[0] * POLE_SIZE[0]) / 12.0
POLE_INERTIA_ABOUT_HINGE = POLE_INERTIA_ABOUT_CENTER + POLE_MASS * POLE_HALF_LENGTH * POLE_HALF_LENGTH

BALANCE_ENTER_ANGLE = 0.08
BALANCE_ENTER_ANGULAR_VELOCITY = 4.0
BALANCE_EXIT_ANGLE = 0.30
BALANCE_EXIT_ANGULAR_VELOCITY = 3.0

SWING_UP_ENERGY_GAIN = 5.0
SWING_UP_POSITION_GAIN = 3.0
SWING_UP_VELOCITY_DAMPING = 0.5


DEFAULT_CARTPOLE_LQR_GAIN = (
    1.0,
    1.5,
    20.0,
    5.0,
)


class CartPoleLQRController:
    def __init__(
        self,
        gain=DEFAULT_CARTPOLE_LQR_GAIN,
        target_position=1.0,
        force_limit=20.0,
        theta_reference=0.0,
    ):
        if len(gain) != 4:
            raise ValueError("CartPole LQR gain must contain 4 values")
        self.gain = tuple(float(value) for value in gain)
        self.target_position = float(target_position)
        self.force_limit = float(force_limit)
        self.theta_reference = float(theta_reference)
        self.last_effort = 0.0
        self.last_action = 0.0

    def reset(self):
        self.last_effort = 0.0
        self.last_action = 0.0

    def effort(self, observation, target_position=None):
        x, x_dot, theta, theta_dot, target_error = _cartpole_state(observation)
        target = float(target_position) if target_position is not None else x + target_error
        if not math.isfinite(target):
            target = self.target_position

        state = (
            x - target,
            x_dot,
            theta - self.theta_reference,
            theta_dot,
        )
        value = sum(gain * state_value for gain, state_value in zip(self.gain, state))
        value = max(-self.force_limit, min(self.force_limit, value))
        self.last_effort = value
        self.last_action = value / self.force_limit if self.force_limit > 0.0 else 0.0
        return value


class CartPoleHybridController:
    def __init__(
        self,
        target_position=1.0,
        force_limit=20.0,
        lqr_gain=DEFAULT_CARTPOLE_LQR_GAIN,
    ):
        self.target_position = float(target_position)
        self.force_limit = float(force_limit)
        self.lqr = CartPoleLQRController(
            gain=lqr_gain,
            target_position=target_position,
            force_limit=force_limit,
        )
        self.mode = "swing_up"
        self.balance_target = self.target_position
        self.last_effort = 0.0
        self.last_action = 0.0

    def reset(self):
        self.mode = "swing_up"
        self.balance_target = self.target_position
        self.last_effort = 0.0
        self.last_action = 0.0
        self.lqr.reset()

    def effort(self, observation, target_position=None, delta=0.0):
        x, x_dot, theta, theta_dot, target_error = _cartpole_state(observation)
        target = float(target_position) if target_position is not None else x + target_error
        if not math.isfinite(target):
            target = self.target_position
        self.target_position = target
        self.lqr.target_position = target

        if self.mode == "balance":
            if abs(theta) > BALANCE_EXIT_ANGLE or abs(theta_dot) > BALANCE_EXIT_ANGULAR_VELOCITY:
                self.mode = "swing_up"
        elif (
            abs(theta) <= BALANCE_ENTER_ANGLE
            and abs(theta_dot) <= BALANCE_ENTER_ANGULAR_VELOCITY
        ):
            self.mode = "balance"

        if self.mode == "balance":
            self.balance_target = target
            balance_observation = [x, x_dot, theta, theta_dot, self.balance_target - x]
            value = self.lqr.effort(balance_observation, target_position=self.balance_target)
        else:
            value = _swing_up_effort(x, x_dot, theta, theta_dot, target)
            value = max(-self.force_limit, min(self.force_limit, value))

        self.last_effort = value
        self.last_action = value / self.force_limit if self.force_limit > 0.0 else 0.0
        return value

    @property
    def straightening(self):
        return self.mode == "swing_up"


def _cartpole_state(observation):
    values = [float(value) for value in observation]
    if len(values) < 5:
        raise ValueError(f"expected CartPole observation size 5, got {len(values)}")
    return tuple(0.0 if not math.isfinite(value) else value for value in values[:5])


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _swing_up_effort(x, x_dot, theta, theta_dot, target):
    energy = (
        0.5 * POLE_INERTIA_ABOUT_HINGE * theta_dot * theta_dot
        + POLE_MASS * abs(GRAVITY[2]) * POLE_HALF_LENGTH * (math.cos(theta) - 1.0)
    )
    effort = (
        SWING_UP_ENERGY_GAIN * energy * theta_dot * math.cos(theta)
        - SWING_UP_POSITION_GAIN * (x - target)
        - SWING_UP_VELOCITY_DAMPING * x_dot
    )
    return effort


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
        self.controller = CartPoleHybridController(
            target_position=TARGET_CART_POSITION,
            force_limit=FORCE_LIMIT,
        )
        self.target_cart_position = TARGET_CART_POSITION
        self.previous_x = float(self.slider.joint_position)
        self.previous_theta = _wrap_angle(float(self.hinge.joint_position))
        self.observation = self._observation(0.0)
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False

        self.slider.effort_limit = FORCE_LIMIT
        self.slider.velocity_limit = max(float(getattr(self.slider, "velocity_limit", 0.0)), 20.0)
        self.hinge.effort_limit = 0.0
        print(
            "CartPole hybrid NodeScript started. gravity={} pole_length={:.2f}m force_limit={:.1f}N".format(
                tuple(float(value) for value in self.context.gravity),
                POLE_LENGTH,
                FORCE_LIMIT,
            )
        )

    def _process(self, delta: float):
        pass

    def _physics_process(self, delta: float):
        if not self.playing:
            return

        if not self._ensure_world_controls():
            return
        self.observation = self._observation(delta)
        effort = self.controller.effort(
            self.observation,
            target_position=self.target_cart_position,
            delta=delta,
        )
        self.context.set_joint_effort_target(self.robot.name, SLIDER_JOINT, effort)

        self.ticks += 1
        if PRINT_EVERY_TICKS > 0 and self.ticks % PRINT_EVERY_TICKS == 0:
            x, x_dot, theta, theta_dot, target_error = self.observation
            print(
                "CartPole {} t={:.2f}s x={:.3f} x_dot={:.3f} theta={:.4f} theta_dot={:.3f} target_error={:.3f} effort={:.3f}".format(
                    self.controller.mode,
                    self.context.simulation_time,
                    x,
                    x_dot,
                    theta,
                    theta_dot,
                    target_error,
                    effort,
                )
            )

    def reset(self):
        self.controller.reset()
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
        self.controller.target_position = self.target_cart_position

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
            raise RuntimeError("CartPole LQR script has no scene root")
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
