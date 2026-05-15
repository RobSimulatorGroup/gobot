import math
import gobot


ROBOT = "cartpole"
SLIDER_JOINT = "slider"
HINGE_JOINT = "hinge"
TARGET_CART_POSITION = 1.0
GRAVITY = (0.0, 0.0, -9.81)
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_LENGTH = 1.0
POLE_SIZE = (0.04, 0.04, POLE_LENGTH)
FORCE_LIMIT = 15.0
PRINT_EVERY_TICKS = 240

POLE_HALF_LENGTH = POLE_LENGTH * 0.5
POLE_INERTIA_ABOUT_CENTER = POLE_MASS * (POLE_LENGTH * POLE_LENGTH + POLE_SIZE[0] * POLE_SIZE[0]) / 12.0
POLE_INERTIA_ABOUT_HINGE = POLE_INERTIA_ABOUT_CENTER + POLE_MASS * POLE_HALF_LENGTH * POLE_HALF_LENGTH
CARTPOLE_TOTAL_MASS = CART_MASS + POLE_MASS
POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH
POLE_GRAVITY_MOMENT = POLE_MASS * 9.81 * POLE_HALF_LENGTH
SLIDER_LOWER_LIMIT = -20.0
SLIDER_UPPER_LIMIT = 20.0

# Energy-based swing-up parameters
SWINGUP_GAIN = 15.0
SWINGUP_VELOCITY_DAMPING = 1.2

# LQR balance parameters
LQR_Q = (1.0, 1.0, 80.0, 8.0)
LQR_R = 0.1
BALANCE_ENTER_ANGLE = 0.25
BALANCE_ENTER_ANGULAR_VELOCITY = 2.5
BALANCE_EXIT_ANGLE = 0.45
BALANCE_EXIT_ANGULAR_VELOCITY = 6.0
BALANCE_TARGET_RATE = 1.5

DEFAULT_CARTPOLE_LQR_GAIN = None


class CartPoleLQRController:
    def __init__(
        self,
        gain=DEFAULT_CARTPOLE_LQR_GAIN,
        target_position=1.0,
        force_limit=20.0,
    ):
        if gain is None:
            gain = _discrete_lqr_gain(LQR_Q, LQR_R)
        self.gain = tuple(float(value) for value in gain)
        self.target_position = float(target_position)
        self.force_limit = float(force_limit)
        self.last_effort = 0.0

    def reset(self):
        self.last_effort = 0.0

    def effort(self, x, x_dot, theta, theta_dot, target_position=None):
        target = float(target_position) if target_position is not None else self.target_position
        state = (x - target, x_dot, theta, theta_dot)
        value = sum(g * s for g, s in zip(self.gain, state))
        value = _clamp(value, -self.force_limit, self.force_limit)
        self.last_effort = value
        return value


class CartPoleEnergySwingUp:
    """Swing-up controller using energy shaping + acceleration coupling.

    Uses the pendulum's energy error to determine force magnitude,
    and applies force in the direction of sin(theta) to exploit the
    inertial coupling between cart acceleration and pendulum torque.
    """

    def __init__(self, force_limit=FORCE_LIMIT):
        self.force_limit = float(force_limit)
        self.target_energy = POLE_MASS * 9.81 * POLE_HALF_LENGTH

    def reset(self):
        pass

    def effort(self, x, x_dot, theta, theta_dot, target_position=0.0):
        # Pendulum energy relative to upright
        kinetic = 0.5 * POLE_INERTIA_ABOUT_HINGE * theta_dot * theta_dot
        potential = POLE_MASS * 9.81 * POLE_HALF_LENGTH * math.cos(theta)
        energy = kinetic + potential
        energy_error = energy - self.target_energy  # negative = need more energy

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        # Dead zone: pole hanging straight down with no velocity
        # Apply a constant force to break the equilibrium
        if abs(sin_theta) < 0.15 and abs(theta_dot) < 0.5 and cos_theta < -0.5:
            swing_effort = self.force_limit
        else:
            # Energy pumping (Astrom-Furuta sign-based):
            # F = gain * sign(E - E_d) * sign(cos(θ)*θ̇)
            pump_signal = cos_theta * theta_dot
            if abs(pump_signal) > 0.05:
                swing_effort = SWINGUP_GAIN * _sign(energy_error) * _sign(pump_signal)
            else:
                swing_effort = SWINGUP_GAIN * 0.3 * sin_theta

        # Velocity damping only — no position regulation during swing-up
        # The cart will drift but the sign-based controller naturally oscillates it
        # Hard brake only near rail limits
        position_effort = -SWINGUP_VELOCITY_DAMPING * x_dot
        dist_from_center = abs(x)
        if dist_from_center > 16.0:
            # Emergency: near rail limit, brake hard
            position_effort += -5.0 * x

        effort = swing_effort + position_effort
        return _clamp(effort, -self.force_limit, self.force_limit)


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
        self.swingup = CartPoleEnergySwingUp(force_limit=force_limit)
        self.mode = "swingup"
        self.balance_target = 0.0
        self.last_effort = 0.0
        self.last_action = 0.0

    def reset(self):
        self.mode = "swingup"
        self.balance_target = 0.0
        self.last_effort = 0.0
        self.last_action = 0.0
        self.lqr.reset()
        self.swingup.reset()

    def effort(self, observation, target_position=None, delta=0.0):
        x, x_dot, theta, theta_dot, target_error = _cartpole_state(observation)
        target = float(target_position) if target_position is not None else x + target_error
        if not math.isfinite(target):
            target = self.target_position
        self.target_position = target
        self.lqr.target_position = target

        can_balance = (
            abs(theta) <= BALANCE_ENTER_ANGLE
            and abs(theta_dot) <= BALANCE_ENTER_ANGULAR_VELOCITY
        )

        if self.mode == "balance":
            if abs(theta) > BALANCE_EXIT_ANGLE or abs(theta_dot) > BALANCE_EXIT_ANGULAR_VELOCITY:
                self.mode = "swingup"
        elif can_balance:
            self.mode = "balance"
            self.balance_target = x

        if self.mode == "balance":
            dt = _safe_delta(delta)
            self.balance_target = _move_toward(self.balance_target, target, BALANCE_TARGET_RATE * dt)
            value = self.lqr.effort(x, x_dot, theta, theta_dot, target_position=self.balance_target)
        else:
            value = self.swingup.effort(x, x_dot, theta, theta_dot, target_position=target)

        self.last_effort = value
        self.last_action = value / self.force_limit if self.force_limit > 0.0 else 0.0
        return value

    @property
    def straightening(self):
        return self.mode == "swingup"


def _cartpole_state(observation):
    values = [float(value) for value in observation]
    if len(values) < 5:
        raise ValueError(f"expected CartPole observation size 5, got {len(values)}")
    return tuple(0.0 if not math.isfinite(value) else value for value in values[:5])


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


def _sign(value):
    if value > 0.0:
        return 1.0
    elif value < 0.0:
        return -1.0
    return 0.0


def _safe_delta(delta):
    return float(delta) if delta > 0.0 and math.isfinite(delta) else 1.0 / 240.0


def _move_toward(value, target, max_delta):
    value = float(value)
    target = float(target)
    max_delta = abs(float(max_delta))
    if value < target:
        return min(value + max_delta, target)
    if value > target:
        return max(value - max_delta, target)
    return value


def _dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _mat_vec(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _transpose_mat_vec(matrix, vector):
    return [sum(matrix[row][col] * vector[row] for row in range(len(matrix))) for col in range(len(matrix[0]))]


def _mat_mul(left, right):
    cols = len(right[0])
    rows = len(left)
    inner = len(right)
    return [
        [sum(left[row][index] * right[index][col] for index in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


def _transpose_mat_mul(left, right):
    rows = len(left[0])
    cols = len(right[0])
    inner = len(left)
    return [
        [sum(left[index][row] * right[index][col] for index in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


def _add_diag(matrix, diagonal):
    result = [row[:] for row in matrix]
    for index, value in enumerate(diagonal):
        result[index][index] += float(value)
    return result


def _cartpole_dynamics(state, effort, delta):
    """RK4 integration of cartpole dynamics. Used only for LQR gain computation."""
    dt = float(delta) if delta > 0.0 and math.isfinite(delta) else 1.0 / 240.0
    effort = _clamp(effort, -FORCE_LIMIT, FORCE_LIMIT)

    def derivative(current):
        _, current_x_dot, current_theta, current_theta_dot = current
        if not (math.isfinite(current_theta) and math.isfinite(current_theta_dot)):
            return (0.0, 0.0, 0.0, 0.0)
        sin_theta = math.sin(current_theta)
        cos_theta = math.cos(current_theta)
        det = CARTPOLE_TOTAL_MASS * POLE_INERTIA_ABOUT_HINGE - (
            POLE_MASS_LENGTH * POLE_MASS_LENGTH * cos_theta * cos_theta
        )
        if abs(det) < 1.0e-9:
            det = 1.0e-9 if det >= 0.0 else -1.0e-9
        force_term = effort + POLE_MASS_LENGTH * sin_theta * current_theta_dot * current_theta_dot
        x_accel = (
            POLE_INERTIA_ABOUT_HINGE * force_term
            - POLE_MASS_LENGTH * cos_theta * POLE_GRAVITY_MOMENT * sin_theta
        ) / det
        theta_accel = (
            CARTPOLE_TOTAL_MASS * POLE_GRAVITY_MOMENT * sin_theta
            - POLE_MASS_LENGTH * cos_theta * force_term
        ) / det
        return (current_x_dot, x_accel, current_theta_dot, theta_accel)

    start = tuple(float(v) for v in state)
    if not all(math.isfinite(v) for v in start):
        return (0.0, 0.0, 0.0, 0.0)
    k1 = derivative(start)
    k2 = derivative(tuple(start[i] + 0.5 * dt * k1[i] for i in range(4)))
    k3 = derivative(tuple(start[i] + 0.5 * dt * k2[i] for i in range(4)))
    k4 = derivative(tuple(start[i] + dt * k3[i] for i in range(4)))
    nx = start[0] + dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
    nxd = start[1] + dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
    nt = _wrap_angle(start[2] + dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0)
    ntd = start[3] + dt * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6.0
    return (nx, nxd, nt, ntd)


def _state_error(state, target):
    return (
        float(state[0]) - float(target[0]),
        float(state[1]) - float(target[1]),
        _wrap_angle(float(state[2]) - float(target[2])),
        float(state[3]) - float(target[3]),
    )


def _linear_dynamics_matrices(state, effort, delta):
    eps = 1.0e-4
    a_matrix = [[0.0] * 4 for _ in range(4)]
    for col in range(4):
        plus = list(state)
        minus = list(state)
        plus[col] += eps
        minus[col] -= eps
        plus_next = _cartpole_dynamics(plus, effort, delta)
        minus_next = _cartpole_dynamics(minus, effort, delta)
        diff = _state_error(plus_next, minus_next)
        for row in range(4):
            a_matrix[row][col] = diff[row] / (2.0 * eps)
    plus_next = _cartpole_dynamics(state, effort + eps, delta)
    minus_next = _cartpole_dynamics(state, effort - eps, delta)
    diff = _state_error(plus_next, minus_next)
    b_vector = [value / (2.0 * eps) for value in diff]
    return a_matrix, b_vector


def _discrete_lqr_gain(q_diagonal, r_value):
    target = (0.0, 0.0, 0.0, 0.0)
    a_matrix, b_vector = _linear_dynamics_matrices(target, 0.0, 1.0 / 240.0)
    s_matrix = [[0.0] * 4 for _ in range(4)]
    for index, value in enumerate(q_diagonal):
        s_matrix[index][index] = float(value)

    gain = [0.0, 0.0, 0.0, 0.0]
    for _ in range(2000):
        sb = _mat_vec(s_matrix, b_vector)
        bsb = _dot(b_vector, sb)
        bsa = _transpose_mat_vec(a_matrix, sb)
        denominator = float(r_value) + bsb
        gain = [-value / denominator for value in bsa]
        closed_loop = [
            [a_matrix[row][col] + b_vector[row] * gain[col] for col in range(4)]
            for row in range(4)
        ]
        s_matrix = _add_diag(_transpose_mat_mul(a_matrix, _mat_mul(s_matrix, closed_loop)), q_diagonal)
    return tuple(gain)


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
            "CartPole energy swing-up + LQR started. gravity={} pole_length={:.2f}m force_limit={:.1f}N".format(
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
                "CartPole {} t={:.2f}s x={:.3f} x_dot={:.3f} theta={:.4f} theta_dot={:.3f} effort={:.3f}".format(
                    self.controller.mode,
                    self.context.simulation_time,
                    x,
                    x_dot,
                    theta,
                    theta_dot,
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
