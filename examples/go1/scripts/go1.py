import math
import os
from importlib.metadata import PackageNotFoundError, version

import gobot


ROBOT = "go1"
BASE_LINK = "trunk"
DEFAULT_POLICY_PATH = "res://policies/go1.onnx"
TORCH_POLICY_PATH = "res://policies/go1.pt"
PRINT_EVERY_TICKS = 240
FIXED_TIME_STEP = 0.002
RESET_BASE_POSITION = [0.0, 0.0, 0.27]
COMMAND = [
    float(os.environ.get("GOBOT_GO1_VX", "0.0")),
    float(os.environ.get("GOBOT_GO1_VY", "0.0")),
    float(os.environ.get("GOBOT_GO1_YAW", "0.0")),
]
KEYBOARD_COMMAND_MAX = [0.6, 0.35, 1.2]
COMMAND_SMOOTHING = 8.0
COMMAND_ACTIVE_DEADBAND = 0.02
FALLEN_BASE_Z = 0.18
FALLEN_ROLL_PITCH = 0.8
KEYBOARD_BINDINGS = {
    "forward": ("W", "Up"),
    "backward": ("S", "Down"),
    "turn_left": ("A", "Left"),
    "turn_right": ("D", "Right"),
    "strafe_left": ("Q",),
    "strafe_right": ("E",),
    "stop": ("Space",),
    "reset": ("R",),
}
JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
DEFAULT_POS = [
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
    0.0,
    0.9,
    -1.8,
]
ACTION_SCALE = 0.25
KP = 40.0
KD = 1.0
DECIMATION = 10
POSITION_LIMITS = {
    "FR_hip_joint": (-0.863, 0.863),
    "FR_thigh_joint": (-0.686, 4.501),
    "FR_calf_joint": (-2.818, -0.888),
    "FL_hip_joint": (-0.863, 0.863),
    "FL_thigh_joint": (-0.686, 4.501),
    "FL_calf_joint": (-2.818, -0.888),
    "RR_hip_joint": (-0.863, 0.863),
    "RR_thigh_joint": (-0.686, 4.501),
    "RR_calf_joint": (-2.818, -0.888),
    "RL_hip_joint": (-0.863, 0.863),
    "RL_thigh_joint": (-0.686, 4.501),
    "RL_calf_joint": (-2.818, -0.888),
}


def _parse_version_prefix(value):
    parts = []
    for part in str(value).split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_onnxruntime_version():
    try:
        runtime_version = version("onnxruntime")
    except PackageNotFoundError as error:
        raise ImportError("onnxruntime is not installed; install gobot or onnxruntime>=1.19.") from error

    parsed_version = _parse_version_prefix(runtime_version)
    if parsed_version and parsed_version < (1, 19):
        raise ImportError(
            f"onnxruntime {runtime_version} is installed; NumPy 2 playback requires onnxruntime>=1.19."
        )


class OnnxPolicy:
    def __init__(self, path):
        _check_onnxruntime_version()

        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def action(self, observation):
        obs = self.np.asarray(observation, dtype=self.np.float32).reshape(1, -1)
        output = self.session.run([self.output_name], {self.input_name: obs})[0].reshape(-1)
        return self.np.clip(output, -1.0, 1.0).astype(float).tolist()


class TorchPolicy:
    def __init__(self, path):
        import torch
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        self.torch = torch
        self.device = torch.device("cpu")
        obs = TensorDict(
            {"policy": torch.zeros((1, 48), dtype=torch.float32, device=self.device)},
            batch_size=[1],
            device=self.device,
        )
        self.policy = MLPModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            len(JOINT_NAMES),
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ).to(self.device)
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.policy.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.policy.eval()

    def action(self, observation):
        with self.torch.no_grad():
            from tensordict import TensorDict

            obs = self.torch.as_tensor(observation, dtype=self.torch.float32, device=self.device).reshape(1, -1)
            obs_td = TensorDict({"policy": obs}, batch_size=[1], device=self.device)
            output = self.policy(obs_td)
            if isinstance(output, (tuple, list)):
                output = output[0]
        return output.reshape(-1).clamp(-1.0, 1.0).cpu().tolist()


def _resolve_project_path(context, path):
    if path.startswith("res://"):
        project_path = context.project_path if context is not None else ""
        path = os.path.join(project_path, path.removeprefix("res://"))
    return path


def _clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


def _key_held(input_state, action):
    return any(input_state.is_key_held(key) for key in KEYBOARD_BINDINGS[action])


def _key_pressed(input_state, action):
    return any(input_state.is_key_pressed(key) for key in KEYBOARD_BINDINGS[action])


def _key_axis(input_state, negative_action, positive_action):
    value = 0.0
    if _key_held(input_state, negative_action):
        value -= 1.0
    if _key_held(input_state, positive_action):
        value += 1.0
    return value


def _command_active(command):
    return any(abs(float(value)) > COMMAND_ACTIVE_DEADBAND for value in command)


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


def _quat_rotate_inv(v, q):
    w, x, y, z = q
    return _quat_rotate(v, (w, -x, -y, -z))


def _quat_rotate(v, q):
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return [
        v[0] + w * tx + y * tz - z * ty,
        v[1] + w * ty + z * tx - x * tz,
        v[2] + w * tz + x * ty - y * tx,
    ]


def _quat_to_roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(_clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


class Script(gobot.NodeScript):
    def _ready(self):
        self.context.fixed_time_step = FIXED_TIME_STEP
        self.context.set_default_joint_gains(
            {
                "position_stiffness": KP,
                "velocity_damping": KD,
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self.robot = self._find_robot()
        self.joints = [self._find_joint(name) for name in JOINT_NAMES]
        for joint in self.joints:
            joint.damping = KD
        self.robot.mode = gobot.RobotMode.Motion
        self.policy = self._load_policy()
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.last_action = [0.0] * len(JOINT_NAMES)
        print(
            "Go1 RL policy playback started. policy={} joints={} cmd=({:.2f}, {:.2f}, {:.2f}) keyboard={}".format(
                "loaded" if self.policy is not None else "missing",
                len(self.joints),
                self.command[0],
                self.command[1],
                self.command[2],
                "click 3D Viewer, WASD/QE, Space stop, R reset" if self.policy is not None else "missing res://policies/go1.onnx",
            )
        )

    def _process(self, delta):
        pass

    def _physics_process(self, delta):
        if not self.playing:
            return
        if not self._ensure_world_controls():
            return

        if self._update_keyboard_command(delta):
            return
        if self._reset_if_fallen():
            return
        observation = self._observation()
        if self.ticks % DECIMATION == 0:
            action = [0.0] * len(JOINT_NAMES)
            if self.policy is not None and _command_active(self.command):
                action = self.policy.action(observation)
            if len(action) != len(JOINT_NAMES):
                print(f"Go1 policy produced {len(action)} actions, expected {len(JOINT_NAMES)}")
                action = [0.0] * len(JOINT_NAMES)
            self.last_action = [float(value) for value in action]

        for index, joint_name in enumerate(JOINT_NAMES):
            lower, upper = POSITION_LIMITS[joint_name]
            target = _clamp(DEFAULT_POS[index] + ACTION_SCALE * self.last_action[index], lower, upper)
            self.context.set_joint_position_target(self.robot.name, joint_name, target)

        self.ticks += 1
        if PRINT_EVERY_TICKS > 0 and self.ticks % PRINT_EVERY_TICKS == 0:
            self._print_state(observation)

    def reset(self):
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.last_action = [0.0] * len(JOINT_NAMES)

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def _load_policy(self):
        path = os.environ.get("GOBOT_GO1_POLICY")
        if path is None:
            path = DEFAULT_POLICY_PATH
        if not path:
            print("Go1 RL policy disabled; set GOBOT_GO1_POLICY to a .onnx or .pt policy to enable playback.")
            return None
        path = _resolve_project_path(self.context, path)
        if not path or not os.path.exists(path):
            fallback = _resolve_project_path(self.context, TORCH_POLICY_PATH)
            if os.path.exists(fallback):
                print(
                    "Go1 ONNX policy not found at '{}'. Falling back to '{}' "
                    "requires torch/rsl-rl-lib from gobot[train].".format(path or "<empty>", fallback)
                )
                path = fallback
            else:
                print(f"Go1 RL policy not found at '{path or '<empty>'}'; set GOBOT_GO1_POLICY to a .onnx or .pt policy.")
                return None
        extension = os.path.splitext(path)[1].lower()
        if extension == ".onnx":
            try:
                print(f"Go1 RL loading ONNX policy: {path}")
                return OnnxPolicy(path)
            except ImportError as error:
                print("Go1 ONNX policy load failed: onnxruntime>=1.19 is required for NumPy 2 policy playback.")
                print(f"Go1 ONNX import error: {error}")
                fallback = _resolve_project_path(self.context, TORCH_POLICY_PATH)
                if os.path.exists(fallback) and path != fallback:
                    print(f"Go1 RL falling back to Torch checkpoint: {fallback}")
                    path = fallback
                    extension = ".pt"
                else:
                    return None
            except Exception as error:
                print(f"Go1 ONNX policy load failed: {error}")
                return None
        if extension != ".pt":
            print(f"Go1 RL policy format '{extension or '<none>'}' is unsupported; use .onnx or .pt.")
            return None
        try:
            print(f"Go1 RL loading Torch policy: {path}")
            return TorchPolicy(path)
        except ImportError as error:
            print("Go1 Torch policy load failed: install gobot[train] to use .pt policies.")
            print(f"Go1 Torch import error: {error}")
            return None
        except Exception as error:
            print(f"Go1 RL policy load failed: {error}")
            return None

    def _ensure_world_controls(self):
        if self.world_controls_ready:
            return True
        if not self.context.has_world:
            return False
        reset_link_state = getattr(self.context, "reset_link_state", None)
        if reset_link_state is not None:
            reset_link_state(
                self.robot.name,
                BASE_LINK,
                RESET_BASE_POSITION,
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
        for index, joint_name in enumerate(JOINT_NAMES):
            reset_joint_state = getattr(self.context, "reset_joint_state", None)
            if reset_joint_state is not None:
                reset_joint_state(self.robot.name, joint_name, DEFAULT_POS[index], 0.0)
            self.context.set_joint_position_target(self.robot.name, joint_name, DEFAULT_POS[index])
        self.world_controls_ready = True
        return True

    def _update_keyboard_command(self, delta):
        input_state = getattr(self.context, "input", None)
        if input_state is None:
            return False
        if _key_pressed(input_state, "reset"):
            self.reset()
            return True
        if self.policy is None:
            return False

        if not input_state.has_control_focus or _key_held(input_state, "stop"):
            desired = [0.0, 0.0, 0.0]
        else:
            desired = [
                KEYBOARD_COMMAND_MAX[0] * _key_axis(input_state, "backward", "forward"),
                KEYBOARD_COMMAND_MAX[1] * _key_axis(input_state, "strafe_right", "strafe_left"),
                KEYBOARD_COMMAND_MAX[2] * _key_axis(input_state, "turn_right", "turn_left"),
            ]
        alpha = _clamp(float(delta) * COMMAND_SMOOTHING, 0.0, 1.0)
        for index in range(3):
            self.command[index] += (desired[index] - self.command[index]) * alpha
        return False

    def _reset_if_fallen(self):
        state = self._runtime_robot_state()
        if state is None:
            return False
        base = self._base_link_state(state)
        if base is None:
            return False
        position = base.get("position", [0.0, 0.0, 0.0])
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        roll, pitch = _quat_to_roll_pitch(quat)
        if position[2] >= FALLEN_BASE_Z and abs(roll) <= FALLEN_ROLL_PITCH and abs(pitch) <= FALLEN_ROLL_PITCH:
            return False
        print(
            "Go1 reset after fall: z={:.3f} roll={:.2f} pitch={:.2f}".format(
                position[2],
                roll,
                pitch,
            )
        )
        self.reset()
        return True

    def _print_state(self, observation):
        x = y = z = 0.0
        state = self._runtime_robot_state()
        if state is not None:
            base = self._base_link_state(state)
            position = base.get("position", [0.0, 0.0, 0.0]) if base is not None else [0.0, 0.0, 0.0]
            if len(position) >= 3:
                x, y, z = position[:3]
        print(
            "Go1 RL tick t={:.2f}s base=({:.3f}, {:.3f}, {:.3f}) "
            "cmd=({:.2f}, {:.2f}, {:.2f}) policy={} focus={} action_norm={:.3f}".format(
                self.context.simulation_time,
                x,
                y,
                z,
                observation[9],
                observation[10],
                observation[11],
                "on" if self.policy is not None else "off",
                "on" if getattr(self.context.input, "has_control_focus", False) else "off",
                math.sqrt(sum(value * value for value in self.last_action)),
            )
        )

    def _observation(self):
        state = self._runtime_robot_state()
        if state is None:
            return [0.0] * 48

        base = self._base_link_state(state)
        if base is None:
            return [0.0] * 48
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        lin_vel = _quat_rotate_inv(base.get("linear_velocity", [0.0, 0.0, 0.0]), quat)
        ang_vel = base.get("angular_velocity", [0.0, 0.0, 0.0])
        projected_gravity = _quat_rotate_inv([0.0, 0.0, -1.0], quat)
        cmd = self.command

        joint_states = {joint.get("name"): joint for joint in state.get("joints", [])}
        joint_pos = []
        joint_vel = []
        for index, joint_name in enumerate(JOINT_NAMES):
            joint = joint_states.get(joint_name, {})
            joint_pos.append(float(joint.get("position", DEFAULT_POS[index])) - DEFAULT_POS[index])
            joint_vel.append(float(joint.get("velocity", 0.0)))

        return lin_vel + list(ang_vel[:3]) + projected_gravity + cmd + joint_pos + joint_vel + self.last_action

    def _base_link_state(self, robot_state):
        for link in robot_state.get("links", []):
            if link.get("name") != BASE_LINK:
                continue
            transform = link.get("global_transform", {})
            return {
                "position": transform.get("position", [0.0, 0.0, 0.0]),
                "quaternion": transform.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
                "linear_velocity": link.get("linear_velocity", [0.0, 0.0, 0.0]),
                "angular_velocity": link.get("angular_velocity", [0.0, 0.0, 0.0]),
            }
        return None

    def _runtime_robot_state(self):
        if not self.context.has_world:
            return None
        state = self.context.get_runtime_state()
        for robot in state.get("robots", []):
            if robot.get("name") == self.robot.name:
                return robot
        return None

    def _find_robot(self):
        root = self.get_root()
        if root is None:
            raise RuntimeError("Go1 script has no scene root")
        if root.name == ROBOT:
            return root
        robot = root.find(ROBOT) or _find_node_by_name(root, ROBOT)
        if robot is None:
            raise RuntimeError(f"scene has no robot node '{ROBOT}'")
        return robot

    def _find_joint(self, name):
        joint = _find_node_by_name(self.robot, name)
        if joint is None:
            raise RuntimeError(f"robot '{self.robot.name}' has no joint '{name}'")
        return joint
