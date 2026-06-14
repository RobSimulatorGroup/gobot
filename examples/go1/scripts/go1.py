import math
import os
from importlib.metadata import PackageNotFoundError, version

import gobot


ROBOT = "go1"
BASE_LINK = "trunk"
DEFAULT_POLICY_PATH = "res://policies/go1.pt"
TORCH_POLICY_PATH = "res://policies/go1.pt"
DEBUG_VIZ_MODE = os.environ.get("GOBOT_GO1_DEBUG_VIZ", "0").strip().lower()
DEBUG_VIZ_MJLAB = DEBUG_VIZ_MODE in {"1", "true", "yes", "on", "mjlab"}
DEBUG_VIZ_LABEL = "mjlab" if DEBUG_VIZ_MJLAB else "off"
PHYSICS_HZ = float(os.environ.get("GOBOT_GO1_PHYSICS_HZ", "240.0"))
FIXED_TIME_STEP = 1.0 / max(PHYSICS_HZ, 1.0)
POLICY_HZ = float(os.environ.get("GOBOT_GO1_POLICY_HZ", "50.0"))
SENSOR_PERIOD = 1.0 / max(POLICY_HZ, 1.0)
MAX_SUB_STEPS = max(8, int(math.ceil(PHYSICS_HZ / 60.0)) + 2)
PRINT_EVERY_TICKS = max(1, int(os.environ.get("GOBOT_GO1_PRINT_EVERY_TICKS", str(round(PHYSICS_HZ * 2.0)))))
SENSOR_DEBUG = DEBUG_VIZ_MJLAB or os.environ.get("GOBOT_GO1_SENSOR_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
SENSOR_DEBUG_ALL = os.environ.get("GOBOT_GO1_SENSOR_DEBUG_ALL", "0").lower() in {"1", "true", "yes", "on"}
RESET_BASE_CLEARANCE = float(os.environ.get("GOBOT_GO1_RESET_BASE_Z", "0.278"))
RESET_BASE_POSITION = [0.0, 0.0, RESET_BASE_CLEARANCE]
ROBOT_ROOT_TO_BASE_Z = 0.4449999928474426
COMMAND = [
    float(os.environ.get("GOBOT_GO1_VX", "0.0")),
    float(os.environ.get("GOBOT_GO1_VY", "0.0")),
    float(os.environ.get("GOBOT_GO1_YAW", "0.0")),
]
KEYBOARD_COMMAND_MAX = [
    float(os.environ.get("GOBOT_GO1_MAX_VX", "1.0")),
    float(os.environ.get("GOBOT_GO1_MAX_VY", "1.0")),
    float(os.environ.get("GOBOT_GO1_MAX_YAW", "0.5")),
]
COMMAND_SMOOTHING = 8.0
COMMAND_ACTIVE_DEADBAND = 0.02
DEBUG_ARROW_SCALE = 0.55
DEBUG_ARROW_Z_OFFSET = 0.30
DEBUG_ARROW_SEPARATION = 0.08
MJLAB_ARROW_SCALE = 0.5
MJLAB_ARROW_Z_OFFSET = 0.2
UPRIGHT_ARROW_SCALE = 0.25
UPRIGHT_ARROW_Z_OFFSET = 0.24
UPRIGHT_ARROW_SIDE_OFFSET = 0.08
TWIST_ARROW_SIDE_OFFSET = 0.06
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
    0.1,
    0.9,
    -1.8,
    -0.1,
    0.9,
    -1.8,
    0.1,
    0.9,
    -1.8,
    -0.1,
    0.9,
    -1.8,
]
ACTION_SCALE = [
    0.3727530386870487,
    0.3727530386870487,
    0.24850202579136574,
    0.3727530386870487,
    0.3727530386870487,
    0.24850202579136574,
    0.3727530386870487,
    0.3727530386870487,
    0.24850202579136574,
    0.3727530386870487,
    0.3727530386870487,
    0.24850202579136574,
]
HIP_KP = float(os.environ.get("GOBOT_GO1_HIP_KP", "15.89524265323492"))
HIP_KD = float(os.environ.get("GOBOT_GO1_HIP_KD", "1.0119225759919113"))
KNEE_KP = float(os.environ.get("GOBOT_GO1_KNEE_KP", "35.764295969778566"))
KNEE_KD = float(os.environ.get("GOBOT_GO1_KNEE_KD", "2.2768257959818003"))
JOINT_KP = [
    HIP_KP,
    HIP_KP,
    KNEE_KP,
    HIP_KP,
    HIP_KP,
    KNEE_KP,
    HIP_KP,
    HIP_KP,
    KNEE_KP,
    HIP_KP,
    HIP_KP,
    KNEE_KP,
]
JOINT_KD = [
    HIP_KD,
    HIP_KD,
    KNEE_KD,
    HIP_KD,
    HIP_KD,
    KNEE_KD,
    HIP_KD,
    HIP_KD,
    KNEE_KD,
    HIP_KD,
    HIP_KD,
    KNEE_KD,
]
DECIMATION = max(1, int(round((1.0 / max(POLICY_HZ, 1.0)) / FIXED_TIME_STEP)))
STATE_UPDATE_STRIDE = max(1, int(os.environ.get("GOBOT_GO1_STATE_STRIDE", str(DECIMATION))))
TERRAIN_SCAN_GRID_SIZE = (1.6, 1.0)
TERRAIN_SCAN_GRID_RESOLUTION = 0.1
TERRAIN_SCAN_DIM = (
    int(round(TERRAIN_SCAN_GRID_SIZE[0] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
) * (
    int(round(TERRAIN_SCAN_GRID_SIZE[1] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
)
HEIGHT_SCAN_MAX_DISTANCE = 5.0
TERRAIN_SCAN_SENSOR = "terrain_scan"
FOOT_HEIGHT_SENSOR_NAMES = (
    "FR_foot_height_scan",
    "FL_foot_height_scan",
    "RR_foot_height_scan",
    "RL_foot_height_scan",
)
DEBUG_SENSOR_NAMES = frozenset((TERRAIN_SCAN_SENSOR, *FOOT_HEIGHT_SENSOR_NAMES))
VELOCITY_OBS_SCHEMA_VERSION = "gobot_velocity_v1"
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


class ObservationSchema:
    def __init__(self, version, fields):
        self.version = version
        self.fields = tuple(fields)

    @property
    def dim(self):
        return sum(dim for _, dim in self.fields)

    @property
    def names(self):
        names = []
        for name, dim in self.fields:
            if dim == 1:
                names.append(name)
            else:
                names.extend(f"{name}.{index}" for index in range(dim))
        return tuple(names)


def velocity_actor_observation_schema(action_dim, height_scan_dim):
    return ObservationSchema(
        VELOCITY_OBS_SCHEMA_VERSION,
        (
            ("base_lin_vel_b", 3),
            ("base_ang_vel_b", 3),
            ("projected_gravity", 3),
            ("joint_pos_rel", action_dim),
            ("joint_vel", action_dim),
            ("last_action", action_dim),
            ("command", 3),
            ("height_scan", height_scan_dim),
        ),
    )


def build_velocity_actor_observation(
    *,
    base_lin_vel_b,
    base_ang_vel_b,
    projected_gravity,
    joint_pos_rel,
    joint_vel,
    last_action,
    command,
    height_scan,
):
    obs = []
    for part in (
        base_lin_vel_b,
        base_ang_vel_b,
        projected_gravity,
        joint_pos_rel,
        joint_vel,
        last_action,
        command,
        height_scan,
    ):
        obs.extend(float(value) for value in part)
    return obs


ACTOR_OBS_SCHEMA = velocity_actor_observation_schema(len(JOINT_NAMES), TERRAIN_SCAN_DIM)


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
        input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = None
        if input_shape and isinstance(input_shape[-1], int):
            self.obs_dim = int(input_shape[-1])
            if self.obs_dim != ACTOR_OBS_SCHEMA.dim:
                raise RuntimeError(
                    "Go1 ONNX policy observation dimension mismatch: "
                    f"policy={self.obs_dim}, playback={ACTOR_OBS_SCHEMA.dim}. "
                    "Retrain or export with the current terrain_scan grid schema."
                )

    def action(self, observation):
        obs = self.np.asarray(observation, dtype=self.np.float32).reshape(1, -1)
        if self.obs_dim is not None and obs.shape[1] != self.obs_dim:
            raise RuntimeError(
                f"Go1 ONNX policy expected {self.obs_dim} observations, got {obs.shape[1]}"
            )
        output = self.session.run([self.output_name], {self.input_name: obs})[0].reshape(-1)
        return self.np.clip(output, -1.0, 1.0).astype(float).tolist()


class TorchPolicy:
    def __init__(self, path):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        actor_state = checkpoint.get("actor_state_dict", {})
        self.obs_dim = _checkpoint_obs_dim(checkpoint)
        self.action_dim = _checkpoint_action_dim(actor_state)
        _validate_checkpoint_schema(checkpoint, self.obs_dim)
        self.policy = _CheckpointMlpPolicy(torch, actor_state, self.obs_dim, self.action_dim).to(self.device)
        self.policy.eval()

    def action(self, observation):
        with self.torch.no_grad():
            obs = self.torch.as_tensor(observation, dtype=self.torch.float32, device=self.device).reshape(1, -1)
            output = self.policy(obs)
        return output.reshape(-1).clamp(-1.0, 1.0).cpu().tolist()


class _CheckpointMlpPolicy:
    def __init__(self, torch, actor_state, obs_dim, action_dim):
        self.torch = torch
        self.module = torch.nn.Module()
        self.module.add_module("mlp", _build_mlp(torch, _checkpoint_mlp_dims(actor_state, obs_dim, action_dim)))
        mlp_state = {
            key: value
            for key, value in actor_state.items()
            if key.startswith("mlp.")
        }
        if not mlp_state:
            raise RuntimeError("Torch checkpoint actor_state_dict does not contain mlp.* weights")
        self.module.load_state_dict(mlp_state, strict=True)
        self.mean = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._mean")
        self.std = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._std")
        if self.std is None:
            var = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._var")
            if var is not None:
                self.std = torch.sqrt(torch.clamp(var, min=1.0e-12))
        if self.mean is not None and self.std is None:
            self.std = torch.ones_like(self.mean)

    def to(self, device):
        self.module.to(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    def eval(self):
        self.module.eval()
        return self

    def __call__(self, obs):
        if self.mean is not None:
            obs = (obs - self.mean.reshape(1, -1)) / self.std.reshape(1, -1).clamp_min(1.0e-6)
        return self.module.mlp(obs)


def _build_mlp(torch, dims):
    layers = []
    for index in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


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


def _vector_xy_norm(vector):
    return math.sqrt(float(vector[0]) * float(vector[0]) + float(vector[1]) * float(vector[1]))


def _checkpoint_obs_dim(checkpoint):
    actor_state = checkpoint.get("actor_state_dict", {})
    normalizer_mean = actor_state.get("obs_normalizer._mean")
    if normalizer_mean is not None and getattr(normalizer_mean, "ndim", 0) > 0:
        return int(normalizer_mean.reshape(-1).shape[0])
    first_linear = actor_state.get("mlp.0.weight")
    if first_linear is not None and getattr(first_linear, "ndim", 0) == 2:
        return int(first_linear.shape[1])
    for key, value in actor_state.items():
        if key.endswith(".weight") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[1])
    metadata = _checkpoint_velocity_metadata(checkpoint)
    if metadata is not None:
        return int(metadata.get("num_obs", ACTOR_OBS_SCHEMA.dim))
    return ACTOR_OBS_SCHEMA.dim


def _checkpoint_action_dim(actor_state):
    last_weight = None
    last_layer_index = -1
    for key, value in actor_state.items():
        if not key.startswith("mlp.") or not key.endswith(".weight") or getattr(value, "ndim", 0) != 2:
            continue
        layer_index = _checkpoint_mlp_layer_index(key)
        if layer_index > last_layer_index:
            last_layer_index = layer_index
            last_weight = value
    if last_weight is None:
        raise RuntimeError("Torch checkpoint actor_state_dict does not contain mlp.*.weight tensors")
    action_dim = int(last_weight.shape[0])
    if action_dim != len(JOINT_NAMES):
        raise RuntimeError(f"Go1 policy action dimension mismatch: checkpoint={action_dim}, playback={len(JOINT_NAMES)}")
    return action_dim


def _checkpoint_mlp_layer_index(key):
    parts = key.split(".")
    if len(parts) < 3:
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def _checkpoint_mlp_dims(actor_state, obs_dim, action_dim):
    weights = []
    for key, value in actor_state.items():
        if not key.startswith("mlp.") or not key.endswith(".weight") or getattr(value, "ndim", 0) != 2:
            continue
        weights.append((_checkpoint_mlp_layer_index(key), value))
    weights.sort(key=lambda item: item[0])
    if not weights:
        return [int(obs_dim), int(action_dim)]
    dims = [int(weights[0][1].shape[1])]
    dims.extend(int(weight.shape[0]) for _, weight in weights)
    if dims[0] != int(obs_dim):
        raise RuntimeError(f"Go1 policy observation dimension mismatch: checkpoint={dims[0]}, playback={obs_dim}")
    if dims[-1] != int(action_dim):
        raise RuntimeError(f"Go1 policy action dimension mismatch: checkpoint={dims[-1]}, playback={action_dim}")
    return dims


def _checkpoint_normalizer_tensor(torch, actor_state, name):
    value = actor_state.get(name)
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def _checkpoint_velocity_metadata(checkpoint):
    infos = checkpoint.get("infos")
    if isinstance(infos, dict):
        metadata = infos.get("gobot_go1_velocity")
        if isinstance(metadata, dict):
            return metadata
    metadata = checkpoint.get("gobot_go1_velocity")
    if isinstance(metadata, dict):
        return metadata
    return None


def _validate_checkpoint_schema(checkpoint, obs_dim):
    metadata = _checkpoint_velocity_metadata(checkpoint)
    if metadata is None:
        if obs_dim != ACTOR_OBS_SCHEMA.dim:
            raise RuntimeError(
                "Go1 checkpoint observation dimension mismatch: "
                f"checkpoint={obs_dim}, playback={ACTOR_OBS_SCHEMA.dim}. "
                "Retrain or export with the current terrain_scan grid schema."
            )
        return
    version = metadata.get("obs_schema_version")
    names = tuple(metadata.get("obs_names", ()))
    if version != VELOCITY_OBS_SCHEMA_VERSION or names != ACTOR_OBS_SCHEMA.names:
        raise RuntimeError(
            "Go1 checkpoint observation schema differs from playback: "
            f"checkpoint version={version!r} dim={metadata.get('num_obs')} current={ACTOR_OBS_SCHEMA.version} dim={ACTOR_OBS_SCHEMA.dim}"
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


def _quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Script(gobot.NodeScript):
    def _ready(self):
        self.context.fixed_time_step = FIXED_TIME_STEP
        if hasattr(self.context, "max_sub_steps"):
            self.context.max_sub_steps = MAX_SUB_STEPS
        self.context.set_default_joint_gains(
            {
                "position_stiffness": HIP_KP,
                "velocity_damping": HIP_KD,
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self.robot = self._find_robot()
        self._configure_sensor_debug()
        self.joints = [self._find_joint(name) for name in JOINT_NAMES]
        self.reset_base_position = list(RESET_BASE_POSITION)
        self._set_robot_editor_transform(self.reset_base_position)
        for index, joint in enumerate(self.joints):
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = JOINT_KP[index]
            joint.drive_damping = JOINT_KD[index]
            joint.damping = JOINT_KD[index]
        self.robot.mode = gobot.RobotMode.Motion
        self.policy = self._load_policy()
        self.policy_obs_dim = getattr(self.policy, "obs_dim", ACTOR_OBS_SCHEMA.dim)
        if self.policy_obs_dim is None:
            self.policy_obs_dim = ACTOR_OBS_SCHEMA.dim
        self.height_scan_dim = TERRAIN_SCAN_DIM
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.last_action = [0.0] * len(JOINT_NAMES)
        self.last_targets = list(DEFAULT_POS)
        print(
            "Go1 RL policy playback started. policy={} joints={} physics_hz={:.1f} policy_hz={:.1f} decimation={} "
            "reset_z={:.3f} debug_viz={} sensor_debug={} cmd=({:.2f}, {:.2f}, {:.2f}) keyboard={}".format(
                "loaded" if self.policy is not None else "missing",
                len(self.joints),
                PHYSICS_HZ,
                POLICY_HZ,
                DECIMATION,
                RESET_BASE_CLEARANCE,
                DEBUG_VIZ_LABEL,
                "on" if SENSOR_DEBUG else "off",
                self.command[0],
                self.command[1],
                self.command[2],
                "click 3D Viewer, WASD/QE, Space stop, R reset" if self.policy is not None else "missing policy",
            )
        )

    def _process(self, delta):
        pass

    def _exit_tree(self):
        gobot.clear_debug_arrows()

    def _physics_process(self, delta):
        if not self.playing:
            return
        if not self._ensure_world_controls():
            return

        if self._update_keyboard_command(delta):
            return

        policy_tick = self.ticks % DECIMATION == 0
        state_tick = policy_tick or self.ticks % STATE_UPDATE_STRIDE == 0
        robot_state = self._runtime_robot_state() if state_tick else None
        if robot_state is not None and self._reset_if_fallen(robot_state):
            return

        if policy_tick:
            action = [0.0] * len(JOINT_NAMES)
            if self.policy is not None and _command_active(self.command):
                observation = self._observation(robot_state)
                action = self.policy.action(observation)
            if len(action) != len(JOINT_NAMES):
                print(f"Go1 policy produced {len(action)} actions, expected {len(JOINT_NAMES)}")
                action = [0.0] * len(JOINT_NAMES)
            self.last_action = [float(value) for value in action]
            self.last_targets = self._action_targets(self.last_action)
            self._set_joint_position_targets(self.last_targets)
            if robot_state is not None:
                self._update_debug_arrows(robot_state)

        self.ticks += 1
        if PRINT_EVERY_TICKS > 0 and self.ticks % PRINT_EVERY_TICKS == 0:
            self._print_state(robot_state)

    def reset(self):
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.last_action = [0.0] * len(JOINT_NAMES)
        self.last_targets = list(DEFAULT_POS)
        gobot.clear_debug_arrows()
        self._set_robot_editor_transform(self.reset_base_position)

    def pause(self):
        self.playing = False
        gobot.clear_debug_arrows()

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
                    "requires torch from gobot[train].".format(path or "<empty>", fallback)
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
                self.reset_base_position,
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
        for index, joint_name in enumerate(JOINT_NAMES):
            reset_joint_state = getattr(self.context, "reset_joint_state", None)
            if reset_joint_state is not None:
                reset_joint_state(self.robot.name, joint_name, DEFAULT_POS[index], 0.0)
        self.last_targets = list(DEFAULT_POS)
        self._set_joint_position_targets(self.last_targets)
        self.world_controls_ready = True
        return True

    def _set_joint_position_targets(self, targets):
        set_targets = getattr(self.context, "set_joint_position_targets", None)
        if set_targets is not None:
            set_targets(self.robot.name, JOINT_NAMES, targets)
            return
        set_target = getattr(self.context, "set_joint_position_target", None)
        if set_target is None:
            raise RuntimeError("Gobot AppContext has no joint position target API")
        for joint_name, target in zip(JOINT_NAMES, targets):
            set_target(self.robot.name, joint_name, target)

    def _action_targets(self, action):
        targets = []
        for index, joint_name in enumerate(JOINT_NAMES):
            lower, upper = POSITION_LIMITS[joint_name]
            targets.append(_clamp(DEFAULT_POS[index] + ACTION_SCALE[index] * float(action[index]), lower, upper))
        return targets

    def _set_robot_editor_transform(self, position):
        set_transform = getattr(self.robot, "set_transform", None)
        if set_transform is None:
            return
        set_transform(
            [float(position[0]), float(position[1]), float(position[2]) - ROBOT_ROOT_TO_BASE_Z],
            [1.0, 0.0, 0.0, 0.0],
        )

    def _configure_sensor_debug(self):
        for node in self._iter_nodes(self.robot):
            if hasattr(node, "sensor_period") and getattr(node, "name", "") in DEBUG_SENSOR_NAMES:
                node.sensor_period = SENSOR_PERIOD
            if hasattr(node, "visualize_debug"):
                node.visualize_debug = self._should_visualize_sensor(node)

    def _should_visualize_sensor(self, node):
        if not SENSOR_DEBUG:
            return False
        if SENSOR_DEBUG_ALL:
            return True
        return getattr(node, "name", "") in DEBUG_SENSOR_NAMES

    def _iter_nodes(self, node):
        if node is None:
            return
        yield node
        for child in getattr(node, "children", []):
            yield from self._iter_nodes(child)

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

    def _reset_if_fallen(self, state=None):
        if state is None:
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

    def _print_state(self, state=None):
        x = y = z = 0.0
        if state is None:
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
                self.command[0],
                self.command[1],
                self.command[2],
                "on" if self.policy is not None else "off",
                "on" if getattr(self.context.input, "has_control_focus", False) else "off",
                math.sqrt(sum(value * value for value in self.last_action)),
            )
        )

    def _update_debug_arrows(self, state=None):
        if DEBUG_VIZ_MJLAB:
            self._update_mjlab_debug_arrows(state)
            return
        if state is None:
            state = self._runtime_robot_state()
        if state is None:
            gobot.clear_debug_arrows()
            return
        base = self._base_link_state(state)
        if base is None:
            gobot.clear_debug_arrows()
            return

        position = base.get("position", [0.0, 0.0, 0.0])
        if len(position) < 3:
            gobot.clear_debug_arrows()
            return
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        yaw = _quat_yaw(quat)
        command_world = [
            math.cos(yaw) * self.command[0] - math.sin(yaw) * self.command[1],
            math.sin(yaw) * self.command[0] + math.cos(yaw) * self.command[1],
            0.0,
        ]
        actual = base.get("linear_velocity", [0.0, 0.0, 0.0])
        actual_world = [
            float(actual[0]) if len(actual) > 0 else 0.0,
            float(actual[1]) if len(actual) > 1 else 0.0,
            0.0,
        ]
        start = [
            float(position[0]),
            float(position[1]),
            float(position[2]) + DEBUG_ARROW_Z_OFFSET,
        ]
        arrows = []
        if _vector_xy_norm(command_world) > COMMAND_ACTIVE_DEADBAND:
            arrows.append(
                {
                    "start": start,
                    "vector": command_world,
                    "color": (0.15, 0.85, 0.20, 1.0),
                    "scale": DEBUG_ARROW_SCALE,
                    "label": "command_velocity",
                }
            )
        if _vector_xy_norm(actual_world) > COMMAND_ACTIVE_DEADBAND:
            arrows.append(
                {
                    "start": [start[0], start[1], start[2] + DEBUG_ARROW_SEPARATION],
                    "vector": actual_world,
                    "color": (0.10, 0.42, 1.0, 1.0),
                    "scale": DEBUG_ARROW_SCALE,
                    "label": "actual_velocity",
                }
            )
        gobot.set_debug_arrows(arrows)

    def _update_mjlab_debug_arrows(self, state=None):
        if state is None:
            state = self._runtime_robot_state()
        if state is None:
            gobot.clear_debug_arrows()
            return
        base = self._base_link_state(state)
        if base is None:
            gobot.clear_debug_arrows()
            return

        position = base.get("position", [0.0, 0.0, 0.0])
        if len(position) < 3:
            gobot.clear_debug_arrows()
            return
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        command_start = self._offset_from_base(
            position,
            quat,
            [0.0, TWIST_ARROW_SIDE_OFFSET, MJLAB_ARROW_Z_OFFSET * MJLAB_ARROW_SCALE],
        )
        actual_start = self._offset_from_base(
            position,
            quat,
            [0.0, -TWIST_ARROW_SIDE_OFFSET, MJLAB_ARROW_Z_OFFSET * MJLAB_ARROW_SCALE],
        )
        angular_start = self._offset_from_base(
            position,
            quat,
            [TWIST_ARROW_SIDE_OFFSET, 0.0, MJLAB_ARROW_Z_OFFSET * MJLAB_ARROW_SCALE],
        )
        actual_angular_start = self._offset_from_base(
            position,
            quat,
            [-TWIST_ARROW_SIDE_OFFSET, 0.0, MJLAB_ARROW_Z_OFFSET * MJLAB_ARROW_SCALE],
        )
        terrain_origin = self._offset_from_base(
            position,
            quat,
            [0.0, UPRIGHT_ARROW_SIDE_OFFSET, UPRIGHT_ARROW_Z_OFFSET],
        )
        body_up_origin = self._offset_from_base(
            position,
            quat,
            [0.0, -UPRIGHT_ARROW_SIDE_OFFSET, UPRIGHT_ARROW_Z_OFFSET],
        )
        actual_lin_b = _quat_rotate_inv(base.get("linear_velocity", [0.0, 0.0, 0.0]), quat)
        actual_ang_b = _quat_rotate_inv(base.get("angular_velocity", [0.0, 0.0, 0.0]), quat)
        arrows = [
            {
                "start": command_start,
                "vector": _quat_rotate([self.command[0], self.command[1], 0.0], quat),
                "color": (0.2, 0.2, 0.6, 0.6),
                "scale": MJLAB_ARROW_SCALE,
                "label": "twist_command_linear",
            },
            {
                "start": angular_start,
                "vector": _quat_rotate([0.0, 0.0, self.command[2]], quat),
                "color": (0.2, 0.6, 0.2, 0.6),
                "scale": MJLAB_ARROW_SCALE,
                "label": "twist_command_angular",
            },
            {
                "start": actual_start,
                "vector": _quat_rotate([actual_lin_b[0], actual_lin_b[1], 0.0], quat),
                "color": (0.0, 0.6, 1.0, 0.7),
                "scale": MJLAB_ARROW_SCALE,
                "label": "twist_actual_linear",
            },
            {
                "start": actual_angular_start,
                "vector": _quat_rotate([0.0, 0.0, actual_ang_b[2] if len(actual_ang_b) > 2 else 0.0], quat),
                "color": (0.0, 1.0, 0.4, 0.7),
                "scale": MJLAB_ARROW_SCALE,
                "label": "twist_actual_angular",
            },
        ]

        terrain_normal = self._terrain_normal(state)
        if terrain_normal is not None:
            arrows.append(
                {
                    "start": terrain_origin,
                    "vector": terrain_normal,
                    "color": (0.8, 0.2, 0.8, 0.8),
                    "scale": UPRIGHT_ARROW_SCALE,
                    "label": "terrain_normal",
                }
            )
        arrows.append(
            {
                "start": body_up_origin,
                "vector": _quat_rotate([0.0, 0.0, 1.0], quat),
                "color": (1.0, 0.5, 0.0, 0.8),
                "scale": UPRIGHT_ARROW_SCALE,
                "label": "body_up",
            }
        )
        gobot.set_debug_arrows(arrows)

    def _offset_from_base(self, position, quat, local_offset):
        offset = _quat_rotate(local_offset, quat)
        return [
            float(position[0]) + offset[0],
            float(position[1]) + offset[1],
            float(position[2]) + offset[2],
        ]

    def _terrain_normal(self, robot_state):
        sensor = self._sensor_map(robot_state).get(TERRAIN_SCAN_SENSOR)
        if sensor is None:
            return None
        total = [0.0, 0.0, 0.0]
        count = 0
        for hit in sensor.get("hits", []):
            if not hit.get("hit", False):
                continue
            normal = hit.get("normal", [0.0, 0.0, 0.0])
            if len(normal) < 3:
                continue
            total[0] += float(normal[0])
            total[1] += float(normal[1])
            total[2] += float(normal[2])
            count += 1
        if count == 0:
            return None
        length = math.sqrt(total[0] * total[0] + total[1] * total[1] + total[2] * total[2])
        if length <= 1.0e-6:
            return None
        if total[2] < 0.0:
            length = -length
        return [total[0] / length, total[1] / length, total[2] / length]

    def _observation(self, state=None):
        if state is None:
            state = self._runtime_robot_state()
        if state is None:
            return [0.0] * self.policy_obs_dim

        base = self._base_link_state(state)
        if base is None:
            return [0.0] * self.policy_obs_dim
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        lin_vel = _quat_rotate_inv(base.get("linear_velocity", [0.0, 0.0, 0.0]), quat)
        ang_vel = _quat_rotate_inv(base.get("angular_velocity", [0.0, 0.0, 0.0]), quat)
        projected_gravity = _quat_rotate_inv([0.0, 0.0, -1.0], quat)
        cmd = self.command

        joint_states = {joint.get("name"): joint for joint in state.get("joints", [])}
        joint_pos = []
        joint_vel = []
        for index, joint_name in enumerate(JOINT_NAMES):
            joint = joint_states.get(joint_name, {})
            joint_pos.append(float(joint.get("position", DEFAULT_POS[index])) - DEFAULT_POS[index])
            joint_vel.append(float(joint.get("velocity", 0.0)))

        height_scan = self._height_scan(state)
        obs = build_velocity_actor_observation(
            base_lin_vel_b=lin_vel,
            base_ang_vel_b=ang_vel[:3],
            projected_gravity=projected_gravity,
            joint_pos_rel=joint_pos,
            joint_vel=joint_vel,
            last_action=self.last_action,
            command=cmd,
            height_scan=height_scan,
        )
        if len(obs) != self.policy_obs_dim:
            raise RuntimeError(
                f"Go1 playback observation schema mismatch: expected {self.policy_obs_dim}, got {len(obs)}"
            )
        return obs

    def _height_scan(self, robot_state):
        sensor = self._sensor_map(robot_state).get(TERRAIN_SCAN_SENSOR)
        if sensor is None:
            raise RuntimeError(f"Go1 runtime state is missing required sensor {TERRAIN_SCAN_SENSOR!r}")
        values = [float(value) for value in sensor.get("values", [])]
        expected = getattr(self, "height_scan_dim", TERRAIN_SCAN_DIM)
        if len(values) != expected or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Go1 sensor {TERRAIN_SCAN_SENSOR!r} produced invalid height scan values: "
                f"expected {expected}, got {len(values)}"
            )
        return [value / HEIGHT_SCAN_MAX_DISTANCE for value in values]

    def _sensor_map(self, robot_state):
        return {
            sensor.get("name") or sensor.get("sensor_name"): sensor
            for sensor in robot_state.get("sensors", [])
        }

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
