import math
import os
from importlib.metadata import PackageNotFoundError, version

import gobot


ROBOT = "go1"
BASE_LINK = "trunk"
POLICY_ENV = "GOBOT_GO1_POLICY"
DEFAULT_POLICY_PATH = "res://policies/go1.onnx"
TORCH_POLICY_PATH = "res://policies/go1.pt"
PHYSICS_HZ = 200.0
POLICY_HZ = 50.0
FIXED_TIME_STEP = 1.0 / PHYSICS_HZ
MAX_SUB_STEPS = 4
PRINT_EVERY_TICKS = int(PHYSICS_HZ * 2.0)
UNILAB_RESET_BASE_POSITION = [0.0, 0.0, 0.32]
RESET_BASE_POSITION = [0.0, 0.0, 0.278]
ROBOT_ROOT_TO_BASE_Z = 0.4449999928474426
COMMAND = [0.0, 0.0, 0.0]
KEYBOARD_COMMAND_MAX = [1.0, 1.0, 0.5]
COMMAND_SMOOTHING = 8.0
UNILAB_HEADING_STIFFNESS = 0.5
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
LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_KIND_ORDER = ("hip", "thigh", "calf")
JOINT_NAMES = [f"{leg}_{kind}_joint" for leg in LEG_ORDER for kind in JOINT_KIND_ORDER]
UNILAB_DEFAULT_POS_BY_LEG = {
    "FR": (0.0, 0.9, -1.8),
    "FL": (0.0, 0.9, -1.8),
    "RR": (0.0, 1.0, -1.8),
    "RL": (0.0, 1.0, -1.8),
}
DEFAULT_POS_BY_LEG = {
    "FR": (0.1, 0.9, -1.8),
    "FL": (-0.1, 0.9, -1.8),
    "RR": (0.1, 0.9, -1.8),
    "RL": (-0.1, 0.9, -1.8),
}
UNILAB_DEFAULT_POS = [
    UNILAB_DEFAULT_POS_BY_LEG[leg][kind_index]
    for leg in LEG_ORDER
    for kind_index, _kind in enumerate(JOINT_KIND_ORDER)
]
DEFAULT_POS = [
    DEFAULT_POS_BY_LEG[leg][kind_index]
    for leg in LEG_ORDER
    for kind_index, _kind in enumerate(JOINT_KIND_ORDER)
]
ACTION_SCALE_BY_KIND = {
    "hip": 0.3727530386870487,
    "thigh": 0.3727530386870487,
    "calf": 0.24850202579136574,
}
ACTION_SCALE = [
    ACTION_SCALE_BY_KIND[kind]
    for _leg in LEG_ORDER
    for kind in JOINT_KIND_ORDER
]
UNILAB_FLAT_ACTION_SCALE = [0.25 for _ in JOINT_NAMES]
UNILAB_ROUGH_ACTION_SCALE = [
    0.125 if kind == "hip" else 0.25
    for _leg in LEG_ORDER
    for kind in JOINT_KIND_ORDER
]
UNILAB_KP = 35.0
UNILAB_KD = 0.5
UNILAB_MUJOCO_SOLVER_SETTINGS = {
    "cone": 1,
    "convex_collision_iterations": 500,
    "impedance_ratio": 100.0,
}
MUJOCO_SOLVER_SETTINGS = {
    "cone": 1,
    "iterations": 10,
    "line_search_iterations": 20,
    "convex_collision_iterations": 500,
    "impedance_ratio": 10.0,
}
HIP_KP = 15.89524265323492
HIP_KD = 1.0119225759919113
KNEE_KP = 35.764295969778566
KNEE_KD = 2.2768257959818003
HIP_ARMATURE = 0.000111842 * 6.0**2
KNEE_ARMATURE = 0.000111842 * 9.0**2
HIP_EFFORT_LIMIT = 23.7
KNEE_EFFORT_LIMIT = 35.55
HIP_VELOCITY_LIMIT = 30.1
KNEE_VELOCITY_LIMIT = 20.06
JOINT_KP_BY_KIND = {"hip": HIP_KP, "thigh": HIP_KP, "calf": KNEE_KP}
JOINT_KD_BY_KIND = {"hip": HIP_KD, "thigh": HIP_KD, "calf": KNEE_KD}
JOINT_ARMATURE_BY_KIND = {"hip": HIP_ARMATURE, "thigh": HIP_ARMATURE, "calf": KNEE_ARMATURE}
JOINT_EFFORT_LIMIT_BY_KIND = {"hip": HIP_EFFORT_LIMIT, "thigh": HIP_EFFORT_LIMIT, "calf": KNEE_EFFORT_LIMIT}
JOINT_VELOCITY_LIMIT_BY_KIND = {"hip": HIP_VELOCITY_LIMIT, "thigh": HIP_VELOCITY_LIMIT, "calf": KNEE_VELOCITY_LIMIT}
JOINT_KP = [JOINT_KP_BY_KIND[kind] for _leg in LEG_ORDER for kind in JOINT_KIND_ORDER]
JOINT_KD = [JOINT_KD_BY_KIND[kind] for _leg in LEG_ORDER for kind in JOINT_KIND_ORDER]
JOINT_ARMATURE = [JOINT_ARMATURE_BY_KIND[kind] for _leg in LEG_ORDER for kind in JOINT_KIND_ORDER]
JOINT_EFFORT_LIMIT = [JOINT_EFFORT_LIMIT_BY_KIND[kind] for _leg in LEG_ORDER for kind in JOINT_KIND_ORDER]
JOINT_VELOCITY_LIMIT = [JOINT_VELOCITY_LIMIT_BY_KIND[kind] for _leg in LEG_ORDER for kind in JOINT_KIND_ORDER]
DECIMATION = max(1, int(round((1.0 / max(POLICY_HZ, 1.0)) / FIXED_TIME_STEP)))
TERRAIN_SCAN_GRID_SIZE = (1.6, 1.0)
TERRAIN_SCAN_GRID_RESOLUTION = 0.1
TERRAIN_SCAN_DIM = (
    int(round(TERRAIN_SCAN_GRID_SIZE[0] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
) * (
    int(round(TERRAIN_SCAN_GRID_SIZE[1] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
)
HEIGHT_SCAN_MAX_DISTANCE = 5.0
TERRAIN_SCAN_SENSOR = "terrain_scan"
VELOCITY_OBS_SCHEMA_VERSION = "gobot_velocity_v1"
GO1_ROUGH_OBS_SCHEMA_VERSION = VELOCITY_OBS_SCHEMA_VERSION
UNILAB_FLAT_OBS_SCHEMA_VERSION = "gobot_go1_unilab_flat_actor_v1"
UNILAB_ROUGH_OBS_SCHEMA_VERSION = "gobot_go1_unilab_rough_actor_v1"
POSITION_LIMITS_BY_KIND = {
    "hip": (-0.863, 0.863),
    "thigh": (-0.686, 4.501),
    "calf": (-2.818, -0.888),
}
POSITION_LIMITS = {
    f"{leg}_{kind}_joint": POSITION_LIMITS_BY_KIND[kind]
    for leg in LEG_ORDER
    for kind in JOINT_KIND_ORDER
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


def unilab_flat_actor_observation_schema(action_dim, foot_count):
    return ObservationSchema(
        UNILAB_FLAT_OBS_SCHEMA_VERSION,
        (
            ("gyro", 3),
            ("projected_gravity", 3),
            ("joint_pos_rel", action_dim),
            ("joint_vel", action_dim),
            ("current_action", action_dim),
            ("command", 3),
            ("feet_phase", foot_count),
        ),
    )


def unilab_rough_actor_observation_schema(action_dim):
    return ObservationSchema(
        UNILAB_ROUGH_OBS_SCHEMA_VERSION,
        (
            ("gyro_scaled", 3),
            ("projected_gravity", 3),
            ("command", 3),
            ("joint_pos_rel", action_dim),
            ("joint_vel_scaled", action_dim),
            ("current_action", action_dim),
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
UNILAB_FLAT_ACTOR_OBS_SCHEMA = unilab_flat_actor_observation_schema(len(JOINT_NAMES), len(LEG_ORDER))
UNILAB_ROUGH_ACTOR_OBS_SCHEMA = unilab_rough_actor_observation_schema(len(JOINT_NAMES))


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


def _policy_extension(path):
    return os.path.splitext(path)[1].lower()


def _policy_disabled(value):
    return value is not None and value.strip() == ""


class OnnxPolicy:
    def __init__(self, path):
        _check_onnxruntime_version()

        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.profile = "legacy"
        self.schema = None
        input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = None
        if input_shape and isinstance(input_shape[-1], int):
            self.obs_dim = int(input_shape[-1])
            self.schema = _schema_for_obs_dim(self.obs_dim)
            if self.schema is None:
                raise RuntimeError(
                    "Go1 ONNX policy observation dimension mismatch: "
                    f"policy={self.obs_dim}, supported={_supported_obs_dims()}. "
                    "Retrain or export with a supported Go1 profile."
                )
            self.profile = _profile_for_schema(self.schema)

    def action(self, observation):
        obs = self.np.asarray(observation, dtype=self.np.float32).reshape(1, -1)
        if self.obs_dim is not None and obs.shape[1] != self.obs_dim:
            raise RuntimeError(
                f"Go1 ONNX policy expected {self.obs_dim} observations, got {obs.shape[1]}"
            )
        output = self.session.run([self.output_name], {self.input_name: obs})[0].reshape(-1)
        return output.astype(float).tolist()


class TorchPolicy:
    def __init__(self, path):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        actor_state = checkpoint.get("actor_state_dict", {})
        self.obs_dim = _checkpoint_obs_dim(checkpoint)
        self.action_dim = _checkpoint_action_dim(actor_state)
        self.schema = _validate_checkpoint_schema(checkpoint, self.obs_dim)
        self.profile = _profile_for_schema(self.schema)
        self.policy = _CheckpointMlpPolicy(torch, actor_state, self.obs_dim, self.action_dim).to(self.device)
        self.policy.eval()

    def action(self, observation):
        with self.torch.no_grad():
            obs = self.torch.as_tensor(observation, dtype=self.torch.float32, device=self.device).reshape(1, -1)
            output = self.policy(obs)
        return output.reshape(-1).cpu().tolist()


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


def _resolve_policy_path(context, path):
    if not path:
        return ""
    return _resolve_project_path(context, path)


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
        for key in ("gobot_go1_rough", "gobot_go1_unilab_flat", "gobot_go1_unilab_rough"):
            metadata = infos.get(key)
            if isinstance(metadata, dict):
                return metadata
    metadata = checkpoint.get("gobot_go1_velocity")
    if isinstance(metadata, dict):
        return metadata
    return None


def _supported_schemas():
    return (ACTOR_OBS_SCHEMA, UNILAB_FLAT_ACTOR_OBS_SCHEMA, UNILAB_ROUGH_ACTOR_OBS_SCHEMA)


def _supported_obs_dims():
    return tuple(schema.dim for schema in _supported_schemas())


def _schema_for_obs_dim(obs_dim):
    matches = [schema for schema in _supported_schemas() if schema.dim == int(obs_dim)]
    return matches[0] if len(matches) == 1 else None


def _schema_for_metadata(metadata, obs_dim):
    version = metadata.get("obs_schema_version")
    names = tuple(metadata.get("obs_names", ()))
    for schema in _supported_schemas():
        if version == schema.version and names == schema.names:
            return schema
    return _schema_for_obs_dim(obs_dim)


def _profile_for_schema(schema):
    if schema.version == UNILAB_FLAT_OBS_SCHEMA_VERSION:
        return "unilab_flat"
    if schema.version == UNILAB_ROUGH_OBS_SCHEMA_VERSION:
        return "unilab_rough"
    if schema.version == GO1_ROUGH_OBS_SCHEMA_VERSION:
        return "go1_rough"
    return "legacy"


def _validate_checkpoint_schema(checkpoint, obs_dim):
    metadata = _checkpoint_velocity_metadata(checkpoint)
    if metadata is None:
        schema = _schema_for_obs_dim(obs_dim)
        if schema is None:
            raise RuntimeError(
                "Go1 checkpoint observation dimension mismatch: "
                f"checkpoint={obs_dim}, supported={_supported_obs_dims()}. "
                "Retrain or export with a supported Go1 profile."
            )
        return schema
    schema = _schema_for_metadata(metadata, obs_dim)
    version = metadata.get("obs_schema_version")
    names = tuple(metadata.get("obs_names", ()))
    if schema is None or version != schema.version or names != schema.names:
        raise RuntimeError(
            "Go1 checkpoint observation schema differs from playback: "
            f"checkpoint version={version!r} dim={metadata.get('num_obs')} supported={[(s.version, s.dim) for s in _supported_schemas()]}"
        )
    return schema


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


def _quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_pi(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _unilab_projected_gravity_from_upvector(q):
    return [-value for value in _quat_rotate([0.0, 0.0, 1.0], q)]


class Script(gobot.NodeScript):
    def _ready(self):
        self.context.fixed_time_step = FIXED_TIME_STEP
        self.context.max_sub_steps = MAX_SUB_STEPS
        self.robot = self._find_robot()
        self.base_link = self._find_link(BASE_LINK)
        self.joints = [self._find_joint(name) for name in JOINT_NAMES]
        self.terrain_scan = self._find_sensor(TERRAIN_SCAN_SENSOR)
        self.policy = self._load_policy()
        self.policy_schema = self._policy_observation_schema()
        self.policy_profile = _profile_for_schema(self.policy_schema)
        self.policy_obs_dim = self.policy_schema.dim
        self._apply_profile_terrain_overrides()
        if hasattr(self.context, "set_mujoco_solver_settings"):
            self.context.set_mujoco_solver_settings(self._profile_mujoco_solver_settings())
        self.default_pos = self._profile_default_pos()
        self.reset_base_position = self._profile_reset_base_position()
        self.reset_base_position = self._resolve_reset_base_position()
        self.height_scan_dim = TERRAIN_SCAN_DIM if self.policy_profile in {"legacy", "go1_rough"} else 0
        self.action_scale = self._profile_action_scale()
        self.action_clip = self._profile_action_clip()
        default_kp = UNILAB_KP if self.policy_profile in {"unilab_flat", "unilab_rough"} else HIP_KP
        default_kd = UNILAB_KD if self.policy_profile in {"unilab_flat", "unilab_rough"} else HIP_KD
        self.context.set_default_joint_gains(
            {
                "position_stiffness": default_kp,
                "velocity_damping": default_kd,
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self._configure_robot_for_playback()
        self._reset_playback_state()
        self._print_startup()

    def _configure_robot_for_playback(self):
        self._set_robot_editor_transform(self.reset_base_position)
        use_unilab_gains = getattr(self, "policy_profile", "legacy") in {"unilab_flat", "unilab_rough"}
        use_go1_rough_dynamics = getattr(self, "policy_profile", "legacy") == "go1_rough"
        for index, joint in enumerate(self.joints):
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = UNILAB_KP if use_unilab_gains else JOINT_KP[index]
            joint.drive_damping = UNILAB_KD if use_unilab_gains else JOINT_KD[index]
            if use_go1_rough_dynamics:
                effort_limit = JOINT_EFFORT_LIMIT[index]
                joint.armature = JOINT_ARMATURE[index]
                joint.effort_limit = effort_limit
                joint.velocity_limit = JOINT_VELOCITY_LIMIT[index]
                joint.force_lower_limit = -effort_limit
                joint.force_upper_limit = effort_limit
                joint.control_lower_limit = 0.0
                joint.control_upper_limit = 0.0
                joint.damping = 0.0
                joint.friction_loss = 0.0
            else:
                joint.damping = UNILAB_KD if use_unilab_gains else JOINT_KD[index]
        self.robot.mode = gobot.RobotMode.Motion

    def _policy_observation_dim(self):
        obs_dim = getattr(self.policy, "obs_dim", ACTOR_OBS_SCHEMA.dim)
        return ACTOR_OBS_SCHEMA.dim if obs_dim is None else obs_dim

    def _policy_observation_schema(self):
        schema = getattr(self.policy, "schema", None)
        if schema is not None:
            return schema
        obs_dim = self._policy_observation_dim()
        schema = _schema_for_obs_dim(obs_dim)
        if schema is None:
            raise RuntimeError(f"Go1 policy observation dimension {obs_dim} is unsupported")
        return schema

    def _profile_action_scale(self):
        if self.policy_profile == "unilab_flat":
            return list(UNILAB_FLAT_ACTION_SCALE)
        if self.policy_profile == "unilab_rough":
            return list(UNILAB_ROUGH_ACTION_SCALE)
        return list(ACTION_SCALE)

    def _profile_action_clip(self):
        if self.policy_profile == "unilab_rough":
            return 100.0
        if self.policy_profile == "go1_rough":
            return None
        return 1.0

    def _profile_default_pos(self):
        if self.policy_profile in {"unilab_flat", "unilab_rough"}:
            return list(UNILAB_DEFAULT_POS)
        return list(DEFAULT_POS)

    def _profile_reset_base_position(self):
        if self.policy_profile in {"unilab_flat", "unilab_rough"}:
            return list(UNILAB_RESET_BASE_POSITION)
        return list(RESET_BASE_POSITION)

    def _profile_mujoco_solver_settings(self):
        if self.policy_profile == "go1_rough":
            return dict(MUJOCO_SOLVER_SETTINGS)
        return dict(UNILAB_MUJOCO_SOLVER_SETTINGS)

    def _apply_profile_terrain_overrides(self):
        if self.policy_profile != "go1_rough":
            return
        root = self.get_root()
        terrain_world = root.find("terrain_world") if root is not None else None
        if terrain_world is None:
            terrain_world = _find_node_by_name(root, "terrain_world")
        if terrain_world is None:
            return
        old_terrain = terrain_world.find("terrain") or _find_node_by_name(terrain_world, "terrain")
        if old_terrain is not None:
            terrain_world.remove_child(old_terrain, delete=True)
        terrain_cfg = gobot.terrain.go1_rough_terrain_cfg(seed=42, curriculum=False)
        terrain_cfg.num_rows = 5
        terrain_cfg.num_cols = 5
        terrain_cfg.border_width = 10.0
        terrain = gobot.terrain.create_terrain_node(terrain_cfg, "terrain")
        terrain_world.add_child(terrain)

    def _resolve_reset_base_position(self):
        fallback = list(getattr(self, "reset_base_position", RESET_BASE_POSITION))
        if getattr(self, "policy_profile", "legacy") not in {"unilab_rough", "go1_rough"}:
            return fallback
        origins = self._terrain_spawn_origins()
        if not origins:
            return fallback
        spawn = min(origins, key=lambda value: value[0] * value[0] + value[1] * value[1])
        return [spawn[0], spawn[1], spawn[2] + fallback[2]]

    def _terrain_spawn_origins(self):
        root = self.get_root()
        terrain_world = None
        if root is not None:
            terrain_world = root.find("terrain_world") or _find_node_by_name(root, "terrain_world")
        terrain = None
        if terrain_world is not None:
            terrain = terrain_world.find("terrain") or _find_node_by_name(terrain_world, "terrain")
        if terrain is None and root is not None:
            terrain = root.find("terrain") or _find_node_by_name(root, "terrain")
        if terrain is None:
            return []
        origins = getattr(terrain, "spawn_origins", None)
        if origins is None:
            return []
        parsed = []
        for origin in origins:
            try:
                values = [float(origin[index]) for index in range(3)]
            except (TypeError, IndexError, ValueError):
                try:
                    values = [float(value) for value in origin]
                except (TypeError, ValueError):
                    continue
                if len(values) < 3:
                    continue
                values = values[:3]
            parsed.append(values)
        return parsed

    def _reset_playback_state(self):
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.command_target = list(COMMAND)
        self.heading_target = 0.0
        self.heading_target_initialized = False
        self.last_action = [0.0] * len(JOINT_NAMES)
        self.last_targets = list(self.default_pos)
        self.phase = 0.0
        self.feet_phase = [0.0] * len(LEG_ORDER)

    def _print_startup(self):
        print(
            "Go1 RL policy playback started. policy={} joints={} physics_hz={:.1f} policy_hz={:.1f} decimation={} "
            "reset_z={:.3f} cmd=({:.2f}, {:.2f}, {:.2f}) keyboard={}".format(
                "loaded" if self.policy is not None else "missing",
                len(self.joints),
                PHYSICS_HZ,
                POLICY_HZ,
                DECIMATION,
                self.reset_base_position[2],
                self.command[0],
                self.command[1],
                self.command[2],
                "click 3D Viewer, WASD/QE, Space stop, R reset" if self.policy is not None else "missing policy",
            )
        )
        print(f"Go1 RL playback profile={self.policy_profile} obs_dim={self.policy_obs_dim}")
        print(
            "Go1 RL playback runtime fixed_dt={:.4f} max_sub_steps={} solver={} reset_base_z={:.3f}".format(
                FIXED_TIME_STEP,
                MAX_SUB_STEPS,
                self._profile_mujoco_solver_settings(),
                self.reset_base_position[2],
            )
        )

    def _physics_process(self, delta):
        if not self.playing:
            return
        if not self._ensure_world_controls():
            return

        if self._update_keyboard_command(delta):
            return

        policy_tick = self.ticks % DECIMATION == 0
        robot_state = self._runtime_robot_state() if policy_tick else None
        if robot_state is not None and self._reset_if_fallen(robot_state):
            return

        if policy_tick:
            self._update_feet_phase()
            action = [0.0] * len(JOINT_NAMES)
            if self.policy is not None:
                observation = self._observation(robot_state)
                action = self.policy.action(observation)
            if len(action) != len(JOINT_NAMES):
                print(f"Go1 policy produced {len(action)} actions, expected {len(JOINT_NAMES)}")
                action = [0.0] * len(JOINT_NAMES)
            self.last_action = [float(value) for value in action]
            self.last_targets = self._action_targets(self.last_action)
            self._set_joint_position_targets(self.last_targets)

        self.ticks += 1
        if PRINT_EVERY_TICKS > 0 and self.ticks % PRINT_EVERY_TICKS == 0:
            self._print_state(robot_state)

    def reset(self):
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.command_target = list(COMMAND)
        self.heading_target = 0.0
        self.heading_target_initialized = False
        self.last_action = [0.0] * len(JOINT_NAMES)
        self.last_targets = list(self.default_pos)
        self.phase = 0.0
        self.feet_phase = [0.0] * len(LEG_ORDER)
        self._set_robot_editor_transform(self.reset_base_position)

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def _load_policy(self):
        requested_policy = os.environ.get(POLICY_ENV)
        if _policy_disabled(requested_policy):
            print(f"Go1 RL policy disabled; set {POLICY_ENV} to a .onnx or .pt policy to enable playback.")
            return None

        policy_ref = requested_policy if requested_policy is not None else DEFAULT_POLICY_PATH
        explicit_policy = requested_policy is not None
        path = _resolve_policy_path(self.context, policy_ref)
        if not path or not os.path.exists(path):
            print(f"Go1 RL policy not found at '{path or '<empty>'}'.")
            if explicit_policy:
                print(f"Set {POLICY_ENV} to an existing .onnx or .pt policy.")
                return None
            return self._load_torch_fallback("default ONNX policy is missing")

        extension = _policy_extension(path)
        if extension == ".onnx":
            return self._load_onnx_policy(path, allow_torch_fallback=not explicit_policy)
        if extension == ".pt":
            return self._load_torch_policy(path)

        print(f"Go1 RL policy format '{extension or '<none>'}' is unsupported; use .onnx or .pt.")
        return None

    def _load_onnx_policy(self, path, allow_torch_fallback):
        try:
            print(f"Go1 RL loading ONNX policy: {path}")
            return OnnxPolicy(path)
        except ImportError as error:
            print("Go1 ONNX policy load failed: onnxruntime>=1.19 is required for NumPy 2 policy playback.")
            print(f"Go1 ONNX import error: {error}")
            if allow_torch_fallback:
                return self._load_torch_fallback("ONNX runtime is unavailable")
            return None
        except Exception as error:
            print(f"Go1 ONNX policy load failed: {error}")
            if allow_torch_fallback:
                return self._load_torch_fallback("default ONNX policy failed")
            return None

    def _load_torch_fallback(self, reason):
        fallback = _resolve_policy_path(self.context, TORCH_POLICY_PATH)
        if not fallback or not os.path.exists(fallback):
            print(f"Go1 Torch fallback skipped: {reason}; missing {fallback or '<empty>'}.")
            return None
        print(f"Go1 RL falling back to Torch checkpoint ({reason}): {fallback}")
        return self._load_torch_policy(fallback)

    def _load_torch_policy(self, path):
        try:
            print(f"Go1 RL loading Torch policy: {path}")
            return TorchPolicy(path)
        except ImportError as error:
            print("Go1 Torch policy load failed: install/use gobot[train] to use .pt policies.")
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
        self.base_link.reset_runtime_state(
            self.reset_base_position,
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        for index, joint in enumerate(self.joints):
            joint.reset_runtime_state(self.default_pos[index], 0.0)
        self.last_targets = list(self.default_pos)
        self._set_joint_position_targets(self.last_targets)
        self._sync_heading_target_from_runtime()
        self.world_controls_ready = True
        return True

    def _set_joint_position_targets(self, targets):
        for joint, target in zip(self.joints, targets):
            joint.set_position_target(float(target))

    def _action_targets(self, action):
        targets = []
        for index, joint_name in enumerate(JOINT_NAMES):
            lower, upper = POSITION_LIMITS[joint_name]
            raw_action = float(action[index])
            clipped_action = raw_action if self.action_clip is None else _clamp(raw_action, -self.action_clip, self.action_clip)
            targets.append(_clamp(self.default_pos[index] + self.action_scale[index] * clipped_action, lower, upper))
        return targets

    def _set_robot_editor_transform(self, position):
        self.robot.set_transform(
            [float(position[0]), float(position[1]), float(position[2]) - ROBOT_ROOT_TO_BASE_Z],
            [1.0, 0.0, 0.0, 0.0],
        )

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
            yaw_axis = _key_axis(input_state, "turn_right", "turn_left")
            if getattr(self, "policy_profile", "legacy") == "unilab_rough":
                self._ensure_heading_target()
                self.heading_target = _wrap_pi(
                    self.heading_target + yaw_axis * KEYBOARD_COMMAND_MAX[2] * float(delta)
                )
                yaw_command = 0.0
            else:
                yaw_command = KEYBOARD_COMMAND_MAX[2] * yaw_axis
            desired = [
                KEYBOARD_COMMAND_MAX[0] * _key_axis(input_state, "backward", "forward"),
                KEYBOARD_COMMAND_MAX[1] * _key_axis(input_state, "strafe_right", "strafe_left"),
                yaw_command,
            ]
        alpha = _clamp(float(delta) * COMMAND_SMOOTHING, 0.0, 1.0)
        for index in range(3):
            self.command_target[index] += (desired[index] - self.command_target[index]) * alpha
        self.command = self._policy_command()
        return False

    def _sync_heading_target_from_runtime(self):
        state = self._runtime_robot_state()
        base = self._base_link_state(state) if state is not None else None
        if base is None:
            self.heading_target = 0.0
            self.heading_target_initialized = False
            return
        self.heading_target = _quat_to_yaw(base.get("quaternion", [1.0, 0.0, 0.0, 0.0]))
        self.heading_target_initialized = True

    def _ensure_heading_target(self, state=None):
        if getattr(self, "heading_target_initialized", False):
            return
        if state is None:
            state = self._runtime_robot_state()
        base = self._base_link_state(state) if state is not None else None
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0]) if base is not None else [1.0, 0.0, 0.0, 0.0]
        self.heading_target = _quat_to_yaw(quat)
        self.heading_target_initialized = True

    def _policy_command(self, state=None):
        command = list(getattr(self, "command_target", self.command))
        if getattr(self, "policy_profile", "legacy") != "unilab_rough":
            return command
        self._ensure_heading_target(state)
        base = self._base_link_state(state) if state is not None else None
        quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0]) if base is not None else [1.0, 0.0, 0.0, 0.0]
        heading_error = _wrap_pi(self.heading_target - _quat_to_yaw(quat))
        command[2] = _clamp(UNILAB_HEADING_STIFFNESS * heading_error, -1.0, 1.0)
        return command

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

    def _update_feet_phase(self):
        self.phase = (self.phase + (1.0 / POLICY_HZ) * 2.0) % 1.0
        if len(self.feet_phase) >= 4:
            self.feet_phase[0] = self.phase
            self.feet_phase[3] = self.phase
            self.feet_phase[1] = (self.phase + 0.5) % 1.0
            self.feet_phase[2] = (self.phase + 0.5) % 1.0

    def _print_state(self, state=None):
        x = y = z = 0.0
        lin_x = lin_y = lin_z = 0.0
        ang_x = ang_y = ang_z = 0.0
        if state is None:
            state = self._runtime_robot_state()
        if state is not None:
            base = self._base_link_state(state)
            position = base.get("position", [0.0, 0.0, 0.0]) if base is not None else [0.0, 0.0, 0.0]
            if len(position) >= 3:
                x, y, z = position[:3]
            if base is not None:
                quat = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
                lin_vel = _quat_rotate_inv(base.get("linear_velocity", [0.0, 0.0, 0.0]), quat)
                ang_vel = _quat_rotate_inv(base.get("angular_velocity", [0.0, 0.0, 0.0]), quat)
                if len(lin_vel) >= 3:
                    lin_x, lin_y, lin_z = lin_vel[:3]
                if len(ang_vel) >= 3:
                    ang_x, ang_y, ang_z = ang_vel[:3]
        key_axis = self._debug_key_axis()
        print(
            "Go1 RL tick t={:.2f}s base=({:.3f}, {:.3f}, {:.3f}) "
            "lin_b=({:.3f}, {:.3f}, {:.3f}) ang_b=({:.3f}, {:.3f}, {:.3f}) "
            "cmd=({:.2f}, {:.2f}, {:.2f}) keys=({:.0f}, {:.0f}, {:.0f}) "
            "policy={} focus={} action_norm={:.3f}".format(
                self.context.simulation_time,
                x,
                y,
                z,
                lin_x,
                lin_y,
                lin_z,
                ang_x,
                ang_y,
                ang_z,
                self.command[0],
                self.command[1],
                self.command[2],
                key_axis[0],
                key_axis[1],
                key_axis[2],
                "on" if self.policy is not None else "off",
                "on" if getattr(self.context.input, "has_control_focus", False) else "off",
                math.sqrt(sum(value * value for value in self.last_action)),
            )
        )

    def _debug_key_axis(self):
        input_state = getattr(self.context, "input", None)
        if input_state is None or not input_state.has_control_focus:
            return [0.0, 0.0, 0.0]
        return [
            _key_axis(input_state, "backward", "forward"),
            _key_axis(input_state, "strafe_right", "strafe_left"),
            _key_axis(input_state, "turn_right", "turn_left"),
        ]

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
        unilab_projected_gravity = _unilab_projected_gravity_from_upvector(quat)
        cmd = self._policy_command(state)

        joint_states = {joint.get("name"): joint for joint in state.get("joints", [])}
        joint_pos = []
        joint_vel = []
        for index, joint_name in enumerate(JOINT_NAMES):
            joint = joint_states.get(joint_name, {})
            joint_pos.append(float(joint.get("position", self.default_pos[index])) - self.default_pos[index])
            joint_vel.append(float(joint.get("velocity", 0.0)))

        if self.policy_profile == "unilab_flat":
            obs = []
            for part in (
                ang_vel[:3],
                unilab_projected_gravity,
                joint_pos,
                joint_vel,
                self.last_action,
                cmd,
                self.feet_phase,
            ):
                obs.extend(float(value) for value in part)
        elif self.policy_profile == "unilab_rough":
            obs = []
            joint_vel_scaled = [float(value) * 0.05 for value in joint_vel]
            for part in (
                [float(value) * 0.25 for value in ang_vel[:3]],
                unilab_projected_gravity,
                cmd,
                joint_pos,
                joint_vel_scaled,
                self.last_action,
            ):
                obs.extend(float(value) for value in part)
        else:
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
        links = [self.base_link.get_runtime_state()]
        joints = [joint.get_runtime_state() for joint in self.joints]
        sensors = [self.terrain_scan.get_runtime_state()]
        return {
            "name": self.robot.name,
            "links": links,
            "joints": joints,
            "sensors": sensors,
            "contacts": [],
        }

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

    def _find_link(self, name):
        link = _find_node_by_name(self.robot, name)
        if link is None:
            raise RuntimeError(f"robot '{self.robot.name}' has no link '{name}'")
        return link

    def _find_sensor(self, name):
        sensor = _find_node_by_name(self.robot, name)
        if sensor is None:
            raise RuntimeError(f"robot '{self.robot.name}' has no sensor '{name}'")
        return sensor
