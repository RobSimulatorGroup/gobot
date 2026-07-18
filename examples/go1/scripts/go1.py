import math
import os
from importlib.metadata import PackageNotFoundError, version

import gobot
from gobot.rl.locomotion import (
    build_velocity_actor_observation,
    velocity_actor_observation_schema,
)
from gobot.rl.policy import (
    policy_manifest_from_checkpoint,
    policy_manifest_from_onnx_metadata,
    read_policy_manifest_sidecar,
    scene_bundle_digest,
)
from gobot.rl.spec import ActionSpec, SpecField
from go1_profile import (
    GO1_DEFAULT_BASE_POSITION,
    GO1_JOINT_NAMES,
)
from go1_velocity_contract import GO1_TASK_NAME, GO1_TASK_VERSION


TASK_NAME = GO1_TASK_NAME
TASK_VERSION = GO1_TASK_VERSION
ROBOT = "go1"
BASE_LINK = "trunk"
POLICY_ENV = "GOBOT_GO1_POLICY"
RUN_POLICY_ENV = "GOBOT_GO1_RUN_POLICY"
DEFAULT_POLICY_PATH = "res://policies/go1_velocity.onnx"
DEFAULT_TORCH_POLICY_PATH = "res://policies/go1_velocity.pt"
DEFAULT_RUN_POLICY_PATH = "res://policies/go1_velocity_run.onnx"
DEFAULT_RUN_TORCH_POLICY_PATH = "res://policies/go1_velocity_run.pt"
PRINT_INTERVAL_SECONDS = 2.0
ROBOT_ROOT_TO_BASE_Z = 0.4449999928474426
COMMAND = [0.0, 0.0, 0.0]
KEYBOARD_COMMAND_MAX = [1.0, 1.0, 0.5]
KEYBOARD_RUN_COMMAND_X = [-1.5, 3.0]
COMMAND_SMOOTHING = 8.0
FALLEN_BASE_CLEARANCE = 0.16
FALLEN_ROLL_PITCH = math.radians(70.0)
HEIGHT_SCAN_MAX_DISTANCE = 5.0
TERRAIN_SCAN_SENSOR = "terrain_scan"
IMU_SENSOR = "imu"
TERRAIN_SCAN_GRID_SIZE = (1.6, 1.0)
TERRAIN_SCAN_GRID_RESOLUTION = 0.1
TERRAIN_SCAN_DIM = (
    int(round(TERRAIN_SCAN_GRID_SIZE[0] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
) * (
    int(round(TERRAIN_SCAN_GRID_SIZE[1] / TERRAIN_SCAN_GRID_RESOLUTION)) + 1
)
KEYBOARD_BINDINGS = {
    "forward": ("W", "Up"),
    "backward": ("S", "Down"),
    "turn_left": ("A", "Left"),
    "turn_right": ("D", "Right"),
    "strafe_left": ("Q",),
    "strafe_right": ("E",),
    "run": ("LeftShift", "RightShift"),
    "stop": ("Space",),
    "reset": ("R",),
}
JOINT_NAMES = GO1_JOINT_NAMES
ACTOR_OBS_SPEC = velocity_actor_observation_schema(len(JOINT_NAMES), TERRAIN_SCAN_DIM)
ACTION_SPEC = ActionSpec(
    version=f"{ACTOR_OBS_SPEC.version}_action_v1",
    fields=tuple(SpecField(name, 1) for name in JOINT_NAMES),
    lower=-math.inf,
    upper=math.inf,
)


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
        raise ImportError("onnxruntime is not installed; install gobot or onnxruntime>=1.19") from error
    parsed_version = _parse_version_prefix(runtime_version)
    if parsed_version and parsed_version < (1, 19):
        raise ImportError(
            f"onnxruntime {runtime_version} is installed; NumPy 2 playback requires onnxruntime>=1.19"
        )


def _policy_extension(path):
    return os.path.splitext(path)[1].lower()


def _resolve_project_path(context, path):
    if path.startswith("res://"):
        return os.path.join(context.project_path, path.removeprefix("res://"))
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


def _quat_rotate_inv(vector, quaternion):
    w, x, y, z = quaternion
    return _quat_rotate(vector, (w, -x, -y, -z))


def _quat_rotate(vector, quaternion):
    w, x, y, z = quaternion
    tx = 2.0 * (y * vector[2] - z * vector[1])
    ty = 2.0 * (z * vector[0] - x * vector[2])
    tz = 2.0 * (x * vector[1] - y * vector[0])
    return [
        vector[0] + w * tx + y * tz - z * ty,
        vector[1] + w * ty + z * tx - x * tz,
        vector[2] + w * tz + x * ty - y * tx,
    ]


def _quat_to_roll_pitch(quaternion):
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(_clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _checkpoint_mlp_layer_index(key):
    parts = key.split(".")
    return int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else -1


def _checkpoint_mlp_dims(actor_state, obs_dim, action_dim):
    weights = [
        (_checkpoint_mlp_layer_index(key), value)
        for key, value in actor_state.items()
        if key.startswith("mlp.")
        and key.endswith(".weight")
        and getattr(value, "ndim", 0) == 2
    ]
    weights = [item for item in weights if item[0] >= 0]
    weights.sort(key=lambda item: item[0])
    if not weights:
        raise RuntimeError("checkpoint actor_state_dict does not contain mlp.*.weight tensors")
    dims = [int(weights[0][1].shape[1])]
    dims.extend(int(weight.shape[0]) for _, weight in weights)
    if dims[0] != int(obs_dim) or dims[-1] != int(action_dim):
        raise RuntimeError(
            f"checkpoint MLP dimensions {dims} do not match manifest obs/action "
            f"{obs_dim}/{action_dim}"
        )
    return dims


def _activation_module(torch, name):
    activations = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh,
    }
    activation = activations.get(str(name).strip().lower())
    if activation is None:
        raise RuntimeError(f"unsupported policy activation {name!r}")
    return activation()


def _build_mlp(torch, dims, activation):
    layers = []
    for index in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[index], dims[index + 1]))
        if index + 2 < len(dims):
            layers.append(_activation_module(torch, activation))
    return torch.nn.Sequential(*layers)


def _checkpoint_normalizer_tensor(torch, actor_state, name):
    value = actor_state.get(name)
    return None if value is None else torch.as_tensor(value, dtype=torch.float32).reshape(-1)


class OnnxPolicy:
    def __init__(self, path):
        _check_onnxruntime_version()
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.manifest = policy_manifest_from_onnx_metadata(
            self.session.get_modelmeta().custom_metadata_map
        )
        if self.manifest is None:
            self.manifest = read_policy_manifest_sidecar(path)
        if self.manifest is None:
            raise RuntimeError(
                "ONNX policy has no Gobot policy manifest; retrain and export it with "
                "examples/go1/tools/export_policy_onnx.py"
            )
        self.obs_dim = int(self.manifest.observation_spec["dim"])
        self.action_dim = int(self.manifest.action_spec["dim"])
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if input_shape and isinstance(input_shape[-1], int) and int(input_shape[-1]) != self.obs_dim:
            raise RuntimeError("ONNX input dimension does not match its policy manifest")
        if output_shape and isinstance(output_shape[-1], int) and int(output_shape[-1]) != self.action_dim:
            raise RuntimeError("ONNX output dimension does not match its policy manifest")

    def action(self, observation):
        obs = self.np.asarray(observation, dtype=self.np.float32).reshape(1, -1)
        if obs.shape[1] != self.obs_dim:
            raise RuntimeError(f"policy expected {self.obs_dim} observations, got {obs.shape[1]}")
        output = self.session.run([self.output_name], {self.input_name: obs})[0].reshape(-1)
        return output.astype(float).tolist()


class TorchPolicy:
    def __init__(self, path):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.manifest = policy_manifest_from_checkpoint(checkpoint)
        if self.manifest is None:
            self.manifest = read_policy_manifest_sidecar(path)
        if self.manifest is None:
            raise RuntimeError(
                "Torch checkpoint has no Gobot policy manifest; retrain it with the current training script"
            )
        actor_state = checkpoint.get("actor_state_dict", {})
        self.obs_dim = int(self.manifest.observation_spec["dim"])
        self.action_dim = int(self.manifest.action_spec["dim"])
        activation = self.manifest.model.get("activation", "elu")
        self.policy = _CheckpointMlpPolicy(
            torch,
            actor_state,
            self.obs_dim,
            self.action_dim,
            activation,
        ).to(self.device)
        self.policy.eval()

    def action(self, observation):
        with self.torch.no_grad():
            obs = self.torch.as_tensor(
                observation,
                dtype=self.torch.float32,
                device=self.device,
            ).reshape(1, -1)
            output = self.policy(obs)
        return output.reshape(-1).cpu().tolist()


class _CheckpointMlpPolicy:
    def __init__(self, torch, actor_state, obs_dim, action_dim, activation):
        self.module = torch.nn.Module()
        self.module.add_module(
            "mlp",
            _build_mlp(
                torch,
                _checkpoint_mlp_dims(actor_state, obs_dim, action_dim),
                activation,
            ),
        )
        mlp_state = {key: value for key, value in actor_state.items() if key.startswith("mlp.")}
        self.module.load_state_dict(mlp_state, strict=True)
        self.mean = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._mean")
        self.std = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._std")
        if self.std is None:
            variance = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._var")
            if variance is not None:
                self.std = torch.sqrt(torch.clamp(variance, min=1.0e-12))
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


class Script(gobot.NodeScript):
    def _ready(self):
        self.robot = self._find_robot()
        self.base_link = self._find_link(BASE_LINK)
        self.joints = [self._find_joint(name) for name in JOINT_NAMES]
        self.imu = self._find_sensor(IMU_SENSOR)
        self.terrain_scan = self._find_sensor(TERRAIN_SCAN_SENSOR)
        self._load_and_validate_policy()
        self._configure_runtime_profile()

        self.context.fixed_time_step = self.physics_dt
        self.context.max_sub_steps = self.decimation
        self.print_every_ticks = max(1, int(round(PRINT_INTERVAL_SECONDS / self.physics_dt)))
        self.terrain_origins = self._terrain_spawn_origins()
        self.terrain_rows, self.terrain_type_names = self._terrain_grid_metadata()
        self.reset_base_position = self._resolve_reset_base_position()
        if self.solver_settings and hasattr(self.context, "set_mujoco_solver_settings"):
            self.context.set_mujoco_solver_settings(self.solver_settings)
        self.context.set_default_joint_gains(
            {
                "position_stiffness": float(self.joint_kp[0]),
                "velocity_damping": float(self.joint_kd[0]),
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self._configure_robot_for_playback()
        self._reset_playback_state()
        self._print_startup()

    def _load_and_validate_policy(self):
        self.policy = None
        self.run_policy = None
        self.manifest = None
        self.policy_error = ""
        try:
            policy = self._load_policy()
            self.policy = policy
            self.manifest = policy.manifest
            self._validate_policy_contract(policy, "base")
            run_policy = self._load_run_policy()
            if run_policy is not None:
                self._validate_policy_contract(run_policy, "run")
                self._validate_matching_control_contract(run_policy.manifest)
                self.run_policy = run_policy
        except Exception as error:
            self.policy = None
            self.run_policy = None
            self.manifest = None
            self.policy_error = f"{type(error).__name__}: {error}"
            raise RuntimeError(
                "Go1 Play requires a current manifest-backed policy; "
                f"rejected policy: {self.policy_error}"
            ) from error

    def _configure_runtime_profile(self):
        self.physics_dt = float(self.manifest.physics_dt)
        self.decimation = int(self.manifest.decimation)
        self.policy_dt = self.manifest.policy_dt
        self.default_pos = self._control_vector("default_joint_position")
        self.action_scale = self._control_vector("action_scale")
        self.joint_kp = self._control_vector("kp")
        self.joint_kd = self._control_vector("kd")
        action_clip = self.manifest.control.get("action_clip")
        self.action_clip = None if action_clip is None else float(action_clip)
        self.reset_base_height = float(
            self.manifest.control.get("reset_base_height", GO1_DEFAULT_BASE_POSITION[2])
        )
        self.height_scan_max_distance = float(
            self.manifest.extras.get("height_scan_max_distance", HEIGHT_SCAN_MAX_DISTANCE)
        )
        self.solver_settings = dict(self.manifest.extras.get("solver_settings", {}))

    def _validate_policy_contract(self, policy, label):
        manifest = policy.manifest
        digest = scene_bundle_digest(self.context.project_path, manifest.scene_path)
        manifest.validate_runtime(
            observation_spec=ACTOR_OBS_SPEC,
            action_spec=ACTION_SPEC,
            joint_names=JOINT_NAMES,
            physics_dt=manifest.physics_dt,
            decimation=manifest.decimation,
            task_name=TASK_NAME,
            task_version=TASK_VERSION,
            scene_digest=digest,
        )
        if policy.action_dim != len(JOINT_NAMES):
            raise RuntimeError(
                f"Go1 {label} policy action dimension mismatch: "
                f"{policy.action_dim} != {len(JOINT_NAMES)}"
            )

    def _validate_matching_control_contract(self, manifest):
        fields = (
            "mode",
            "default_joint_position",
            "action_scale",
            "action_clip",
            "kp",
            "kd",
            "reset_base_height",
        )
        for name in fields:
            if manifest.control.get(name) != self.manifest.control.get(name):
                raise RuntimeError(f"Go1 run policy control {name!r} differs from the base policy")

    def _control_vector(self, name):
        value = self.manifest.control.get(name)
        if value is None:
            raise RuntimeError(f"policy manifest control is missing {name!r}")
        if isinstance(value, (int, float)):
            return [float(value)] * len(JOINT_NAMES)
        values = [float(item) for item in value]
        if len(values) != len(JOINT_NAMES):
            raise RuntimeError(
                f"policy manifest control {name!r} has {len(values)} values, expected {len(JOINT_NAMES)}"
            )
        return values

    def _configure_robot_for_playback(self):
        self._set_robot_editor_transform(self.reset_base_position)
        for index, joint in enumerate(self.joints):
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = self.joint_kp[index]
            joint.drive_damping = self.joint_kd[index]
        self.robot.mode = gobot.RobotMode.Motion

    def _resolve_reset_base_position(self):
        origins = getattr(self, "terrain_origins", None)
        if origins is None:
            origins = self._terrain_spawn_origins()
        if not origins:
            return [0.0, 0.0, self.reset_base_height]
        spawn = min(origins, key=lambda value: value[0] * value[0] + value[1] * value[1])
        return [spawn[0], spawn[1], spawn[2] + self.reset_base_height]

    def _terrain_spawn_origins(self):
        terrain = self._terrain_node()
        origins = getattr(terrain, "spawn_origins", None) if terrain is not None else None
        if origins is None:
            return []
        parsed = []
        for origin in origins:
            try:
                values = [float(origin[index]) for index in range(3)]
            except (TypeError, IndexError, ValueError):
                continue
            parsed.append(values)
        return parsed

    def _terrain_node(self):
        root = self.get_root()
        terrain_world = root.find("terrain_world") if root is not None else None
        return terrain_world.find("terrain") if terrain_world is not None else None

    def _terrain_grid_metadata(self):
        terrain = self._terrain_node()
        config = getattr(terrain, "generator_config", None) if terrain is not None else None
        properties = config.get("properties", config) if isinstance(config, dict) else None
        if not isinstance(properties, dict):
            return 0, []
        entries = properties.get("sub_terrains", [])
        names = [str(entry.get("name", f"type_{index}")) for index, entry in enumerate(entries)]
        try:
            rows = int(properties.get("num_rows", 0))
        except (TypeError, ValueError):
            rows = 0
        if rows <= 0 or not names or rows * len(names) != len(self.terrain_origins):
            return 0, []
        return rows, names

    def _terrain_cell(self, position):
        origins = getattr(self, "terrain_origins", [])
        type_names = getattr(self, "terrain_type_names", [])
        rows = getattr(self, "terrain_rows", 0)
        if not origins or not type_names or rows <= 0:
            return None
        index = min(
            range(len(origins)),
            key=lambda value: (
                (origins[value][0] - position[0]) ** 2
                + (origins[value][1] - position[1]) ** 2
            ),
        )
        level, terrain_type = divmod(index, len(type_names))
        return level, type_names[terrain_type], rows - 1

    def _reset_playback_state(self):
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False
        self.command = list(COMMAND)
        self.command_target = list(COMMAND)
        self.run_requested = False
        self.last_action = [0.0] * len(JOINT_NAMES)
        self.last_targets = list(self.default_pos)

    def _print_startup(self):
        print(
            "Go1 policy playback started: task={} obs={} actions={} fixed_dt={:.4f} "
            "policy_dt={:.4f} decimation={} reset_z={:.3f} run_policy={}".format(
                self.manifest.task_name,
                self.policy.obs_dim,
                self.policy.action_dim,
                self.physics_dt,
                self.policy_dt,
                self.decimation,
                self.reset_base_position[2],
                "enabled" if self.run_policy is not None else "base_fallback",
            )
        )

    def _physics_process(self, delta):
        if not self.playing or not self._ensure_world_controls():
            return
        if self._update_keyboard_command(delta):
            return

        policy_tick = self.ticks % self.decimation == 0
        robot_state = self._runtime_robot_state() if policy_tick else None
        if robot_state is not None and self._reset_if_fallen(robot_state):
            return
        if policy_tick:
            active_policy = self._active_policy()
            action = active_policy.action(self._observation(robot_state))
            if len(action) != len(JOINT_NAMES):
                raise RuntimeError(
                    f"Go1 policy produced {len(action)} actions, expected {len(JOINT_NAMES)}"
                )
            self.last_action = [float(value) for value in action]
            self.last_targets = self._action_targets(self.last_action)
            self._set_joint_position_targets(self.last_targets)

        self.ticks += 1
        if self.ticks % self.print_every_ticks == 0:
            self._print_state(robot_state)

    def reset(self):
        self._reset_playback_state()
        self._set_robot_editor_transform(self.reset_base_position)

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def _load_policy(self):
        return self._load_policy_candidates(
            env_name=POLICY_ENV,
            defaults=(DEFAULT_POLICY_PATH, DEFAULT_TORCH_POLICY_PATH),
            label="base",
            required=True,
        )

    def _load_run_policy(self):
        return self._load_policy_candidates(
            env_name=RUN_POLICY_ENV,
            defaults=(DEFAULT_RUN_POLICY_PATH, DEFAULT_RUN_TORCH_POLICY_PATH),
            label="run",
            required=False,
        )

    def _load_policy_candidates(self, *, env_name, defaults, label, required):
        requested_policy = os.environ.get(env_name)
        if requested_policy is not None and not requested_policy.strip():
            raise RuntimeError(f"{env_name} is empty; provide a manifest-backed .onnx or .pt policy")
        policy_refs = (requested_policy,) if requested_policy is not None else defaults
        candidates = [_resolve_project_path(self.context, policy_ref) for policy_ref in policy_refs]
        path = next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), None)
        if path is None:
            if not required and requested_policy is None:
                return None
            raise FileNotFoundError(
                "Go1 {} policy not found; tried {}. Train or export a manifest-backed policy, "
                "or set {} explicitly.".format(
                    label,
                    ", ".join(repr(candidate) for candidate in candidates),
                    env_name,
                )
            )
        extension = _policy_extension(path)
        print(f"Go1 loading {label} policy: {path}")
        if extension == ".onnx":
            return OnnxPolicy(path)
        if extension == ".pt":
            return TorchPolicy(path)
        raise RuntimeError(f"unsupported Go1 policy format {extension!r}; expected .onnx or .pt")

    def _active_policy(self):
        run_policy = getattr(self, "run_policy", None)
        return run_policy if getattr(self, "run_requested", False) and run_policy is not None else self.policy

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
        self.world_controls_ready = True
        return True

    def _set_joint_position_targets(self, targets):
        for joint, target in zip(self.joints, targets):
            joint.set_position_target(float(target))

    def _action_targets(self, action):
        targets = []
        for index in range(len(self.joints)):
            raw_action = float(action[index])
            clipped_action = (
                raw_action
                if self.action_clip is None
                else _clamp(raw_action, -self.action_clip, self.action_clip)
            )
            target = self.default_pos[index] + self.action_scale[index] * clipped_action
            targets.append(target)
        return targets

    def _set_robot_editor_transform(self, position):
        self.robot.set_transform(
            [float(position[0]), float(position[1]), float(position[2]) - ROBOT_ROOT_TO_BASE_Z],
            [1.0, 0.0, 0.0, 0.0],
        )

    def _update_keyboard_command(self, delta):
        input_state = getattr(self.context, "input", None)
        if input_state is None:
            self.run_requested = False
            return False
        if _key_pressed(input_state, "reset"):
            self.reset()
            return True
        if not input_state.has_control_focus or _key_held(input_state, "stop"):
            desired = [0.0, 0.0, 0.0]
            self.run_requested = False
        else:
            forward_axis = _key_axis(input_state, "backward", "forward")
            forward_limit = KEYBOARD_COMMAND_MAX[0]
            run_held = _key_held(input_state, "run")
            self.run_requested = run_held and forward_axis > 0.0
            if run_held:
                forward_limit = (
                    KEYBOARD_RUN_COMMAND_X[1]
                    if forward_axis >= 0.0
                    else abs(KEYBOARD_RUN_COMMAND_X[0])
                )
            desired = [
                forward_limit * forward_axis,
                KEYBOARD_COMMAND_MAX[1] * _key_axis(input_state, "strafe_right", "strafe_left"),
                KEYBOARD_COMMAND_MAX[2] * _key_axis(input_state, "turn_right", "turn_left"),
            ]
            normalized_planar_speed = math.hypot(
                desired[0] / forward_limit,
                desired[1] / KEYBOARD_COMMAND_MAX[1],
            )
            if normalized_planar_speed > 1.0:
                desired[0] /= normalized_planar_speed
                desired[1] /= normalized_planar_speed
        alpha = _clamp(float(delta) * COMMAND_SMOOTHING, 0.0, 1.0)
        for index in range(3):
            self.command_target[index] += (desired[index] - self.command_target[index]) * alpha
        self.command = list(self.command_target)
        return False

    def _reset_if_fallen(self, state):
        base = self._base_link_state(state)
        if base is None:
            return False
        position = base.get("position", [0.0, 0.0, 0.0])
        quaternion = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        roll, pitch = _quat_to_roll_pitch(quaternion)
        terrain_clearance = self._terrain_scan_values(state)[TERRAIN_SCAN_DIM // 2]
        if (
            terrain_clearance >= FALLEN_BASE_CLEARANCE
            and abs(roll) <= FALLEN_ROLL_PITCH
            and abs(pitch) <= FALLEN_ROLL_PITCH
        ):
            return False
        print(
            "Go1 reset after fall: z={:.3f} clearance={:.3f} roll={:.2f} pitch={:.2f}".format(
                position[2], terrain_clearance, roll, pitch
            )
        )
        self.reset()
        return True

    def _print_state(self, state=None):
        if state is None:
            state = self._runtime_robot_state()
        base = self._base_link_state(state) if state is not None else None
        position = base.get("position", [0.0, 0.0, 0.0]) if base is not None else [0.0, 0.0, 0.0]
        quaternion = base.get("quaternion", [1.0, 0.0, 0.0, 0.0]) if base is not None else [1.0, 0.0, 0.0, 0.0]
        linear = _quat_rotate_inv(base.get("linear_velocity", [0.0, 0.0, 0.0]), quaternion) if base is not None else [0.0, 0.0, 0.0]
        angular = _quat_rotate_inv(base.get("angular_velocity", [0.0, 0.0, 0.0]), quaternion) if base is not None else [0.0, 0.0, 0.0]
        terrain_cell = self._terrain_cell(position)
        terrain_label = (
            "unknown"
            if terrain_cell is None
            else "{}:L{}/{}".format(terrain_cell[1], terrain_cell[0], terrain_cell[2])
        )
        print(
            "Go1 t={:.2f}s base=({:.3f},{:.3f},{:.3f}) lin_b=({:.3f},{:.3f},{:.3f}) "
            "ang_b=({:.3f},{:.3f},{:.3f}) cmd=({:.2f},{:.2f},{:.2f}) "
            "mode={} terrain={} action_norm={:.3f}".format(
                self.context.simulation_time,
                *position[:3],
                *linear[:3],
                *angular[:3],
                *self.command,
                "run" if self._active_policy() is getattr(self, "run_policy", None) else "walk",
                terrain_label,
                math.sqrt(sum(value * value for value in self.last_action)),
            )
        )

    def _observation(self, state):
        if state is None:
            return [0.0] * ACTOR_OBS_SPEC.dim
        base = self._base_link_state(state)
        if base is None:
            return [0.0] * ACTOR_OBS_SPEC.dim
        quaternion = base.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        linear_velocity, angular_velocity = self._imu_velocity(state)
        projected_gravity = _quat_rotate_inv([0.0, 0.0, -1.0], quaternion)
        joint_states = {joint.get("name"): joint for joint in state.get("joints", [])}
        joint_position = []
        joint_velocity = []
        for index, joint_name in enumerate(JOINT_NAMES):
            joint = joint_states.get(joint_name, {})
            joint_position.append(
                float(joint.get("position", self.default_pos[index])) - self.default_pos[index]
            )
            joint_velocity.append(float(joint.get("velocity", 0.0)))
        observation = build_velocity_actor_observation(
            base_lin_vel_b=linear_velocity,
            base_ang_vel_b=angular_velocity,
            projected_gravity=projected_gravity,
            joint_pos_rel=joint_position,
            joint_vel=joint_velocity,
            last_action=self.last_action,
            command=self.command,
            height_scan=self._height_scan(state),
        )
        if observation.shape != (ACTOR_OBS_SPEC.dim,):
            raise RuntimeError(
                f"Go1 playback observation shape mismatch: {observation.shape} != {(ACTOR_OBS_SPEC.dim,)}"
            )
        return observation

    def _imu_velocity(self, robot_state):
        sensor = self._sensor_map(robot_state).get(IMU_SENSOR)
        if sensor is None:
            raise RuntimeError(f"Go1 runtime state is missing sensor {IMU_SENSOR!r}")
        values = [float(value) for value in sensor.get("values", [])]
        if len(values) < 10 or not all(math.isfinite(value) for value in values[:10]):
            raise RuntimeError(
                f"Go1 sensor {IMU_SENSOR!r} produced {len(values)} values; "
                "expected at least 10 finite values"
            )
        return values[7:10], values[4:7]

    def _height_scan(self, robot_state):
        values = self._terrain_scan_values(robot_state)
        return [value / self.height_scan_max_distance for value in values]

    def _terrain_scan_values(self, robot_state):
        sensor = self._sensor_map(robot_state).get(TERRAIN_SCAN_SENSOR)
        if sensor is None:
            raise RuntimeError(f"Go1 runtime state is missing sensor {TERRAIN_SCAN_SENSOR!r}")
        values = [float(value) for value in sensor.get("values", [])]
        if len(values) != TERRAIN_SCAN_DIM or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Go1 sensor {TERRAIN_SCAN_SENSOR!r} produced {len(values)} values; "
                f"expected {TERRAIN_SCAN_DIM} finite values"
            )
        return values

    def _sensor_map(self, robot_state):
        return {
            sensor.get("name") or sensor.get("sensor_name"): sensor
            for sensor in robot_state.get("sensors", [])
        }

    def _base_link_state(self, robot_state):
        if robot_state is None:
            return None
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
        return {
            "name": self.robot.name,
            "links": [self.base_link.get_runtime_state()],
            "joints": [joint.get_runtime_state() for joint in self.joints],
            "sensors": [self.imu.get_runtime_state(), self.terrain_scan.get_runtime_state()],
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
            raise RuntimeError(f"scene has no robot node {ROBOT!r}")
        return robot

    def _find_joint(self, name):
        joint = _find_node_by_name(self.robot, name)
        if joint is None:
            raise RuntimeError(f"robot {self.robot.name!r} has no joint {name!r}")
        return joint

    def _find_link(self, name):
        link = _find_node_by_name(self.robot, name)
        if link is None:
            raise RuntimeError(f"robot {self.robot.name!r} has no link {name!r}")
        return link

    def _find_sensor(self, name):
        sensor = _find_node_by_name(self.robot, name)
        if sensor is None:
            raise RuntimeError(f"robot {self.robot.name!r} has no sensor {name!r}")
        return sensor
