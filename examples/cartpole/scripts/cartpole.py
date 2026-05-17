import math
import os
import random

import gobot


ROBOT = "cartpole"
SLIDER_JOINT = "slider"
HINGE_JOINT = "hinge"
TARGET_CART_POSITION = 0.0
FORCE_LIMIT = 3.0
PRINT_EVERY_TICKS = 240
DEFAULT_POLICY_PATH = "res://policies/cartpole.pt"
INITIAL_CART_POSITION = 0.0
INITIAL_POLE_ANGLE = 0.0
DISTURBANCE_ENABLED = True
DISTURBANCE_STD = 0.05
DISTURBANCE_CLIP = 0.20
DISTURBANCE_INTERVAL_TICKS = 480
DISTURBANCE_DURATION_TICKS = 60
DISTURBANCE_START_TICK = 240


class TorchPolicy:
    def __init__(self, path):
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.uses_normalized_action = False
        self.policy = self._load_policy_module(checkpoint)
        self.policy.to(self.device)
        self.policy.eval()

    def action(self, observation):
        with self.torch.no_grad():
            observation = self._adapt_observation(observation)
            obs = self.torch.as_tensor(observation, dtype=self.torch.float32, device=self.device).reshape(1, -1)
            output = self.policy(obs)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if self.uses_normalized_action:
                return float(output.reshape(-1)[0].clamp(-1.0, 1.0).cpu().item()) * FORCE_LIMIT
            return float(output.reshape(-1)[0].clamp(-FORCE_LIMIT, FORCE_LIMIT).cpu().item())

    def _load_policy_module(self, checkpoint):
        if isinstance(checkpoint, self.torch.nn.Module):
            return checkpoint
        if not isinstance(checkpoint, dict):
            raise TypeError("expected a torch module or checkpoint dictionary")

        state = checkpoint.get("actor_state_dict") or checkpoint.get("state_dict") or checkpoint
        if not isinstance(state, dict):
            raise TypeError("checkpoint has no actor_state_dict/state_dict")

        mlp_state = {}
        for key, value in state.items():
            mlp_index = str(key).find("mlp.")
            if mlp_index >= 0:
                mlp_state[str(key)[mlp_index:]] = value
        if not mlp_state:
            raise ValueError("checkpoint has no rsl_rl MLP actor weights")

        activation = checkpoint.get("activation")
        policy_meta = checkpoint.get("gobot_policy")
        if activation is None and isinstance(policy_meta, dict):
            activation = policy_meta.get("activation")
        self.uses_normalized_action = isinstance(policy_meta, dict)
        return self._build_mlp(mlp_state, str(activation or "elu"), state)

    def _build_mlp(self, state, activation, checkpoint_state):
        torch = self.torch
        linear_indices = sorted(
            int(key.split(".")[1])
            for key in state
            if key.startswith("mlp.") and key.endswith(".weight")
        )
        if not linear_indices:
            raise ValueError("checkpoint has no linear layer weights")

        layers = []
        last_index = linear_indices[-1]
        for layer_index in linear_indices:
            weight = state[f"mlp.{layer_index}.weight"]
            bias = state[f"mlp.{layer_index}.bias"]
            layers.append(torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
            if bias.shape[0] != weight.shape[0]:
                raise ValueError(f"invalid bias shape for layer {layer_index}")
            if layer_index != last_index:
                layers.append(self._activation_module(activation))

        class MlpPolicy(torch.nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.mlp = torch.nn.Sequential(*modules)

            def forward(self, x):
                return self.mlp(x)

        module = MlpPolicy(layers)
        module.load_state_dict(state, strict=False)
        self.input_size = int(layers[0].in_features)
        if "obs_normalizer._mean" in checkpoint_state:
            mean = checkpoint_state["obs_normalizer._mean"].detach().to(dtype=self.torch.float32)
            std = checkpoint_state.get("obs_normalizer._std", self.torch.ones_like(mean)).detach().to(dtype=self.torch.float32)
            std = self.torch.clamp(std, min=1.0e-6)
            module = self.torch.nn.Sequential(self._Normalizer(mean, std), module)
        return module

    def _activation_module(self, activation):
        name = activation.lower()
        if name == "elu":
            return self.torch.nn.ELU()
        if name == "relu":
            return self.torch.nn.ReLU()
        if name == "tanh":
            return self.torch.nn.Tanh()
        raise ValueError(f"unsupported policy activation: {activation}")

    class _Normalizer:
        def __new__(cls, mean, std):
            import torch

            class Normalizer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("mean", mean)
                    self.register_buffer("std", std)

                def forward(self, x):
                    return (x - self.mean) / self.std

            return Normalizer()

    def _adapt_observation(self, observation):
        if len(observation) == self.input_size:
            return observation
        if self.input_size == 5 and len(observation) == 7:
            cos_theta, sin_theta, x, x_dot, theta_dot, target, pos_error = observation
            theta = math.atan2(sin_theta, cos_theta)
            return [x, x_dot, theta, theta_dot, target - x]
        raise ValueError(f"policy expects {self.input_size} observations, got {len(observation)}")



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
        self.rng = random.Random(int(os.environ.get("GOBOT_CARTPOLE_DISTURBANCE_SEED", "42")))
        self.ticks = 0
        self.playing = True
        self.world_controls_ready = False

        self.slider.effort_limit = FORCE_LIMIT
        self.slider.velocity_limit = max(float(getattr(self.slider, "velocity_limit", 0.0)), 20.0)
        self.hinge.effort_limit = max(float(getattr(self.hinge, "effort_limit", 0.0)), DISTURBANCE_CLIP)
        print(
            "CartPole RL policy playback started. policy={} force_limit={:.1f}N disturbance_std={:.3f}Nm".format(
                "loaded" if self.policy is not None else "missing",
                FORCE_LIMIT,
                DISTURBANCE_STD if DISTURBANCE_ENABLED else 0.0,
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
        action = _clamp(action, -FORCE_LIMIT, FORCE_LIMIT)
        effort = action
        self.context.set_joint_effort_target(self.robot.name, SLIDER_JOINT, effort)
        disturbance = self._sample_disturbance()
        if disturbance != 0.0:
            self.context.set_joint_effort_target(self.robot.name, HINGE_JOINT, disturbance)
        else:
            self.context.set_joint_passive(self.robot.name, HINGE_JOINT)

        self.ticks += 1
        if PRINT_EVERY_TICKS > 0 and self.ticks % PRINT_EVERY_TICKS == 0:
            self._print_state("tick", action, effort, disturbance)

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
            print(
                "CartPole RL policy not found at '{}'; set GOBOT_CARTPOLE_POLICY to a .pt policy.".format(
                    path or "<empty>"
                )
            )
            return None
        try:
            print(f"CartPole RL loading policy: {path}")
            return TorchPolicy(path)
        except Exception as error:
            print(f"CartPole RL policy load failed: {error}")
            return None

    def _ensure_world_controls(self):
        if self.world_controls_ready:
            return True
        if not self.context.has_world:
            return False
        reset_joint_state = getattr(self.context, "reset_joint_state", None)
        if reset_joint_state is not None:
            reset_joint_state(self.robot.name, SLIDER_JOINT, INITIAL_CART_POSITION, 0.0)
            reset_joint_state(self.robot.name, HINGE_JOINT, INITIAL_POLE_ANGLE, 0.0)
        else:
            self.slider.joint_position = INITIAL_CART_POSITION
            self.hinge.joint_position = INITIAL_POLE_ANGLE
            self.context.rebuild_world(False)
        self.context.set_joint_passive(self.robot.name, HINGE_JOINT)
        self.world_controls_ready = True
        self.previous_x = INITIAL_CART_POSITION
        self.previous_theta = INITIAL_POLE_ANGLE
        self.observation = self._observation(0.0)
        self._print_state("origin", 0.0, 0.0, 0.0)
        return True

    def _print_state(self, label, action, effort, disturbance):
        cos_theta, sin_theta, x, x_dot, theta_dot, target, pos_error = self.observation
        theta = math.atan2(sin_theta, cos_theta)
        print(
            "CartPole RL {} t={:.2f}s cart_position={:.3f}m x_dot={:.3f}m/s "
            "theta={:.4f}rad theta_dot={:.3f}rad/s target={:.3f}m error={:.3f}m "
            "action={:.3f} applied_force={:.3f}N "
            "pole_noise={:.4f}Nm".format(
                label,
                self.context.simulation_time,
                x,
                x_dot,
                theta,
                theta_dot,
                target,
                pos_error,
                action,
                effort,
                disturbance,
            )
        )

    def _sample_disturbance(self):
        if not DISTURBANCE_ENABLED or DISTURBANCE_STD <= 0.0:
            return 0.0
        if self.ticks < DISTURBANCE_START_TICK:
            return 0.0
        if DISTURBANCE_INTERVAL_TICKS > 0:
            offset = (self.ticks - DISTURBANCE_START_TICK) % DISTURBANCE_INTERVAL_TICKS
            if offset >= max(1, DISTURBANCE_DURATION_TICKS):
                return 0.0
        value = self.rng.gauss(0.0, DISTURBANCE_STD)
        if DISTURBANCE_CLIP > 0.0:
            value = _clamp(value, -DISTURBANCE_CLIP, DISTURBANCE_CLIP)
        return value

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
                return [
                    math.cos(theta),
                    math.sin(theta),
                    x,
                    x_dot,
                    theta_dot,
                    self.target_cart_position,
                    x - self.target_cart_position,
                ]

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
        return [
            math.cos(theta),
            math.sin(theta),
            x,
            x_dot,
            theta_dot,
            self.target_cart_position,
            x - self.target_cart_position,
        ]

    def _runtime_joint_states(self):
        if not self.context.has_world:
            return None
        state = self.context.get_runtime_state()
        for robot in state.get("robots", []):
            if robot.get("name") == self.robot.name:
                return {joint.get("name"): joint for joint in robot.get("joints", [])}
        return None
