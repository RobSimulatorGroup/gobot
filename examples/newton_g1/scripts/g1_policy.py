"""Gobot-rendered playback of Newton's pretrained Unitree G1 policy."""

from __future__ import annotations

import json
import os
import time

import gobot
from gobot.rl.providers import NewtonModelConfig, NewtonProvider

from scripts.g1_policy_contract import (
    ACTION_DIM,
    BASE_LINK,
    BASE_POSE_XYZW,
    JOINT_NAMES,
    LINK_NAMES,
    OBSERVATION_DIM,
    PHYSICS_DT,
    POLICY_DECIMATION,
    POLICY_DT,
    load_native_policy_contract,
)


TASK_CONFIG_PATH = "res://newton_g1.task.json"
PRINT_INTERVAL_SECONDS = 2.0


def _load_task_config(project_path):
    path = os.path.join(project_path, TASK_CONFIG_PATH.removeprefix("res://"))
    with open(path, encoding="utf-8") as stream:
        task = json.load(stream)
    expected = {
        "version": 1,
        "fixed_dt": PHYSICS_DT,
        "policy_decimation": POLICY_DECIMATION,
        "actions": ACTION_DIM,
        "observations": OBSERVATION_DIM,
        "initial_base_pose_xyzw": list(BASE_POSE_XYZW),
    }
    actual = {
        "version": task.get("version"),
        "fixed_dt": task.get("physics", {}).get("fixed_dt"),
        "policy_decimation": task.get("physics", {}).get("policy_decimation"),
        "actions": task.get("dimensions", {}).get("actions"),
        "observations": task.get("dimensions", {}).get("observations"),
        "initial_base_pose_xyzw": task.get("physics", {}).get("initial_base_pose_xyzw"),
    }
    if actual != expected:
        raise RuntimeError(
            f"Newton G1 task config does not match the pinned policy contract: {actual!r}"
        )
    return task


def _walk_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _nodes_by_name(root, names, *, type_name=None):
    expected = set(names)
    result = {}
    for node in _walk_nodes(root):
        if node.name not in expected:
            continue
        if type_name is not None and node.type_name != type_name:
            continue
        if node.name in result:
            raise RuntimeError(f"scene contains more than one {type_name or 'node'} named {node.name!r}")
        result[node.name] = node
    missing = [name for name in names if name not in result]
    if missing:
        raise RuntimeError(f"scene is missing {type_name or 'node'} nodes: {', '.join(missing)}")
    return result


def _key_axis(input_state, negative, positive, scale):
    value = 0.0
    if input_state.is_key_held(negative):
        value -= scale
    if input_state.is_key_held(positive):
        value += scale
    return value


def _validate_native_g1_scene(robot, joints, native_contract):
    """Reject generated scenes that predate the versioned task contract."""

    default_position = native_contract["mjw_joint_pos"]
    stiffness = native_contract["mjw_joint_stiffness"]
    damping = native_contract["mjw_joint_damping"]
    armature = native_contract["mjw_joint_armature"]
    for index, name in enumerate(JOINT_NAMES):
        joint = joints[name]
        expected = (
            ("drive_stiffness", stiffness[index]),
            ("drive_damping", damping[index]),
            ("armature", armature[index]),
            ("initial_position", default_position[index]),
            ("friction_loss", 0.0),
            ("effort_limit", 0.0),
            ("force_lower_limit", 0.0),
            ("force_upper_limit", 0.0),
        )
        if joint.drive_mode != gobot.JointDriveMode.Position or any(
            abs(float(getattr(joint, field)) - float(value)) > 1.0e-6
            for field, value in expected
        ):
            raise RuntimeError(
                f"generated G1 scene has stale physics settings for {name!r}; "
                "re-run the project asset hook to rebuild its cache"
            )

    # Newton's reference G1 import disables articulation self-collision. Keep
    # that example policy out of the generic USD importer while retaining
    # collision with the world (world shapes use layer/mask 1/1).
    for node in _walk_nodes(robot):
        if node.type_name == "CollisionShape3D" and (
            int(node.collision_layer) != 0 or int(node.collision_mask) != 1
        ):
            raise RuntimeError(
                "generated G1 scene has stale collision filters; re-run the project asset hook"
            )
    if robot.mode != gobot.RobotMode.Motion:
        raise RuntimeError(
            "generated G1 scene has a stale Robot3D mode; re-run the project asset hook"
        )


def _quat_rotate_inverse(torch, quaternion_xyzw, vector):
    vector_part = quaternion_xyzw[:, :3]
    scalar_part = quaternion_xyzw[:, 3:4]
    first_cross = torch.cross(vector_part, vector, dim=1)
    second_cross = torch.cross(vector_part, first_cross, dim=1)
    return vector + 2.0 * (second_cross - scalar_part * first_cross)


class WarpOnnxPolicy:
    """Small Warp-NN wrapper that keeps policy input and output on CUDA."""

    def __init__(self, path, *, device, torch):
        try:
            import warp as wp
            from warp_nn.runtime import OnnxRuntime
        except ImportError as error:
            raise ImportError(
                "Newton G1 playback requires Gobot's newton[onnx,sim] dependency"
            ) from error

        self.wp = wp
        self.torch = torch
        self.device = wp.get_device(device)
        self.runtime = OnnxRuntime(path, device=self.device)
        # OnnxRuntime uploads weights asynchronously on Warp's current stream.
        # Complete that one-time setup before inference follows Torch's stream.
        wp.synchronize_device(self.device)
        if len(self.runtime.input_names) != 1 or len(self.runtime.output_names) != 1:
            raise RuntimeError("G1 policy must have exactly one input and one output")
        self.input_name = self.runtime.input_names[0]
        self.output_name = self.runtime.output_names[0]

    def action(self, observation):
        if tuple(observation.shape) != (1, OBSERVATION_DIM):
            raise RuntimeError(
                f"G1 policy expected observation shape (1, {OBSERVATION_DIM}), "
                f"got {tuple(observation.shape)}"
            )
        stream = self.wp.stream_from_torch(self.torch.cuda.current_stream(observation.device))
        with self.wp.ScopedStream(stream, sync_enter=False):
            output = self.runtime(
                {self.input_name: self.wp.from_torch(observation.contiguous())}
            )[self.output_name]
        action = self.wp.to_torch(output)
        if tuple(action.shape) != (1, ACTION_DIM):
            raise RuntimeError(
                f"G1 policy produced shape {tuple(action.shape)}, expected (1, {ACTION_DIM})"
            )
        return action


class Script(gobot.NodeScript):
    def _startup_begin(self, message):
        self._startup_stage_started_at = time.perf_counter()
        print(f"Newton G1 startup: {message}...", flush=True)

    def _startup_finish(self, message):
        now = time.perf_counter()
        stage_seconds = now - self._startup_stage_started_at
        total_seconds = now - self._startup_started_at
        print(
            f"Newton G1 startup: {message} in {stage_seconds:.2f}s "
            f"(total {total_seconds:.2f}s)",
            flush=True,
        )

    def _ready(self):
        self._startup_started_at = time.perf_counter()
        self._startup_stage_started_at = self._startup_started_at
        self._first_frame_warmup_started_at = None
        self.provider = None
        self.play_session = None
        self.ticks = 0
        self.command = None
        self.previous_action = None

        try:
            self._startup_begin("validating the Gobot scene and policy contract")
            root = self.get_root()
            if root is None:
                raise RuntimeError("Newton G1 script has no scene root")
            task_config = _load_task_config(self.context.project_path)
            resources = task_config["resources"]
            robots = [node for node in _walk_nodes(root) if node.type_name == "Robot3D"]
            if len(robots) != 1:
                raise RuntimeError(
                    f"Newton G1 scene must contain exactly one Robot3D, got {len(robots)}"
                )
            self.robot = robots[0]
            self.links = _nodes_by_name(self.robot, LINK_NAMES, type_name="Link3D")
            joints = _nodes_by_name(self.robot, JOINT_NAMES, type_name="Joint3D")

            contract_path = os.path.join(
                self.context.project_path,
                resources["policy_contract"].removeprefix("res://"),
            )
            native_contract = load_native_policy_contract(contract_path)
            if native_contract["mjw_joint_names"] != JOINT_NAMES:
                raise RuntimeError("G1 USD joint order does not match the released policy YAML")
            if native_contract["num_dofs"] != ACTION_DIM:
                raise RuntimeError("G1 policy YAML does not describe 43 joints")
            self.action_scale = float(native_contract["action_scale"])
            default_position = native_contract["mjw_joint_pos"]
            _validate_native_g1_scene(self.robot, joints, native_contract)
            self._startup_finish("scene and policy contract validated")

            self._startup_begin("compiling the Gobot scene artifact")
            artifact = self.context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
            robot_names = tuple(artifact.get("robot_names", ()))
            if len(robot_names) != 1:
                raise RuntimeError(
                    f"Newton G1 scene must compile exactly one robot, got {robot_names}"
                )
            if int(artifact["dimensions"]["nu"]) != ACTION_DIM:
                raise RuntimeError(
                    "G1 compiled artifact must contain one position drive per policy "
                    f"joint, got nu={artifact['dimensions']['nu']}"
                )
            self._startup_finish("Gobot scene artifact compiled")

            device = os.environ.get("GOBOT_NEWTON_DEVICE", "cuda:0")
            self._startup_begin(f"initializing the Newton provider on {device}")
            self.provider = NewtonProvider(
                artifact,
                num_envs=1,
                device=device,
                fixed_time_step=PHYSICS_DT,
                nconmax=30,
                njmax=100,
                use_mujoco_contacts=True,
                model_config=NewtonModelConfig(**task_config["newton_model"]),
            )
            if not self.provider.use_mujoco_contacts:
                raise RuntimeError("Newton G1 playback requires the official MuJoCo contact path")
            self._startup_finish("Newton provider initialized")

            self._startup_begin("resolving the G1 layout and allocating CUDA tensors")
            self.robot_view = self.provider.create_robot_view(
                robot_name=robot_names[0],
                base_link=BASE_LINK,
                joint_names=JOINT_NAMES,
                link_names=LINK_NAMES,
                scene_context=self.context,
                scene_links=tuple(self.links[name] for name in LINK_NAMES),
            )

            arrays = self.provider.arrays
            import torch

            self.torch = torch
            self.device = arrays["joint_q"].device
            self.default_joint_position = torch.as_tensor(
                default_position,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, ACTION_DIM)
            self.command = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            self.previous_action = torch.zeros(
                (1, ACTION_DIM), dtype=torch.float32, device=self.device
            )
            self.gravity = torch.tensor(
                [[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device
            )
            self._startup_finish("G1 layout and CUDA tensors ready")

            policy_path = os.path.join(
                self.context.project_path,
                resources["policy"].removeprefix("res://"),
            )
            if not os.path.isfile(policy_path):
                raise FileNotFoundError(
                    f"Newton G1 policy is missing: {policy_path}. Re-run the project asset hook."
                )
            self._startup_begin("loading the Warp ONNX policy and uploading its weights")
            self.policy = WarpOnnxPolicy(policy_path, device=device, torch=torch)
            self._startup_finish("Warp ONNX policy loaded")

            self._startup_begin("resetting Newton state and synchronizing Gobot transforms")
            self._reset_provider()
            self._sync_robot_links()
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.provider,
                fixed_dt=PHYSICS_DT,
                max_sub_steps=max(POLICY_DECIMATION, 8),
                reset=self._reset_provider,
                sync_scene=self._sync_robot_links,
            ).start()
            self.play_session.set_status("Ready; first policy and physics tick pending")
            self._startup_finish("initial Newton state synchronized")
            print(
                "Newton G1 policy playback started: physics=Newton renderer=Gobot "
                f"obs={OBSERVATION_DIM} actions={ACTION_DIM} fixed_dt={PHYSICS_DT:.4f} "
                f"policy_dt={POLICY_DT:.4f} device={device}; the first simulation frame "
                "will warm up CUDA kernels"
            )
        except Exception:
            try:
                self._close_play_session()
            except Exception as cleanup_error:
                print(f"Newton G1 provider cleanup failed: {cleanup_error}")
            raise

    def _physics_process(self, delta):
        del delta
        if self.provider is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            if self.play_session is not None:
                self.play_session.reset()
            return

        if input_state is not None and input_state.has_control_focus:
            command = (
                _key_axis(input_state, "K", "I", 1.0),
                _key_axis(input_state, "L", "J", 0.5),
                _key_axis(input_state, "O", "U", 1.0),
            )
        else:
            command = (0.0, 0.0, 0.0)
        self.command.copy_(self.torch.tensor([command], device=self.device))

        if self.ticks % POLICY_DECIMATION == 0:
            if self.ticks == 0:
                if self.play_session is not None:
                    self.play_session.set_status("Warming up policy and physics CUDA kernels")
                self._first_frame_warmup_started_at = time.perf_counter()
                print(
                    "Newton G1 startup: warming up the first policy and physics CUDA kernels...",
                    flush=True,
                )
            action = self.policy.action(self._observation())
            targets = self.default_joint_position + self.action_scale * action
            self.robot_view.set_position_targets(targets)
            self.previous_action.copy_(action)

        self.ticks += 1
        print_every = max(1, int(round(PRINT_INTERVAL_SECONDS / PHYSICS_DT)))
        if self.ticks % print_every == 0:
            state = self.robot_view.read_state()
            base = state.base_pose[0]
            speed = state.base_velocity[0]
            self.provider.synchronize()
            print(
                "Newton G1 t={:.2f}s base=({:.3f},{:.3f},{:.3f}) "
                "lin=({:.3f},{:.3f},{:.3f}) cmd=({:.2f},{:.2f},{:.2f})".format(
                    self.ticks * PHYSICS_DT,
                    *base[:3].detach().cpu().tolist(),
                    *speed[:3].detach().cpu().tolist(),
                    *command,
                )
            )

    def _process(self, delta):
        del delta
        if self.provider is not None:
            if self._first_frame_warmup_started_at is not None:
                elapsed = time.perf_counter() - self._first_frame_warmup_started_at
                print(
                    f"Newton G1 startup: first CUDA simulation frame ready in {elapsed:.2f}s",
                    flush=True,
                )
                self._first_frame_warmup_started_at = None
                if self.play_session is not None:
                    self.play_session.set_status("Running")

    def _exit_tree(self):
        self._close_play_session()

    def _close_play_session(self):
        play_session = self.play_session
        self.play_session = None
        if play_session is not None:
            play_session.close()
        provider = self.provider
        self.provider = None
        if provider is not None and play_session is None:
            provider.close()

    def _observation(self):
        state = self.robot_view.read_state()
        base_pose = state.base_pose
        base_velocity = state.base_velocity
        joint_position = state.joint_position
        joint_velocity = state.joint_velocity
        orientation = base_pose[:, 3:7]
        observation = self.torch.cat(
            (
                _quat_rotate_inverse(self.torch, orientation, base_velocity[:, :3]),
                _quat_rotate_inverse(self.torch, orientation, base_velocity[:, 3:6]),
                _quat_rotate_inverse(self.torch, orientation, self.gravity),
                self.command,
                joint_position - self.default_joint_position,
                joint_velocity,
                self.previous_action,
            ),
            dim=1,
        )
        if tuple(observation.shape) != (1, OBSERVATION_DIM):
            raise RuntimeError(
                f"Newton G1 observation shape is {tuple(observation.shape)}, "
                f"expected (1, {OBSERVATION_DIM})"
            )
        return observation

    def _reset_provider(self):
        torch = self.torch
        base_pose = torch.as_tensor(
            BASE_POSE_XYZW,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 7)
        zeros6 = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        zeros43 = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=self.device)
        self.robot_view.reset(
            torch.ones(1, dtype=torch.bool, device=self.device),
            base_pose=base_pose,
            base_velocity=zeros6,
            joint_position=self.default_joint_position,
            joint_velocity=zeros43,
            controls=self.default_joint_position,
        )
        self.previous_action.zero_()
        self.command.zero_()
        self.ticks = 0

    def _sync_robot_links(self):
        self.provider.synchronize()
        self.robot_view.sync_scene()
