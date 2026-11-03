"""GPU-native locomotion rough-terrain velocity environment using MuJoCo Warp."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import gobot
from gobot.rl import (
    ActionSpec,
    BatchEnvState,
    SpecField,
    TaskRuntimeMetadata,
)
from gobot.sim.providers import (
    CompiledSceneArtifact,
    MuJoCoWarpContactSensorSpec,
    MuJoCoWarpProvider,
    MuJoCoWarpRaycastSensorSpec,
)
from gobot.rl.locomotion import (
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)

from .scene import (
    prepare_velocity_scene,
    terrain_generator_config,
    terrain_spawn_origins,
)
from .training_state import (
    build_terrain_curriculum_state,
    restore_terrain_curriculum_assignments,
)
from .config import VELOCITY_REWARD_TERM_NAMES, VelocityTaskCfg
from .terms import posture_std, locomotion_failure_mask
from .math import find_node_by_name
from .curriculum import progress_masks



_CONTACT_HISTORY_LENGTH = 4


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., 1:4]
    scalar = quaternion[..., 0:1]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + scalar * cross + torch.cross(xyz, cross, dim=-1)


def _quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., 1:4]
    scalar = quaternion[..., 0:1]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector - scalar * cross + torch.cross(xyz, cross, dim=-1)


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    result = torch.zeros((*yaw.shape, 4), dtype=yaw.dtype, device=yaw.device)
    result[..., 0] = torch.cos(half)
    result[..., 3] = torch.sin(half)
    return result


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _bound_gait_score_torch(
    foot_contact: torch.Tensor,
    foot_velocity: torch.Tensor,
    foot_height: torch.Tensor,
    action: torch.Tensor,
    base_velocity_x: torch.Tensor,
    command_x: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    foot_indices: torch.Tensor,
    joint_indices: torch.Tensor,
    motion_std: float,
    action_std: float,
    height_sync_std: float,
    height_separation_std: float,
    trot_penalty: float,
) -> torch.Tensor:
    gait_contact = foot_contact.to(dtype=torch.bool).index_select(1, foot_indices)
    gait_velocity = foot_velocity.index_select(1, foot_indices)
    sagittal = gait_velocity[:, :, (0, 2)]
    front_delta = sagittal[:, 0] - sagittal[:, 1]
    rear_delta = sagittal[:, 2] - sagittal[:, 3]
    motion_error = 0.5 * (
        front_delta.square().sum(dim=1) + rear_delta.square().sum(dim=1)
    )
    motion_sync = torch.exp(
        -motion_error / max(float(motion_std) ** 2, 1.0e-6)
    )
    paired_action = action.index_select(1, joint_indices.flatten()).view(
        action.shape[0], 4, 3
    )
    front_action_delta = paired_action[:, 0, 1:] - paired_action[:, 1, 1:]
    rear_action_delta = paired_action[:, 2, 1:] - paired_action[:, 3, 1:]
    action_error = 0.5 * (
        front_action_delta.square().mean(dim=1)
        + rear_action_delta.square().mean(dim=1)
    )
    action_sync = torch.exp(
        -action_error / max(float(action_std) ** 2, 1.0e-6)
    )
    height = foot_height.index_select(1, foot_indices)
    height_pair_error = 0.5 * (
        (height[:, 0] - height[:, 1]).square()
        + (height[:, 2] - height[:, 3]).square()
    )
    height_sync = torch.exp(
        -height_pair_error / max(float(height_sync_std) ** 2, 1.0e-6)
    )
    front_height = 0.5 * (height[:, 0] + height[:, 1])
    rear_height = 0.5 * (height[:, 2] + height[:, 3])
    height_separation = 1.0 - torch.exp(
        -(front_height - rear_height).square()
        / max(float(height_separation_std) ** 2, 1.0e-6)
    )
    height_gait = height_sync * height_separation
    contact_sync = 1.0 - 0.5 * (
        torch.logical_xor(gait_contact[:, 0], gait_contact[:, 1]).float()
        + torch.logical_xor(gait_contact[:, 2], gait_contact[:, 3]).float()
    )
    gait_contact_float = gait_contact.float()
    front_contact = 0.5 * (
        gait_contact_float[:, 0] + gait_contact_float[:, 1]
    )
    rear_contact = 0.5 * (
        gait_contact_float[:, 2] + gait_contact_float[:, 3]
    )
    support_separation = (front_contact - rear_contact).abs() * contact_sync
    trot_support = (
        (
            gait_contact[:, 0]
            & ~gait_contact[:, 1]
            & ~gait_contact[:, 2]
            & gait_contact[:, 3]
        )
        | (
            ~gait_contact[:, 0]
            & gait_contact[:, 1]
            & gait_contact[:, 2]
            & ~gait_contact[:, 3]
        )
    ).float()
    speed_progress = (base_velocity_x / command_x.clamp(min=0.1)).clamp(
        min=0.0, max=1.0
    )
    return speed_progress * run_mask.float() * (
        0.25 * action_sync
        + 0.15 * motion_sync
        + 0.15 * contact_sync
        + 0.15 * support_separation
        + 0.30 * height_gait
        - float(trot_penalty) * trot_support
    )


def _grid_offsets(
    size: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
) -> tuple[tuple[float, float, float], ...]:
    x = np.arange(-size[0] / 2.0, size[0] / 2.0 + resolution * 0.5, resolution)
    y = np.arange(-size[1] / 2.0, size[1] / 2.0 + resolution * 0.5, resolution)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    return tuple(
        (float(px), float(py), 0.0)
        for px, py in zip(grid_x.reshape(-1), grid_y.reshape(-1), strict=True)
    )


def _foot_ring_offsets() -> tuple[tuple[float, float, float], ...]:
    offsets = [(0.0, 0.0, 0.0)]
    for index in range(4):
        angle = 2.0 * math.pi * index / 4.0
        offsets.append((0.04 * math.cos(angle), 0.04 * math.sin(angle), 0.0))
    return tuple(offsets)


def _profile_contact_specs(prefix, profile, illegal, shape_names):
    terrain = {"secondary_type": "body", "secondary_name": "world"}
    specs = [MuJoCoWarpContactSensorSpec(
        name="feet_ground_contact", primary_type="geom",
        primary_names=tuple(prefix + shape for foot in profile.feet for shape in foot.collision_shapes),
        fields=("found", "force"), reduce="netforce", **terrain),
        MuJoCoWarpContactSensorSpec(
            name="self_collision", primary_type="subtree", primary_names=(prefix + profile.base_link,),
            secondary_type="subtree", secondary_name=prefix + profile.base_link,
            fields=("found", "force"), reduce="none")]
    for name, patterns in (("thigh_ground_touch", illegal.thigh_shape_patterns),
                           ("shank_ground_touch", illegal.shank_shape_patterns),
                           ("trunk_ground_touch", illegal.trunk_head_shape_patterns)):
        matches = tuple(prefix + shape for shape in shape_names if any(re.fullmatch(pattern, shape) for pattern in patterns))
        if patterns and not matches:
            raise ValueError(f"contact group {name!r} matched no collision shapes")
        if matches:
            specs.append(MuJoCoWarpContactSensorSpec(name=name, primary_type="geom", primary_names=matches,
                         fields=("found", "force"), reduce="none", **terrain))
    return tuple(specs)


def _authored_raycast_specs(prefix, profile, terrain_geom_groups, robot, scan_name):
    """Translate authored Gobot sensor frames once, below the task interface."""
    if not terrain_geom_groups:
        raise ValueError("compiled scene has no terrain geometry groups")

    def scanner(name):
        node = find_node_by_name(robot, name)
        if node is None:
            raise ValueError(f"authored sensor {name!r} is missing")
        if node.ray_alignment != gobot.RayAlignmentMode.Yaw or not np.allclose(node.ray_direction, (0., 0., -1.)):
            raise ValueError("velocity height observations require yaw-aligned downward scans")
        if node.pattern_mode == gobot.RayPatternMode.Grid:
            offsets = _grid_offsets(tuple(node.grid_size), float(node.grid_resolution))
        else:
            offsets = tuple(tuple(float(x) for x in value) for value in node.sample_offsets)
        if not offsets:
            raise ValueError(f"authored sensor {name!r} has no rays")
        return node, offsets

    scan, offsets = scanner(scan_name)
    if scan.parent.name != profile.base_link or not np.allclose(scan.position, 0.) or not np.allclose(scan.rotation_degrees, 0.):
        raise ValueError("velocity terrain scanner must use the unshifted base-link frame")
    feet = []
    for foot in profile.feet:
        height, samples = scanner(foot.height_sensor)
        contact = find_node_by_name(robot, foot.contact_sensor)
        if contact is None or height.parent.name != foot.link or contact.parent.name != foot.link:
            raise ValueError(f"foot {foot.name!r} sensors must belong to its configured link")
        if not np.allclose(height.position, contact.position) or not np.allclose(height.rotation_degrees, contact.rotation_degrees):
            raise ValueError(f"foot {foot.name!r} height and contact sensors must share a reference frame")
        feet.append((samples, float(height.max_distance)))
    if any(value != feet[0] for value in feet[1:]):
        raise ValueError("batched foot scanners must share their sampling pattern and range")
    return (
        MuJoCoWarpRaycastSensorSpec(name="terrain_scan", frame_type="body",
            frame_names=(prefix + profile.base_link,), local_offsets=offsets,
            alignment="yaw", max_distance=float(scan.max_distance), include_geom_groups=tuple(terrain_geom_groups)),
        MuJoCoWarpRaycastSensorSpec(name="foot_height_scan", frame_type="site",
            frame_names=tuple(prefix + f"{foot.link}_{foot.contact_sensor}_site" for foot in profile.feet), local_offsets=feet[0][0],
            alignment="yaw", max_distance=feet[0][1], include_geom_groups=tuple(terrain_geom_groups)),
    )


def _collision_names(root):
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type_name == "CollisionShape3D":
            result.append(node.name)
        stack.extend(node.children)
    return result


class WarpVelocityEnv:
    """MJLab-aligned locomotion task with simulation and task tensors on CUDA."""

    is_vector_env = True
    accepts_device_actions = True

    def __init__(
        self,
        cfg: VelocityTaskCfg,
        *,
        num_envs: int = 256,
        device: str = "cuda:0",
        seed: int = 42,
        max_episode_length: int | None = None,
        profile_step: bool = False,
        collect_step_extras: bool = True,
        context: gobot.AppContext | None = None,
        capture_graphs: bool = True,
        reuse_scene: bool = False,
    ) -> None:
        self._owns_scene = not reuse_scene
        self._reference_initializer = None
        self._fixed_terrain_assignments = False
        self.cfg_obj = cfg
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.device = str(device)
        self._torch_device = torch.device(self.device)
        if self._torch_device.type != "cuda":
            raise ValueError("WarpVelocityEnv requires a CUDA device; select mujoco-cpu for CPU training")
        if not torch.cuda.is_available():
            raise RuntimeError("MuJoCo Warp backend requested but Torch cannot access CUDA")

        self.seed = int(seed)
        self._generator = torch.Generator(device=self._torch_device)
        self._generator.manual_seed(self.seed)
        self.profile_step = bool(profile_step)
        self.collect_step_extras = bool(collect_step_extras)
        self.rsl_rl_include_reward_terms = self.collect_step_extras
        self.physics_dt = float(self.cfg_obj.physics_dt)
        self.decimation = int(self.cfg_obj.decimation)
        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = int(
            max_episode_length
            or math.ceil(float(self.cfg_obj.episode_length_s) / self.step_dt)
        )
        self.joint_names = tuple(self.cfg_obj.joint_names)
        self.num_actions = len(self.joint_names)
        self.default_joint_pos = np.asarray(self.cfg_obj.default_joint_pos, dtype=np.float32)
        self.action_scale = self._resolve_action_scale(self.cfg_obj.action_scale)
        self._default_joint_pos = torch.as_tensor(
            self.default_joint_pos, dtype=torch.float32, device=self._torch_device
        )
        self._action_scale = torch.as_tensor(
            self.action_scale, dtype=torch.float32, device=self._torch_device
        )
        self._reward_term_names = VELOCITY_REWARD_TERM_NAMES + (
            ("run_progress", "bound_gait")
            if self.cfg_obj.training_profile == "run"
            else ()
        )
        self._foot_count = len(self.cfg_obj.foot_names)
        self._gait_foot_indices = torch.tensor(
            self.cfg_obj.gait_foot_indices,
            dtype=torch.long,
            device=self._torch_device,
        )
        self._gait_joint_indices = torch.tensor(
            self.cfg_obj.gait_joint_indices,
            dtype=torch.long,
            device=self._torch_device,
        )

        self.project_path = Path(self.cfg_obj.project_path).resolve()
        self.context, self.robot, terrain = prepare_velocity_scene(self.cfg_obj, context=context, load_scene=not reuse_scene)
        self._terrain_config = terrain_generator_config(terrain)
        self._spawn_origins_np = terrain_spawn_origins(terrain)
        self._spawn_origins = torch.as_tensor(
            self._spawn_origins_np, dtype=torch.float32, device=self._torch_device
        )
        self._spawn_rows, self._spawn_cols = self._infer_spawn_grid_shape()
        artifact_mapping = self.context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        artifact = CompiledSceneArtifact.from_mapping(artifact_mapping)
        prefix = artifact.robot_prefix(self.cfg_obj.robot_name)
        self.compiled_artifact = artifact

        raycast_specs = _authored_raycast_specs(prefix, self.cfg_obj.robot, artifact.terrain_geom_groups,
            self.robot, self.cfg_obj.observations.height_scan_sensor)
        self._height_scan_dim = len(raycast_specs[0].local_offsets)
        self._scan_max_distance = raycast_specs[0].max_distance
        self.provider = MuJoCoWarpProvider(
            artifact,
            num_envs=self.num_envs,
            device=self.device,
            nconmax=self.cfg_obj.nconmax,
            njmax=self.cfg_obj.njmax,
            contact_sensor_maxmatch=500,
            ls_parallel=True,
            capture_graphs=bool(capture_graphs),
            overflow_check_interval=256,
            contact_sensors=_profile_contact_specs(prefix, self.cfg_obj.robot, self.cfg_obj.illegal_contact, _collision_names(self.robot)),
            raycast_sensors=raycast_specs,
        )
        self._arrays = self.provider.arrays
        self._state_link_ids = torch.tensor(self.provider.resolve_object_ids("body", tuple(prefix + name for name in cfg.state_link_names)), dtype=torch.long, device=self._torch_device) if cfg.state_link_names else None
        self._layout = self.provider.resolve_robot_layout(
            self.cfg_obj.robot_name,
            base_link=self.cfg_obj.base_link,
            joint_names=self.joint_names,
            sensor_names=(
                f"{self.cfg_obj.base_link}_{self.cfg_obj.robot.imu_sensor}_linear_velocity",
                f"{self.cfg_obj.base_link}_{self.cfg_obj.robot.imu_sensor}_angular_velocity",
                f"{self.cfg_obj.base_link}_{self.cfg_obj.robot.angular_momentum_sensor}_angular_momentum",
            ),
            site_names=tuple(
                f"{foot.link}_{foot.contact_sensor}_site" for foot in self.cfg_obj.robot.feet
            ),
            geom_names=tuple(shape for foot in self.cfg_obj.robot.feet for shape in foot.collision_shapes),
        )
        self._base_body_id = int(self._layout.base_body_id)
        self._torso_body_id = self.provider.resolve_object_ids("body", (prefix + self.cfg_obj.robot.torso_link,))[0]
        self._held_layout = None
        if self.cfg_obj.robot.held_joint_targets:
            self._held_layout = self.provider.resolve_robot_layout(self.cfg_obj.robot_name,
                base_link=self.cfg_obj.base_link, joint_names=tuple(self.cfg_obj.robot.held_joint_targets))
            self._held_targets = torch.tensor(tuple(self.cfg_obj.robot.held_joint_targets.values()), device=self._torch_device).expand(self.num_envs, -1).contiguous()

        self._joint_qpos_ids = torch.tensor(
            self._layout.joint_qpos_addresses,
            dtype=torch.long,
            device=self._torch_device,
        )
        self._joint_dof_ids = torch.tensor(
            self._layout.joint_dof_addresses,
            dtype=torch.long,
            device=self._torch_device,
        )
        self._site_ids = torch.tensor(
            self._layout.site_ids, dtype=torch.long, device=self._torch_device
        )
        self._site_body_ids = torch.tensor(
            self._layout.site_body_ids, dtype=torch.long, device=self._torch_device
        )
        self._foot_geom_ids = torch.tensor(
            self._layout.geom_ids, dtype=torch.long, device=self._torch_device
        )
        self._sensor_slices = tuple(
            slice(address, address + dimension)
            for address, dimension in zip(
                self._layout.sensor_addresses,
                self._layout.sensor_dimensions,
                strict=True,
            )
        )

        self.actor_obs_schema = velocity_actor_observation_schema(self.num_actions, self._height_scan_dim)
        self.critic_obs_schema = velocity_critic_observation_schema(
            self.num_actions, self._height_scan_dim, self._foot_count
        )
        self.observation_spec = self.actor_obs_schema
        self.critic_observation_spec = self.critic_obs_schema
        self.num_obs = self.actor_obs_schema.dim
        self.num_privileged_obs = self.critic_obs_schema.dim
        self.action_spec = self._make_action_spec()

        self._initialize_buffers()
        self._initialize_terrain_assignments()
        self._initialize_contacts()
        self._apply_startup_domain_randomization()
        self.task_runtime_metadata = self._make_task_runtime_metadata()
        self.task_runtime_info = {
            "mode": "torch",
            "compiled": True,
            "installed": True,
            "backend": "mujoco_warp_cuda_graph",
            "array_count": len(self._arrays),
            "array_names": tuple(sorted(self._arrays)),
            "metadata": self.task_runtime_metadata.metadata(),
        }
        self.cfg = self._make_public_cfg()
        self.resolved_sim_workers = 0
        self.extras: dict[str, Any] = {}
        self._state = BatchEnvState(
            obs={"actor": self._actor_obs, "critic": self._critic_obs},
            reward=self._reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info={"steps": self.episode_length_buf},
        )
        self.reset(seed=self.seed)

    @property
    def state(self) -> BatchEnvState:
        return self._state

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"actor": self.num_obs, "critic": self.num_privileged_obs}

    @property
    def command_b(self) -> torch.Tensor:
        return self._command

    def _resolve_action_scale(self, value: float | Mapping[str, float]) -> np.ndarray:
        if isinstance(value, Mapping):
            result = np.zeros((self.num_actions,), dtype=np.float32)
            matched = np.zeros((self.num_actions,), dtype=bool)
            for pattern, scale in value.items():
                expression = re.compile(str(pattern))
                for index, name in enumerate(self.joint_names):
                    if expression.fullmatch(name):
                        result[index] = float(scale)
                        matched[index] = True
            if not bool(np.all(matched)):
                missing = [name for index, name in enumerate(self.joint_names) if not matched[index]]
                raise ValueError(f"locomotion action scale has no value for joints {missing}")
            return result
        return np.full((self.num_actions,), float(value), dtype=np.float32)

    def _make_action_spec(self) -> ActionSpec:
        clip = self.cfg_obj.action_clip
        lower = -math.inf if clip is None else -float(clip)
        upper = math.inf if clip is None else float(clip)
        return ActionSpec(
            version=f"{self.actor_obs_schema.version}_action_v1",
            fields=tuple(SpecField(name, 1) for name in self.joint_names),
            lower=lower,
            upper=upper,
        )

    def _make_task_runtime_metadata(self) -> TaskRuntimeMetadata:
        return TaskRuntimeMetadata(
            name=self.cfg_obj.name,
            version=self.cfg_obj.task_version,
            obs_groups_spec=self.obs_groups_spec,
            reward_names=self._reward_term_names,
            backend="mujoco_warp_cuda_graph",
            cache_info={
                "scene_source": "jscn",
                "scene_path": self.cfg_obj.scene_path,
                "native_contact_detail": "mujoco_contact_sensor_substep_history",
                "domain_randomization_backend": "per_world_warp_model_fields",
                "raycast_backend": "mujoco_warp_bvh",
            },
        )

    def _make_public_cfg(self) -> dict[str, Any]:
        return {
            "name": self.cfg_obj.name,
            "source": "gobot.rl.locomotion.warp_env",
            "task": self.cfg_obj.name,
            "backend": "mujoco-warp",
            "obs_schema_version": self.actor_obs_schema.version,
            "obs_names": self.actor_obs_schema.names,
            "critic_obs_names": self.critic_obs_schema.names,
            "obs_spec": self.actor_obs_schema.metadata(),
            "critic_obs_spec": self.critic_obs_schema.metadata(),
            "action_spec": self.action_spec.metadata(),
            "task_runtime_metadata": self.task_runtime_metadata.metadata(),
            "task_runtime": dict(self.task_runtime_info),
            "robot": self.cfg_obj.robot_name,
            "scene_path": self.cfg_obj.scene_path,
            "project_path": str(self.project_path),
            "num_envs": self.num_envs,
            "num_obs": self.num_obs,
            "num_privileged_obs": self.num_privileged_obs,
            "num_actions": self.num_actions,
            "height_scan_dim": self._height_scan_dim,
            "decimation": self.decimation,
            "physics_dt": self.physics_dt,
            "step_dt": self.step_dt,
            "mujoco_solver_settings": dict(self.cfg_obj.mujoco_solver_settings),
            "action_clip": self.cfg_obj.action_clip,
            "domain_randomization": self.cfg_obj.domain_randomization.enabled,
            "push_enabled": self.cfg_obj.push_enabled,
            "terrain_curriculum": self.cfg_obj.terrain_curriculum,
            "training_profile": self.cfg_obj.training_profile,
            "curriculum_mode": self.cfg_obj.curriculum_mode,
            "run_environment_ratio": self.cfg_obj.command.rel_run_envs,
            "run_velocity_x": tuple(self.cfg_obj.command.run_velocity_x),
            "run_velocity_curriculum": tuple(
                (int(stage.step), stage.run_velocity_x)
                for stage in self.cfg_obj.run_command_curriculum
            ),
            "bound_gait_reward": self.cfg_obj.rewards.bound_gait,
            "run_progress_reward": self.cfg_obj.rewards.run_progress,
        }

    def _initialize_buffers(self) -> None:
        self._all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        self._actuator_ids = torch.tensor(
            self._layout.actuator_ids, dtype=torch.long, device=self._torch_device
        )
        qpos0 = self.provider.model_constant("qpos0").to(dtype=torch.float32)
        self._reset_qpos = qpos0.view(1, -1).expand(self.num_envs, -1).clone()
        self._reset_qvel = torch.zeros_like(self._arrays["qvel"])
        self._reset_ctrl = torch.zeros_like(self._arrays["ctrl"])
        self._reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._torch_device
        )
        self._reset_qpos[:, self._joint_qpos_ids] = self._default_joint_pos
        self._reset_ctrl[:, self._actuator_ids] = self._default_joint_pos

        self._action = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self._torch_device
        )
        self._previous_action = torch.zeros_like(self._action)
        self._previous_previous_action = torch.zeros_like(self._action)
        self._joint_targets = torch.zeros_like(self._action)
        self._encoder_bias = torch.zeros_like(self._action)

        self._command = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self._torch_device
        )
        self._command_world = torch.zeros_like(self._command)
        self._heading_target = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._torch_device
        )
        self._command_time_left = torch.zeros_like(self._heading_target)
        self._is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._torch_device
        )
        self._is_standing_env = torch.zeros_like(self._is_heading_env)
        self._is_world_env = torch.zeros_like(self._is_heading_env)
        self._is_forward_env = torch.zeros_like(self._is_heading_env)
        self._is_run_env = torch.zeros_like(self._is_heading_env)
        self._command_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        self._current_command_stage = 0

        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        self._episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._torch_device
        )
        self._commanded_displacement = torch.zeros((self.num_envs, 2), device=self._torch_device)
        self._episode_start_xy = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self._torch_device
        )
        self._reset_reasons = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        self.common_step_counter = 0
        self.profile_step_counter = 0
        self.step_counter = 0

        self._push_step_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        self._push_count = torch.zeros_like(self._push_step_left)

        self._terrain_scan_height = torch.zeros(
            (self.num_envs, self._height_scan_dim),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._foot_height = torch.zeros(
            (self.num_envs, self._foot_count),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._foot_peak_height = torch.zeros_like(self._foot_height)
        terrain_ray_count = self._height_scan_dim
        self._terrain_normal_indices = (
            torch.linspace(
                0,
                terrain_ray_count - 1,
                32,
                dtype=torch.float32,
                device=self._torch_device,
            ).long()
            if terrain_ray_count > 32
            else None
        )
        self._gravity_world = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self._torch_device
        )
        self._gravity_world[:, 2] = -1.0
        self._forward_body = torch.zeros_like(self._gravity_world)
        self._forward_body[:, 0] = 1.0
        self._terrain_up_world = torch.zeros_like(self._gravity_world)
        self._terrain_up_world[:, 2] = 1.0

        self._actor_obs_buffers = tuple(
            torch.zeros(
                (self.num_envs, self.num_obs),
                dtype=torch.float32,
                device=self._torch_device,
            )
            for _ in range(3 if self.cfg_obj.capture_transitions else 2)
        )
        self._critic_obs_buffers = tuple(
            torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                dtype=torch.float32,
                device=self._torch_device,
            )
            for _ in range(3 if self.cfg_obj.capture_transitions else 2)
        )
        self._observation_views = tuple(
            {"actor": actor, "critic": critic}
            for actor, critic in zip(
                self._actor_obs_buffers,
                self._critic_obs_buffers,
                strict=True,
            )
        )
        self._observation_buffer_index = 0
        self._actor_obs = self._actor_obs_buffers[self._observation_buffer_index]
        self._critic_obs = self._critic_obs_buffers[self._observation_buffer_index]
        self._reward_terms = torch.zeros(
            (self.num_envs, len(self._reward_term_names)),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._torch_device
        )
        self._terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._torch_device
        )
        self._truncated = torch.zeros_like(self._terminated)
        self._terrain_out = torch.zeros_like(self._terminated)
        self._last_step_profile_ms: dict[str, float] = {}

        reward_cfg = self.cfg_obj.rewards
        self._reward_weights = torch.tensor(
            [float(getattr(reward_cfg, name)) for name in self._reward_term_names],
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._pose_std_standing = self._pose_std(
            reward_cfg.pose_std_standing_hip_thigh,
            reward_cfg.pose_std_standing_calf,
            reward_cfg.pose_std_standing,
        )
        self._pose_std_walking = self._pose_std(
            reward_cfg.pose_std_walking_hip_thigh,
            reward_cfg.pose_std_walking_calf,
            reward_cfg.pose_std_walking,
        )
        self._pose_std_running = self._pose_std(
            reward_cfg.pose_std_running_hip_thigh,
            reward_cfg.pose_std_running_calf,
            reward_cfg.pose_std_running,
        )
        joint_range = self.provider.model_constant("jnt_range").to(dtype=torch.float32)
        joint_ids = torch.tensor(
            self._layout.joint_ids, dtype=torch.long, device=self._torch_device
        )
        self._joint_limits = joint_range.index_select(0, joint_ids)

    def _pose_std(self, hip_thigh, calf, mapping=None):
        return torch.as_tensor(posture_std(self.joint_names, mapping or {}, hip_thigh, calf), device=self._torch_device)

    def _infer_spawn_grid_shape(self) -> tuple[int, int]:
        unique_x = np.unique(np.round(self._spawn_origins_np[:, 0], 6))
        unique_y = np.unique(np.round(self._spawn_origins_np[:, 1], 6))
        if unique_x.size * unique_y.size == self._spawn_origins_np.shape[0]:
            return int(unique_x.size), int(unique_y.size)
        return int(self._spawn_origins_np.shape[0]), 1

    @staticmethod
    def _proportional_counts(count: int, proportions: Sequence[float]) -> np.ndarray:
        values = np.asarray(proportions, dtype=np.float64).reshape(-1)
        values = np.maximum(values, 0.0)
        if values.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if not np.any(values > 0.0):
            values[:] = 1.0
        values /= values.sum()
        if count >= values.size:
            counts = np.ones(values.size, dtype=np.int64)
            remaining = count - values.size
        else:
            counts = np.zeros(values.size, dtype=np.int64)
            remaining = count
        ideal = values * remaining
        floor = np.floor(ideal).astype(np.int64)
        counts += floor
        leftover = remaining - int(floor.sum())
        if leftover > 0:
            counts[np.argsort(-(ideal - floor))[:leftover]] += 1
        return counts

    def _initialize_terrain_assignments(self) -> None:
        rows, cols = self._spawn_rows, self._spawn_cols
        if rows * cols != self._spawn_origins.shape[0]:
            rows, cols = int(self._spawn_origins.shape[0]), 1
            self._spawn_rows, self._spawn_cols = rows, cols
        if not self.cfg_obj.terrain_curriculum:
            self._terrain_levels = torch.randint(
                0,
                rows,
                (self.num_envs,),
                dtype=torch.long,
                device=self._torch_device,
                generator=self._generator,
            )
            self._terrain_types = torch.randint(
                0,
                cols,
                (self.num_envs,),
                dtype=torch.long,
                device=self._torch_device,
                generator=self._generator,
            )
            self._env_origins = self._terrain_origin_for(self._all_env_ids)
            return
        max_init = rows - 1
        if self.cfg_obj.max_init_terrain_level is not None:
            max_init = min(int(self.cfg_obj.max_init_terrain_level), max_init)
        self._terrain_levels = torch.randint(
            0,
            max_init + 1,
            (self.num_envs,),
            dtype=torch.long,
            device=self._torch_device,
            generator=self._generator,
        )
        sub_terrains = self._terrain_config.get("sub_terrains", [])
        proportions = [
            float(entry.get("proportion", 1.0))
            for entry in sub_terrains[:cols]
            if isinstance(entry, Mapping)
        ]
        if len(proportions) == cols:
            counts = self._proportional_counts(self.num_envs, proportions)
            types = np.repeat(np.arange(cols, dtype=np.int64), counts)
        else:
            types = np.floor(
                np.arange(self.num_envs, dtype=np.float64) / (self.num_envs / cols)
            ).astype(np.int64)
        if types.size != self.num_envs:
            raise RuntimeError(
                f"terrain type assignment produced {types.size} entries for {self.num_envs} environments"
            )
        self._terrain_types = torch.as_tensor(
            types, dtype=torch.long, device=self._torch_device
        )
        self._env_origins = self._terrain_origin_for(self._all_env_ids)

    def _terrain_origin_for(self, env_ids: torch.Tensor) -> torch.Tensor:
        indices = self._terrain_levels[env_ids] * self._spawn_cols + self._terrain_types[env_ids]
        return self._spawn_origins.index_select(0, indices)

    def _initialize_contacts(self) -> None:
        self._raw_feet_contact = self.provider.contact_sensor("feet_ground_contact")
        self._foot_shape_index = torch.tensor([i for i, foot in enumerate(self.cfg_obj.robot.feet)
            for _ in foot.collision_shapes], dtype=torch.long, device=self._torch_device)
        self._feet_contact = {
            "found": torch.zeros((self.num_envs, self._foot_count), device=self._torch_device),
            "force": torch.zeros((self.num_envs, self._foot_count, 3), device=self._torch_device),
        }
        self._singleton_feet = all(len(foot.collision_shapes) == 1 for foot in self.cfg_obj.robot.feet)
        if self._singleton_feet:
            self._feet_contact = self._raw_feet_contact
        self._collision_contacts = {
            name: self.provider.contact_sensor(name)
            for name in (
                "self_collision",
                "thigh_ground_touch",
                "shank_ground_touch",
                "trunk_ground_touch",
            ) if name in self.provider.contact_sensors
        }
        self._current_air_time = torch.zeros(
            (self.num_envs, self._foot_count),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._last_air_time = torch.zeros_like(self._current_air_time)
        self._current_contact_time = torch.zeros_like(self._current_air_time)
        self._last_contact_time = torch.zeros_like(self._current_air_time)
        self._contact_last_time = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._torch_device
        )
        self._contact_histories = {
            name: torch.zeros(
                (
                    self.num_envs,
                    int(values["force"].shape[1]),
                    _CONTACT_HISTORY_LENGTH,
                    3,
                ),
                dtype=torch.float32,
                device=self._torch_device,
            )
            for name, values in self._collision_contacts.items()
        }
        self._contact_history_cursor = 0

    def _apply_startup_domain_randomization(self) -> None:
        dr = self.cfg_obj.domain_randomization
        unsupported: list[str] = []
        if dr.randomize_base_mass:
            unsupported.append("randomize_base_mass")
        if dr.randomize_kp:
            unsupported.append("randomize_kp")
        if dr.randomize_kd:
            unsupported.append("randomize_kd")
        if unsupported:
            raise NotImplementedError(
                "locomotion MuJoCo Warp backend does not implement disabled-by-default randomizers: "
                + ", ".join(unsupported)
            )
        if not dr.enabled:
            self._encoder_bias.zero_()
            return

        low, high = dr.encoder_bias_range
        self._encoder_bias.copy_(self._uniform(self._encoder_bias.shape, low, high))
        fields: list[str] = []
        if dr.random_com:
            fields.append("body_ipos")
        if dr.randomize_foot_friction:
            fields.append("geom_friction")
        if not fields:
            return
        self.provider.expand_model_fields(fields)

        if dr.random_com:
            body_ipos = self.provider.model_array("body_ipos")
            defaults = self.provider.model_constant("body_ipos").to(dtype=body_ipos.dtype)
            body_ipos.copy_(defaults.unsqueeze(0).expand_as(body_ipos))
            offsets = torch.zeros(
                (self.num_envs, 3), dtype=body_ipos.dtype, device=self._torch_device
            )
            offsets[:, 0] = self._uniform((self.num_envs,), *dr.com_offset_x)
            if dr.com_offset_y is not None:
                offsets[:, 1] = self._uniform((self.num_envs,), *dr.com_offset_y)
            if dr.com_offset_z is not None:
                offsets[:, 2] = self._uniform((self.num_envs,), *dr.com_offset_z)
            body_ipos[:, self._torso_body_id] += offsets
            self.provider.recompute_constants("set_const")

        if dr.randomize_foot_friction:
            friction = self.provider.model_array("geom_friction")
            defaults = self.provider.model_constant("geom_friction").to(dtype=friction.dtype)
            friction.copy_(defaults.unsqueeze(0).expand_as(friction))
            slide = self._uniform((self.num_envs,), *dr.foot_friction_slide_range)
            spin = self._log_uniform((self.num_envs,), *dr.foot_friction_spin_range)
            roll = self._log_uniform((self.num_envs,), *dr.foot_friction_roll_range)
            sampled = torch.stack((slide, spin, roll), dim=-1)
            friction[:, self._foot_geom_ids, :] = sampled[:, None, :]

    def _uniform(
        self,
        shape: Sequence[int] | torch.Size,
        low: float,
        high: float,
    ) -> torch.Tensor:
        result = torch.empty(tuple(int(value) for value in shape), device=self._torch_device)
        return result.uniform_(float(low), float(high), generator=self._generator)

    def _log_uniform(
        self,
        shape: Sequence[int] | torch.Size,
        low: float,
        high: float,
    ) -> torch.Tensor:
        if low <= 0.0 or high <= 0.0:
            return self._uniform(shape, low, high)
        return self._uniform(shape, math.log(low), math.log(high)).exp_()

    def reset_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        self.seed = int(seed)
        self._generator.manual_seed(self.seed)

    def init_state(self) -> BatchEnvState:
        return self._state

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self._state.obs

    def reset(
        self,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.reset_seed(seed)
        if env_ids is None:
            ids = self._all_env_ids
            self._episode_returns.zero_()
            self._push_count.zero_()
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._torch_device).reshape(-1)
        self._validate_env_ids(ids)
        if ids.numel() == 0:
            return {
                "actor": self._actor_obs[:0],
                "critic": self._critic_obs[:0],
            }, {}

        reasons = torch.zeros_like(ids)
        self._reset_idx(ids, reasons)
        self.provider.forward()
        self._compute_commands(dt=0.0)
        self.provider.sense()
        self._update_raycast_outputs()
        self._compute_observations()
        self._terminated[ids] = False
        self._truncated[ids] = False
        self._state.info = {"steps": self.episode_length_buf}
        return {
            "actor": self._actor_obs.index_select(0, ids),
            "critic": self._critic_obs.index_select(0, ids),
        }, {
            "reset_env_ids": ids,
            "reset_reason": self._reset_reasons.index_select(0, ids),
            "terrain_level": self._terrain_levels.index_select(0, ids).float(),
        }

    def reset_all(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        observations, _ = self.reset(seed=seed)
        return observations

    def step(self, actions: Any) -> BatchEnvState:
        profile_marks: list[tuple[str, torch.cuda.Event]] = []
        self._profile_mark(profile_marks, "start")
        action = self._prepare_actions(actions)
        self.common_step_counter += 1
        self.profile_step_counter += 1
        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(self._action)
        self._action.copy_(action)
        self._joint_targets.copy_(
            self._default_joint_pos.unsqueeze(0)
            + self._action_scale.unsqueeze(0) * self._action
            - self._encoder_bias
        )
        self.provider.set_joint_position_targets(self._layout, self._joint_targets)
        if self._held_layout is not None:
            self.provider.set_joint_position_targets(self._held_layout, self._held_targets)
        self._profile_mark(profile_marks, "action")

        if self.cfg_obj.curriculum_mode == "command_progress":
            command = torch.zeros_like(self._command)
            command[:, :2] = self._command[:, :2]
            self._commanded_displacement.add_(_quat_apply(self._arrays["xquat"][:, self._base_body_id], command)[:, :2] * self.step_dt)
        for _ in range(self.decimation):
            self.provider.step(nsteps=1)
            self._update_contact_state_substep()
        self._profile_mark(profile_marks, "physics")

        self.episode_length_buf.add_(1)
        self.step_counter += 1
        if self.cfg_obj.post_step_sensing:
            self.provider.forward()
            self.provider.sense()
            self._update_raycast_outputs()
            self._refresh_foot_contacts()
        self._compute_terminations()
        self._compute_rewards()
        self._profile_mark(profile_marks, "reward_termination")
        reward = self._reward.clone()
        terminated = self._terminated.clone()
        truncated = self._truncated.clone()
        reset_reason = torch.where(
            terminated,
            torch.ones_like(self._reset_reasons),
            torch.where(
                self._terrain_out,
                torch.full_like(self._reset_reasons, 3),
                torch.where(
                    truncated,
                    torch.full_like(self._reset_reasons, 2),
                    torch.zeros_like(self._reset_reasons),
                ),
            ),
        )
        done = terminated | truncated
        transition_state = transition_obs = None
        if self.cfg_obj.capture_transitions:
            transition_state = {key: value.clone() for key, value in self.locomotion_state().items()}
            # Before reset and command resampling: these belong to the executed action.
            self._compute_observations()
            transition_obs = {key: value.clone() for key, value in self._state.obs.items()}
        reset_ids = torch.nonzero(done, as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self._update_terrain_curriculum(reset_ids)
            self._reset_idx(reset_ids, reset_reason.index_select(0, reset_ids))
        self._profile_mark(profile_marks, "reset")

        self.provider.forward()
        self._compute_commands(dt=self.step_dt)
        self._apply_pushes()
        self.provider.sense()
        self._update_raycast_outputs()
        self._profile_mark(profile_marks, "forward_sense")
        self._compute_observations()
        self._profile_mark(profile_marks, "observation")

        info = self._make_step_info(reset_reason, truncated)
        if transition_obs is not None:
            info["transition_observation"] = transition_obs
            info["transition_state"] = transition_state
            info["final_observation"] = transition_obs
            info["_final_observation"] = done
        self._state.final_observation = transition_obs
        self._state.reward = reward
        self._state.terminated = terminated
        self._state.truncated = truncated
        self._state.info = info
        self.extras = info
        self._finish_step_profile(profile_marks)
        return self._state

    def _profile_mark(
        self,
        marks: list[tuple[str, torch.cuda.Event]],
        name: str,
    ) -> None:
        if not self.profile_step:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self._torch_device))
        marks.append((name, event))

    def _finish_step_profile(
        self,
        marks: list[tuple[str, torch.cuda.Event]],
    ) -> None:
        if not marks:
            self._last_step_profile_ms = {}
            return
        marks[-1][1].synchronize()
        profile: dict[str, float] = {}
        for (previous_name, previous), (name, current) in zip(
            marks, marks[1:], strict=True
        ):
            del previous_name
            profile[f"{name}_ms"] = float(previous.elapsed_time(current))
        profile["total_ms"] = float(marks[0][1].elapsed_time(marks[-1][1]))
        self._last_step_profile_ms = profile

    def _prepare_actions(self, actions: Any) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            value = actions.detach().to(device=self._torch_device, dtype=torch.float32)
        else:
            value = torch.as_tensor(actions, dtype=torch.float32, device=self._torch_device)
        expected = (self.num_envs, self.num_actions)
        if tuple(value.shape) != expected:
            raise ValueError(f"actions must have shape {expected}, got {tuple(value.shape)}")
        clip = self.cfg_obj.action_clip
        return value if clip is None else value.clamp(-float(clip), float(clip))

    def _validate_env_ids(self, env_ids: torch.Tensor) -> None:
        if env_ids.dtype != torch.long:
            raise TypeError("env ids must use torch.long")
        if env_ids.numel() == 0:
            return
        if bool(((env_ids < 0) | (env_ids >= self.num_envs)).any()):
            raise IndexError("reset env_ids contain an out-of-range environment id")

    def set_training_progress(self, common_steps: int) -> None:
        self.common_step_counter = max(0, int(common_steps))
        self._apply_command_curriculum()

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "backend": "mujoco-warp",
            "num_envs": self.num_envs,
            "common_step_counter": int(self.common_step_counter),
            "training_profile": self.cfg_obj.training_profile,
            "curriculum_mode": self.cfg_obj.curriculum_mode,
            "profile_step_counter": int(self.profile_step_counter),
            "terrain_curriculum": build_terrain_curriculum_state(
                self._terrain_levels.detach().cpu().numpy(),
                self._terrain_types.detach().cpu().numpy(),
                rows=self._spawn_rows,
                cols=self._spawn_cols,
            ),
            "rng_state": self._generator.get_state().cpu(),
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> dict[str, Any]:
        if state.get("curriculum_mode", "legacy") != self.cfg_obj.curriculum_mode:
            raise ValueError("checkpoint terrain curriculum mode does not match task configuration")
        version = int(state.get("version", 0))
        if version not in (0, 1, 2):
            raise RuntimeError(f"unsupported locomotion Warp training checkpoint version {version}")
        self.set_training_progress(int(state.get("common_step_counter", 0)))
        saved_profile = state.get("training_profile")
        profile_restored = saved_profile == self.cfg_obj.training_profile
        self.profile_step_counter = (
            max(0, int(state.get("profile_step_counter", 0)))
            if profile_restored
            else 0
        )
        self._apply_command_curriculum()
        terrain_state = state.get("terrain_curriculum")
        if not isinstance(terrain_state, Mapping):
            return {
                "common_step_counter": int(self.common_step_counter),
                "profile_curriculum": "exact" if profile_restored else "restarted",
                "terrain_curriculum": "legacy_initial_levels",
            }

        levels, terrain_types, exact = restore_terrain_curriculum_assignments(
            terrain_state,
            self._terrain_levels.detach().cpu().numpy(),
            self._terrain_types.detach().cpu().numpy(),
            rows=self._spawn_rows,
            cols=self._spawn_cols,
        )
        self._terrain_levels.copy_(torch.as_tensor(levels, device=self._torch_device))
        self._terrain_types.copy_(torch.as_tensor(terrain_types, device=self._torch_device))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._generator.set_state(torch.as_tensor(rng_state, dtype=torch.uint8, device="cpu"))
        self._env_origins.copy_(self._terrain_origin_for(self._all_env_ids))
        self.reset()
        return {
            "common_step_counter": int(self.common_step_counter),
            "profile_curriculum": "exact" if profile_restored else "restarted",
            "terrain_curriculum": "exact" if exact else "resampled_for_num_envs",
            "mean_terrain_level": float(self._terrain_levels.float().mean()),
        }

    @property
    def terrain_origins(self):
        return self._spawn_origins.clone()

    @property
    def terrain_grid_shape(self):
        return self._spawn_rows, self._spawn_cols

    @property
    def terrain_type_names(self):
        return tuple(entry["name"] for entry in self._terrain_config["sub_terrains"])

    def set_command(self, command):
        value = torch.as_tensor(command, dtype=torch.float32, device=self._torch_device)
        if value.shape not in ((3,), (self.num_envs, 3)) or not bool(torch.isfinite(value).all()):
            raise ValueError("command must be finite with shape (3,) or (num_envs, 3)")
        self._command.copy_(value)
        self._compute_observations()

    def locomotion_state(self):
        linear, angular = self._root_velocity_body()
        result = {"base_position": self._arrays["xpos"][:, self._base_body_id],
                "base_quaternion": self._arrays["xquat"][:, self._base_body_id],
                "base_linear_velocity_body": linear, "base_angular_velocity_body": angular,
                "joint_position": self._arrays["qpos"].index_select(1, self._joint_qpos_ids),
                "joint_velocity": self._arrays["qvel"].index_select(1, self._joint_dof_ids),
                "foot_contact": self._feet_contact["found"] > 0,
                "foot_velocity": self._foot_velocity_world(),
                "angular_momentum": self._arrays["sensordata"][:, self._sensor_slices[2]],
                "foot_position": self._arrays["site_xpos"].index_select(1, self._site_ids),
                "foot_force": self._feet_contact["force"],
                "base_clearance": self._terrain_scan_height[:, self._height_scan_dim // 2],
                "command": self._command,
                "terrain_level": self._terrain_levels,
                "terrain_type": self._terrain_types}
        if self._state_link_ids is not None:
            result["link_position"] = self._arrays["xpos"].index_select(1, self._state_link_ids)
        return result

    @property
    def terrain_assignments(self):
        return self._terrain_levels.clone(), self._terrain_types.clone()

    def set_terrain_assignments(self, levels, types, *, fixed=False):
        levels = torch.as_tensor(levels, device=self._torch_device, dtype=torch.long)
        types = torch.as_tensor(types, device=self._torch_device, dtype=torch.long)
        if levels.shape != (self.num_envs,) or types.shape != levels.shape or bool(((levels < 0) | (levels >= self._spawn_rows) | (types < 0) | (types >= self._spawn_cols)).any()):
            raise ValueError("terrain assignments must contain one valid cell per environment")
        self._fixed_terrain_assignments = bool(fixed)
        self._terrain_levels.copy_(levels)
        self._terrain_types.copy_(types)
        self._env_origins.copy_(self._terrain_origin_for(self._all_env_ids))

    def close(self) -> None:
        provider = getattr(self, "provider", None)
        if provider is not None:
            provider.close()
        if self._owns_scene:
            self.context.clear_world()
            self.context.clear_scene()

    def set_reference_initializer(self, initializer):
        self._reference_initializer = initializer

    def _reset_idx(self, env_ids: torch.Tensor, reasons: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._apply_command_curriculum()
        count = int(env_ids.numel())
        if not self.cfg_obj.terrain_curriculum and not self._fixed_terrain_assignments:
            self._terrain_levels[env_ids] = torch.randint(
                0,
                self._spawn_rows,
                (count,),
                dtype=torch.long,
                device=self._torch_device,
                generator=self._generator,
            )
            self._terrain_types[env_ids] = torch.randint(
                0,
                self._spawn_cols,
                (count,),
                dtype=torch.long,
                device=self._torch_device,
                generator=self._generator,
            )
        origins = self._terrain_origin_for(env_ids)
        self._env_origins[env_ids] = origins
        jitter = self._uniform((count, 2), -self.cfg_obj.spawn_jitter, self.cfg_obj.spawn_jitter)
        z_offset = self._uniform((count,), *self.cfg_obj.reset_z_range)
        positions = origins.clone()
        positions[:, :2] += jitter
        positions[:, :2] += torch.tensor(self.cfg_obj.spawn_offset_xy, device=self._torch_device)
        positions[:, 2] += float(self.cfg_obj.base_clearance) + z_offset
        if self.cfg_obj.randomize_reset_yaw:
            yaw = self._uniform((count,), -3.14, 3.14)
            orientation = _quat_from_yaw(yaw)
        else:
            orientation = torch.zeros(
                (count, 4), dtype=torch.float32, device=self._torch_device
            )
            orientation[:, 0] = 1.0

        align_surface = any(self.cfg_obj.spawn_offset_xy)
        if align_surface:
            positions[:, 2] += 1.5
        self._reset_qpos[env_ids, :3] = positions
        self._reset_qpos[env_ids, 3:7] = orientation
        self._reset_qpos[env_ids[:, None], self._joint_qpos_ids] = self._default_joint_pos
        self._reset_qvel[env_ids] = 0.0
        self._reset_ctrl[env_ids] = 0.0
        self._reset_ctrl[env_ids[:, None], self._actuator_ids] = self._default_joint_pos
        if self._held_layout is not None:
            self._reset_qpos[env_ids[:, None], list(self._held_layout.joint_qpos_addresses)] = self._held_targets[env_ids]
            self._reset_ctrl[env_ids[:, None], list(self._held_layout.actuator_ids)] = self._held_targets[env_ids]
        reference_ids = None
        if self._reference_initializer is not None:
            mask, reference_root, reference_joints = self._reference_initializer.sample(count, self._generator)
            reference_ids = env_ids[mask]
            q = reference_joints[mask, :self.num_actions]
            self._reset_qpos[reference_ids, 2] = origins[mask, 2] + reference_root[mask, 2]
            self._reset_qpos[reference_ids, 3:7] = reference_root[mask, 3:7]
            self._reset_qpos[reference_ids[:, None], self._joint_qpos_ids] = q
            self._reset_qvel[reference_ids, :3] = reference_root[mask, 7:10]
            self._reset_qvel[reference_ids, 3:6] = _quat_apply_inverse(reference_root[mask, 3:7], reference_root[mask, 10:13])
            self._reset_qvel[reference_ids[:, None], self._joint_dof_ids] = reference_joints[mask, self.num_actions:]
            self._reset_ctrl[reference_ids[:, None], self._actuator_ids] = q
        self._reset_mask.zero_()
        self._reset_mask[env_ids] = True
        self.provider.reset(
            self._reset_mask,
            qpos=self._reset_qpos,
            qvel=self._reset_qvel,
            ctrl=self._reset_ctrl,
            forward=False,
        )

        if align_surface:
            self.provider.forward()
            self.provider.sense()
            scan = self.provider.raycast_sensor("terrain_scan")
            center = self._height_scan_dim // 2
            if bool((scan["distances"][env_ids, center] < 0).any()):
                raise RuntimeError("evaluation spawn is outside terrain collision geometry")
            positions[:, 2] = scan["hit_pos_w"][env_ids, center, 2] + self.cfg_obj.base_clearance + z_offset
            self._reset_qpos[env_ids, 2] = positions[:, 2]
            self.provider.reset(self._reset_mask, qpos=self._reset_qpos, qvel=self._reset_qvel,
                                ctrl=self._reset_ctrl, forward=False)

        self._episode_start_xy[env_ids] = positions[:, :2] if self.cfg_obj.curriculum_mode == "command_progress" else origins[:, :2]
        self._commanded_displacement[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self._episode_returns[env_ids] = 0.0
        self._reset_reasons[env_ids] = reasons
        self._previous_previous_action[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self._joint_targets[env_ids] = self._default_joint_pos
        if reference_ids is not None:
            target = self._reset_qpos[reference_ids[:, None], self._joint_qpos_ids]
            self._joint_targets[reference_ids] = target
            self._action[reference_ids] = (target - self._default_joint_pos) / self._action_scale
            self._previous_action[reference_ids] = self._action[reference_ids]
        self._foot_peak_height[env_ids] = 0.0
        self._reset_contact_state(env_ids)
        self._reset_commands(env_ids)
        self._reset_push_timers(env_ids)

    def _reset_contact_state(self, env_ids: torch.Tensor) -> None:
        self._current_air_time[env_ids] = 0.0
        self._last_air_time[env_ids] = 0.0
        self._current_contact_time[env_ids] = 0.0
        self._last_contact_time[env_ids] = 0.0
        self._contact_last_time[env_ids] = self._arrays["time"][env_ids]
        for history in self._contact_histories.values():
            history[env_ids] = 0.0

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        if self._fixed_terrain_assignments or not self.cfg_obj.terrain_curriculum or env_ids.numel() == 0:
            return
        position = self._arrays["xpos"][env_ids, self._base_body_id, :2]
        distance = torch.linalg.vector_norm(position - self._env_origins[env_ids, :2], dim=1)
        patch_size = self._terrain_patch_size()
        move_up = distance > patch_size[0] * 0.5
        command_distance = (
            torch.linalg.vector_norm(self._command[env_ids, :2], dim=1)
            * float(self.cfg_obj.episode_length_s)
            * 0.5
        )
        move_down = (distance < command_distance) & ~move_up
        if self.cfg_obj.curriculum_mode == "command_progress":
            distance = torch.linalg.vector_norm(position - self._episode_start_xy[env_ids], dim=1)
            time_left = ((self.max_episode_length - self.episode_length_buf[env_ids]) * self.step_dt).clamp(min=0.)
            required = torch.linalg.vector_norm(self._commanded_displacement[env_ids], dim=1) + torch.linalg.vector_norm(self._command[env_ids, :2], dim=1) * time_left
            move_up, move_down = progress_masks(distance, required, patch_size[0])
            if self.common_step_counter == 0:
                move_up.zero_()
                move_down.zero_()
        move_up &= ~self._terminated[env_ids]
        levels = self._terrain_levels[env_ids] + move_up.long() - move_down.long()
        random_levels = torch.randint(
            0,
            self._spawn_rows,
            (int(env_ids.numel()),),
            dtype=torch.long,
            device=self._torch_device,
            generator=self._generator,
        )
        levels = torch.where(levels >= self._spawn_rows, random_levels, levels.clamp(min=0))
        if self.cfg_obj.max_terrain_level is not None:
            levels.clamp_(max=self.cfg_obj.max_terrain_level)
        self._terrain_levels[env_ids] = levels
        self._env_origins[env_ids] = self._terrain_origin_for(env_ids)

    def _terrain_patch_size(self) -> tuple[float, float]:
        value = np.asarray(self._terrain_config.get("patch_size", (8.0, 8.0))).reshape(-1)
        if value.size != 2:
            raise RuntimeError("Gobot terrain generator has an invalid patch_size")
        return float(value[0]), float(value[1])

    def _apply_command_curriculum(self) -> None:
        ranges = self.cfg_obj.command.ranges
        stage_index = 0
        for index, stage in enumerate(self.cfg_obj.command_curriculum):
            if self.common_step_counter < int(stage.step):
                continue
            stage_index = index
            if stage.lin_vel_x is not None:
                ranges.lin_vel_x = stage.lin_vel_x
            if stage.lin_vel_y is not None:
                ranges.lin_vel_y = stage.lin_vel_y
            if stage.ang_vel_z is not None:
                ranges.ang_vel_z = stage.ang_vel_z
        run_stage = 0
        for index, stage in enumerate(self.cfg_obj.run_command_curriculum):
            if self.profile_step_counter < int(stage.step):
                continue
            run_stage = index
            if stage.run_velocity_x is not None:
                self.cfg_obj.command.run_velocity_x = stage.run_velocity_x
        self._current_command_stage = stage_index
        self._current_run_command_stage = run_stage

    def _reset_commands(self, env_ids: torch.Tensor) -> None:
        self._command_counter[env_ids] = 0
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        cfg = self.cfg_obj.command
        count = int(env_ids.numel())
        self._command_time_left[env_ids] = self._uniform(
            (count,), *cfg.resampling_time_range
        )
        self._command[env_ids, 0] = self._uniform((count,), *cfg.ranges.lin_vel_x)
        self._command[env_ids, 1] = self._uniform((count,), *cfg.ranges.lin_vel_y)
        self._command[env_ids, 2] = self._uniform((count,), *cfg.ranges.ang_vel_z)
        if cfg.heading_command:
            if cfg.ranges.heading is None:
                raise ValueError("heading command requires a heading range")
            self._heading_target[env_ids] = self._uniform((count,), *cfg.ranges.heading)
            self._is_heading_env[env_ids] = (
                self._uniform((count,), 0.0, 1.0) <= float(cfg.rel_heading_envs)
            )
        self._is_standing_env[env_ids] = (
            self._uniform((count,), 0.0, 1.0) <= float(cfg.rel_standing_envs)
        )
        self._is_world_env[env_ids] = (
            self._uniform((count,), 0.0, 1.0) <= float(cfg.rel_world_envs)
        )
        self._command_world[env_ids] = self._command[env_ids]
        self._is_forward_env[env_ids] = (
            self._uniform((count,), 0.0, 1.0) <= float(cfg.rel_forward_envs)
        )
        forward_ids = env_ids[self._is_forward_env[env_ids]]
        if forward_ids.numel() > 0:
            self._command[forward_ids, 0] = self._command[forward_ids, 0].abs().clamp(min=0.3)
            self._command[forward_ids, 1:] = 0.0
        self._is_run_env[env_ids] = (
            self._uniform((count,), 0.0, 1.0) <= float(cfg.rel_run_envs)
        )
        run_ids = env_ids[self._is_run_env[env_ids]]
        if run_ids.numel() > 0:
            run_low, run_high = sorted(float(value) for value in cfg.run_velocity_x)
            self._command[run_ids, 0] = self._uniform(
                (int(run_ids.numel()),), max(0.0, run_low), max(0.0, run_high)
            )
            self._command[run_ids, 1:] = 0.0
            self._command_world[run_ids] = self._command[run_ids]
            self._is_heading_env[run_ids] = False
            self._is_world_env[run_ids] = False
        if self.cfg_obj.running_terrain_types:
            allowed = torch.zeros(count, dtype=torch.bool, device=self._torch_device)
            for kind in self.cfg_obj.running_terrain_types:
                allowed |= self._terrain_types[env_ids] == kind
            allowed &= (self._terrain_types[env_ids] == 0) | (self._terrain_levels[env_ids] <= self.cfg_obj.max_running_terrain_level)
            walking_ids = env_ids[~allowed]
            self._command[walking_ids, 0] = self._command[walking_ids, 0].clamp(max=1.2)
            self._is_run_env[walking_ids] = False
        self._command_counter[env_ids] += 1

    def _compute_commands(self, *, dt: float) -> None:
        self._command_time_left.sub_(float(dt))
        resample_ids = torch.nonzero(self._command_time_left <= 0.0, as_tuple=False).flatten()
        self._resample_commands(resample_ids)
        cfg = self.cfg_obj.command
        if cfg.heading_command:
            heading = self._heading()
            error = _wrap_to_pi(self._heading_target - heading)
            heading_ids = torch.nonzero(self._is_heading_env, as_tuple=False).flatten()
            if heading_ids.numel() > 0:
                self._command[heading_ids, 2] = (
                    float(cfg.heading_control_stiffness) * error[heading_ids]
                ).clamp(*cfg.ranges.ang_vel_z)
        world_ids = torch.nonzero(self._is_world_env, as_tuple=False).flatten()
        if world_ids.numel() > 0:
            heading = self._heading()[world_ids]
            cosine = torch.cos(heading)
            sine = torch.sin(heading)
            vx = self._command_world[world_ids, 0]
            vy = self._command_world[world_ids, 1]
            self._command[world_ids, 0] = cosine * vx + sine * vy
            self._command[world_ids, 1] = -sine * vx + cosine * vy
        standing_ids = torch.nonzero(self._is_standing_env, as_tuple=False).flatten()
        if standing_ids.numel() > 0:
            self._command[standing_ids] = 0.0
            self._command_world[standing_ids] = 0.0
            self._is_run_env[standing_ids] = False

    def _heading(self) -> torch.Tensor:
        quaternion = self._arrays["xquat"][:, self._base_body_id]
        forward_world = _quat_apply(quaternion, self._forward_body)
        return torch.atan2(forward_world[:, 1], forward_world[:, 0])

    def _reset_push_timers(self, env_ids: torch.Tensor) -> None:
        if not self.cfg_obj.push_enabled:
            self._push_step_left[env_ids] = torch.iinfo(torch.long).max
            return
        low, high = self.cfg_obj.push_interval_range_s
        seconds = self._uniform((int(env_ids.numel()),), low, high)
        steps = torch.ceil(seconds / max(self.step_dt, 1.0e-9)).long().clamp(min=1)
        self._push_step_left[env_ids] = steps

    def _apply_pushes(self) -> None:
        if not self.cfg_obj.push_enabled:
            return
        self._push_step_left.sub_(1)
        env_ids = torch.nonzero(self._push_step_left <= 0, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        if self.cfg_obj.push_mode != "velocity":
            raise NotImplementedError("MJLab-aligned locomotion Warp pushes use velocity mode")
        linear, angular = self._root_link_velocity_world()
        ranges = self.cfg_obj.push_velocity_ranges
        count = int(env_ids.numel())
        for column, name in enumerate(("x", "y", "z")):
            linear[env_ids, column] += self._uniform((count,), *ranges.get(name, (0.0, 0.0)))
        for column, name in enumerate(("roll", "pitch", "yaw")):
            angular[env_ids, column] += self._uniform((count,), *ranges.get(name, (0.0, 0.0)))
        quaternion = self._arrays["qpos"][env_ids, 3:7]
        self._arrays["qvel"][env_ids, :3] = linear[env_ids]
        self._arrays["qvel"][env_ids, 3:6] = _quat_apply_inverse(
            quaternion, angular[env_ids]
        )
        self._push_count[env_ids] += 1
        self._reset_push_timers(env_ids)

    def _refresh_foot_contacts(self):
        if self._singleton_feet:
            return
        self._feet_contact["found"].zero_().scatter_add_(1, self._foot_shape_index.expand(self.num_envs, -1), self._raw_feet_contact["found"].float())
        self._feet_contact["force"].zero_().scatter_add_(1, self._foot_shape_index[None, :, None].expand(self.num_envs, -1, 3), self._raw_feet_contact["force"])

    def _update_contact_state_substep(self) -> None:
        current_time = self._arrays["time"]
        elapsed = (current_time - self._contact_last_time).unsqueeze(-1)
        self._refresh_foot_contacts()
        in_contact = self._feet_contact["found"] > 0
        first_contact = (self._current_air_time > 0.0) & in_contact
        first_detached = (self._current_contact_time > 0.0) & ~in_contact
        self._last_air_time.copy_(
            torch.where(
                first_contact,
                self._current_air_time + elapsed,
                self._last_air_time,
            )
        )
        self._current_air_time.copy_(
            torch.where(
                ~in_contact,
                self._current_air_time + elapsed,
                torch.zeros_like(self._current_air_time),
            )
        )
        self._last_contact_time.copy_(
            torch.where(
                first_detached,
                self._current_contact_time + elapsed,
                self._last_contact_time,
            )
        )
        self._current_contact_time.copy_(
            torch.where(
                in_contact,
                self._current_contact_time + elapsed,
                torch.zeros_like(self._current_contact_time),
            )
        )
        self._contact_last_time.copy_(current_time)
        cursor = self._contact_history_cursor
        for name, values in self._collision_contacts.items():
            self._contact_histories[name][:, :, cursor, :].copy_(values["force"])
        self._contact_history_cursor = (cursor + 1) % _CONTACT_HISTORY_LENGTH

    def _root_link_velocity_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._arrays["xpos"][:, self._base_body_id]
        subtree_com = self._arrays["subtree_com"][:, self._base_body_id]
        cvel = self._arrays["cvel"][:, self._base_body_id]
        angular = cvel[:, :3]
        linear = cvel[:, 3:6] - torch.cross(
            angular, subtree_com - position, dim=-1
        )
        return linear, angular

    def _root_velocity_body(self) -> tuple[torch.Tensor, torch.Tensor]:
        linear_world, angular_world = self._root_link_velocity_world()
        quaternion = self._arrays["xquat"][:, self._base_body_id]
        return (
            _quat_apply_inverse(quaternion, linear_world),
            _quat_apply_inverse(quaternion, angular_world),
        )

    def _foot_velocity_world(self) -> torch.Tensor:
        position = self._arrays["site_xpos"].index_select(1, self._site_ids)
        cvel = self._arrays["cvel"].index_select(1, self._site_body_ids)
        subtree_com = self._arrays["subtree_com"][:, self._base_body_id].unsqueeze(1)
        angular = cvel[..., :3]
        return cvel[..., 3:6] - torch.cross(
            angular, subtree_com - position, dim=-1
        )

    def _update_raycast_outputs(self) -> None:
        terrain = self.provider.raycast_sensor("terrain_scan")
        frame_z = terrain["frame_pos_w"][:, 0, 2:3]
        hit_z = terrain["hit_pos_w"][..., 2]
        heights = frame_z - hit_z
        self._terrain_scan_height.copy_(
            torch.where(
                terrain["distances"] < 0.0,
                torch.full_like(heights, self._scan_max_distance),
                heights,
            )
        )

        feet = self.provider.raycast_sensor("foot_height_scan")
        frame_count = int(feet["num_frames"])
        rays_per_frame = int(feet["num_rays_per_frame"])
        frame_z = feet["frame_pos_w"][..., 2]
        hit_z = feet["hit_pos_w"][..., 2].view(
            self.num_envs, frame_count, rays_per_frame
        )
        heights = frame_z.unsqueeze(-1) - hit_z
        distances = feet["distances"].view(
            self.num_envs, frame_count, rays_per_frame
        )
        normal_z = feet["normals_w"][..., 2].view(
            self.num_envs, frame_count, rays_per_frame
        )
        backface = (distances >= 0.0) & (normal_z < 0.0)
        heights = torch.where(backface, torch.zeros_like(heights), heights)
        miss = distances < 0.0
        all_miss = miss.all(dim=-1, keepdim=True).expand_as(miss)
        fallback = frame_z.unsqueeze(-1).clamp(0.0, 1.0).expand_as(heights)
        miss_value = torch.where(all_miss, fallback, torch.ones_like(heights))
        heights = torch.where(miss, miss_value, heights)
        self._foot_height.copy_(heights.min(dim=-1).values)

    def _terrain_normal_world(self) -> torch.Tensor:
        sensor = self.provider.raycast_sensor("terrain_scan")
        points = sensor["hit_pos_w"]
        valid = sensor["distances"] >= 0.0
        if self._terrain_normal_indices is not None:
            points = points.index_select(1, self._terrain_normal_indices)
            valid = valid.index_select(1, self._terrain_normal_indices)
        count = valid.sum(dim=1)
        weights = valid.float().unsqueeze(-1)
        centroid = (points * weights).sum(dim=1) / count.clamp(min=1).float().unsqueeze(-1)
        centered = (points - centroid.unsqueeze(1)) * weights
        covariance = torch.einsum("bni,bnj->bij", centered, centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        normal = eigenvectors[:, :, 0]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
        normal = torch.where((normal[:, 2] < 0.0).unsqueeze(-1), -normal, normal)
        epsilon = torch.finfo(eigenvalues.dtype).eps
        plane_like = eigenvalues[:, 0] / eigenvalues[:, 1].clamp(min=epsilon) < 0.1
        has_spread = eigenvalues[:, 1] > eigenvalues[:, 2].clamp(min=epsilon) * 1.0e-6
        reliable = (count >= 3) & plane_like & has_spread
        return torch.where(reliable.unsqueeze(-1), normal, self._terrain_up_world)

    def _compute_terminations(self) -> None:
        illegal_cfg = self.cfg_obj.illegal_contact
        if illegal_cfg.enabled and illegal_cfg.terminate_on_thigh:
            force_history = self._contact_histories["thigh_ground_touch"]
            illegal = (
                torch.linalg.vector_norm(force_history, dim=-1)
                > float(illegal_cfg.ground_force_threshold)
            ).any(dim=-1).any(dim=-1)
        else:
            illegal = torch.zeros_like(self._terminated)
        quat = self._arrays["xquat"][:, self._base_body_id]
        gravity_z = -(1. - 2. * (quat[:, 1].square() + quat[:, 2].square()))
        trunk = torch.zeros_like(illegal)
        if "trunk_ground_touch" in self._contact_histories:
            force = self._contact_histories["trunk_ground_touch"]
            trunk = (torch.linalg.vector_norm(force, dim=-1) > illegal_cfg.ground_force_threshold).flatten(1).any(1)
        illegal = locomotion_failure_mask(gravity_z, self._terrain_scan_height[:, self._height_scan_dim // 2],
            illegal, trunk, fall_angle=self.cfg_obj.fall_angle, min_clearance=self.cfg_obj.min_base_clearance,
            terminate_on_trunk=illegal_cfg.terminate_on_trunk_contact)
        self._terminated.copy_(illegal)

        if self.cfg_obj.terrain_out_of_bounds:
            position = self._arrays["xpos"][:, self._base_body_id, :2]
            patch_x, patch_y = self._terrain_patch_size()
            border = float(self._terrain_config.get("border_width", 0.0))
            margin = float(self.cfg_obj.terrain_distance_buffer)
            limit_x = max(0.0, 0.5 * self._spawn_rows * patch_x + border - margin)
            limit_y = max(0.0, 0.5 * self._spawn_cols * patch_y + border - margin)
            self._terrain_out.copy_(
                (position[:, 0].abs() > limit_x) | (position[:, 1].abs() > limit_y)
            )
        else:
            self._terrain_out.zero_()
        self._truncated.copy_(
            ((self.episode_length_buf >= self.max_episode_length) | self._terrain_out) & ~self._terminated
        )

    def _compute_rewards(self) -> None:
        reward_cfg = self.cfg_obj.rewards
        linear_body, angular_body = self._root_velocity_body()
        _, angular_world = self._root_link_velocity_world()
        quaternion = self._arrays["xquat"][:, self._base_body_id]
        joint_position = self._arrays["qpos"].index_select(1, self._joint_qpos_ids)
        difference = joint_position - self._default_joint_pos.unsqueeze(0)
        command_speed = torch.linalg.vector_norm(self._command[:, :2], dim=1) + self._command[:, 2].abs()
        active = (command_speed > float(reward_cfg.command_threshold)).float()

        linear_error = (
            (self._command[:, :2] - linear_body[:, :2]).square().sum(dim=1)
            + linear_body[:, 2].square()
        )
        angular_error = (
            (self._command[:, 2] - angular_body[:, 2]).square()
            + angular_body[:, :2].square().sum(dim=1)
        )
        terrain_normal = self._terrain_normal_world()
        terrain_up_body = _quat_apply_inverse(self._arrays["xquat"][:, self._torso_body_id], terrain_normal)
        upright_error = terrain_up_body[:, :2].square().sum(dim=1)
        pose_std = torch.where(
            (command_speed < float(reward_cfg.pose_walking_threshold)).unsqueeze(-1),
            self._pose_std_standing,
            torch.where(
                (command_speed < float(reward_cfg.pose_running_threshold)).unsqueeze(-1),
                self._pose_std_walking,
                self._pose_std_running,
            ),
        )
        pose_error = (difference.square() / pose_std.square()).mean(dim=1)

        factor = float(reward_cfg.soft_joint_pos_limit_factor)
        midpoint = 0.5 * (self._joint_limits[:, 0] + self._joint_limits[:, 1])
        half_range = 0.5 * (self._joint_limits[:, 1] - self._joint_limits[:, 0]) * factor
        soft_lower = midpoint - half_range
        soft_upper = midpoint + half_range
        joint_limit_cost = (
            (soft_lower.unsqueeze(0) - joint_position).clamp(min=0.0)
            + (joint_position - soft_upper.unsqueeze(0)).clamp(min=0.0)
        ).sum(dim=1)
        action_rate = (self._action - self._previous_action).square().sum(dim=1)

        foot_found = self._feet_contact["found"] > 0
        foot_force = self._feet_contact["force"]
        foot_velocity = self._foot_velocity_world()
        foot_speed_xy = torch.linalg.vector_norm(foot_velocity[..., :2], dim=-1)
        foot_clearance = (
            (self._foot_height - float(reward_cfg.foot_clearance_target_height)).abs()
            * foot_speed_xy
        ).sum(dim=1) * active
        self._foot_peak_height.copy_(
            torch.where(
                ~foot_found,
                torch.maximum(self._foot_peak_height, self._foot_height),
                self._foot_peak_height,
            )
        )
        first_contact = (self._current_contact_time > 0.0) & (
            self._current_contact_time < self.step_dt + 1.0e-6
        )
        swing_error = self._foot_peak_height / max(
            float(reward_cfg.foot_clearance_target_height), 1.0e-6
        ) - 1.0
        swing_height = (swing_error.square() * first_contact.float()).sum(dim=1) * active
        self._foot_peak_height.copy_(
            torch.where(first_contact, torch.zeros_like(self._foot_peak_height), self._foot_peak_height)
        )
        foot_slip = (
            foot_speed_xy.square() * foot_found.float()
        ).sum(dim=1) * active
        landing = (
            torch.linalg.vector_norm(foot_force, dim=-1) * first_contact.float()
        ).sum(dim=1) * active
        air_time = (
            (self._current_air_time > 0.05)
            & (self._current_air_time < 0.5)
        ).float().sum(dim=1)
        air_time *= (command_speed > 0.5).float()

        collision_costs = {
            name: (
                torch.linalg.vector_norm(history, dim=-1)
                > float(self.cfg_obj.illegal_contact.self_collision_force_threshold)
            ).any(dim=1).sum(dim=-1).float()
            for name, history in self._contact_histories.items()
        }

        terms = self._reward_terms
        terms.zero_()
        terms[:, 0] = self._reward_weights[0] * torch.exp(
            -linear_error / max(float(reward_cfg.lin_vel_std) ** 2, 1.0e-6)
        )
        terms[:, 1] = self._reward_weights[1] * torch.exp(
            -angular_error / max(float(reward_cfg.ang_vel_std) ** 2, 1.0e-6)
        )
        terms[:, 2] = self._reward_weights[2] * torch.exp(
            -upright_error / max(float(reward_cfg.upright_std) ** 2, 1.0e-6)
        )
        terms[:, 3] = self._reward_weights[3] * torch.exp(-pose_error)
        torso_ang = self._arrays["cvel"][:, self._torso_body_id, :3]
        terms[:, 4] = self._reward_weights[4] * torso_ang[:, :2].square().sum(dim=1)
        terms[:, 5] = self._reward_weights[5] * self._arrays["sensordata"][:, self._sensor_slices[2]].square().sum(dim=1)
        terms[:, 6] = self._reward_weights[6] * joint_limit_cost
        terms[:, 7] = self._reward_weights[7] * action_rate
        terms[:, 8] = self._reward_weights[8] * air_time
        terms[:, 9] = self._reward_weights[9] * foot_clearance
        terms[:, 10] = self._reward_weights[10] * swing_height
        terms[:, 11] = self._reward_weights[11] * foot_slip
        terms[:, 12] = self._reward_weights[12] * landing
        terms[:, 13] = self._reward_weights[13] * collision_costs["self_collision"]
        terms[:, 14] = self._reward_weights[14] * collision_costs.get("shank_ground_touch", 0.)
        terms[:, 15] = self._reward_weights[15] * collision_costs.get("trunk_ground_touch", 0.)
        if "run_progress" in self._reward_term_names:
            progress_index = self._reward_term_names.index("run_progress")
            run_progress = (
                linear_body[:, 0] / self._command[:, 0].clamp(min=0.1)
            ).clamp(min=0.0, max=1.0) * self._is_run_env.float()
            terms[:, progress_index] = (
                self._reward_weights[progress_index] * run_progress
            )
        if "bound_gait" in self._reward_term_names:
            gait_index = self._reward_term_names.index("bound_gait")
            gait_score = _bound_gait_score_torch(
                foot_found,
                foot_velocity,
                self._foot_height,
                self._action,
                linear_body[:, 0],
                self._command[:, 0],
                self._is_run_env,
                foot_indices=self._gait_foot_indices,
                joint_indices=self._gait_joint_indices,
                motion_std=reward_cfg.bound_gait_motion_std,
                action_std=reward_cfg.bound_gait_action_std,
                height_sync_std=reward_cfg.bound_gait_height_sync_std,
                height_separation_std=reward_cfg.bound_gait_height_separation_std,
                trot_penalty=reward_cfg.bound_gait_trot_penalty,
            )
            terms[:, gait_index] = self._reward_weights[gait_index] * gait_score
        torch.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0, out=terms)
        self._reward.copy_(terms.sum(dim=1) * self.step_dt)
        self._episode_returns.add_(self._reward)

    def _compute_observations(self) -> None:
        self._refresh_foot_contacts()
        self._observation_buffer_index = (self._observation_buffer_index + 1) % len(self._actor_obs_buffers)
        self._actor_obs = self._actor_obs_buffers[self._observation_buffer_index]
        self._critic_obs = self._critic_obs_buffers[self._observation_buffer_index]
        sensordata = self._arrays["sensordata"]
        base_linear_velocity = sensordata[:, self._sensor_slices[0]]
        base_angular_velocity = sensordata[:, self._sensor_slices[1]]
        quaternion = self._arrays["xquat"][:, self._base_body_id]
        projected_gravity = _quat_apply_inverse(quaternion, self._gravity_world)
        joint_position = self._arrays["qpos"].index_select(1, self._joint_qpos_ids)
        joint_velocity = self._arrays["qvel"].index_select(1, self._joint_dof_ids)
        actor = self._actor_obs
        actor[:, 0:3].copy_(base_linear_velocity)
        actor[:, 3:6].copy_(base_angular_velocity)
        actor[:, 6:9].copy_(projected_gravity)
        actor[:, 9 : 9 + self.num_actions].copy_(
            joint_position - self._default_joint_pos.unsqueeze(0)
        )
        joint_velocity_start = 9 + self.num_actions
        action_start = joint_velocity_start + self.num_actions
        command_start = action_start + self.num_actions
        height_start = command_start + 3
        actor[:, joint_velocity_start:action_start].copy_(joint_velocity)
        actor[:, action_start:command_start].copy_(self._action)
        actor[:, command_start:height_start].copy_(self._command)
        actor[:, height_start:].copy_(
            self._terrain_scan_height
            / max(float(self.cfg_obj.observations.terrain_scan_max_distance), 1.0e-6)
        )

        critic = self._critic_obs
        critic[:, : self.num_obs].copy_(actor)
        critic_offset = self.num_obs
        critic[:, critic_offset : critic_offset + self._foot_count].copy_(self._foot_height)
        critic_offset += self._foot_count
        critic[:, critic_offset : critic_offset + self._foot_count].copy_(
            self._current_air_time
        )
        critic_offset += self._foot_count
        critic[:, critic_offset : critic_offset + self._foot_count].copy_(
            (self._feet_contact["found"] > 0).float()
        )
        critic_offset += self._foot_count
        contact_force = self._feet_contact["force"].flatten(start_dim=1)
        critic[:, critic_offset:].copy_(
            torch.sign(contact_force) * torch.log1p(contact_force.abs())
        )
        if self.cfg_obj.observations.actor_noise:
            self._apply_actor_noise()
        if hasattr(self, "_state"):
            self._state.obs = self._observation_views[self._observation_buffer_index]

    def _apply_actor_noise(self) -> None:
        ranges = self.cfg_obj.observations.actor_noise_ranges
        slices = {
            "base_lin_vel": slice(0, 3),
            "base_ang_vel": slice(3, 6),
            "projected_gravity": slice(6, 9),
            "joint_pos": slice(9, 9 + self.num_actions),
            "joint_vel": slice(9 + self.num_actions, 9 + 2 * self.num_actions),
        }
        for name, section in slices.items():
            low, high = ranges[name]
            self._actor_obs[:, section].add_(
                self._uniform(self._actor_obs[:, section].shape, low, high)
            )
        height_start = 9 + 3 * self.num_actions + 3
        low, high = ranges["height_scan"]
        scale = max(float(self.cfg_obj.observations.terrain_scan_max_distance), 1.0e-6)
        self._actor_obs[:, height_start:].add_(
            self._uniform(self._actor_obs[:, height_start:].shape, low / scale, high / scale)
        )

    def _make_step_info(
        self,
        reset_reason: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "steps": self.episode_length_buf,
            "time_outs": time_outs,
            "reset_reason": reset_reason,
        }
        if not self.collect_step_extras:
            return info
        linear_body, _ = self._root_velocity_body()
        velocity_error = torch.linalg.vector_norm(
            self._command[:, :2] - linear_body[:, :2], dim=1
        )
        foot_found = self._feet_contact["found"] > 0
        foot_velocity = self._foot_velocity_world()
        foot_slip = (
            torch.linalg.vector_norm(foot_velocity[..., :2], dim=-1).square()
            * foot_found.float()
        ).sum(dim=1)
        terrain_normal = self._terrain_normal_world()
        terrain_up_body = _quat_apply_inverse(
            self._arrays["xquat"][:, self._base_body_id], terrain_normal
        )
        terrain_normal_error = terrain_up_body[:, :2].square().sum(dim=1)
        log = {
            "/velocity/terrain_level": self._terrain_levels.float().mean(),
            "/velocity/velocity_error": velocity_error.mean(),
            "/velocity/foot_clearance": self._foot_height.mean(),
            "/velocity/foot_contact_ratio": foot_found.float().mean(),
            "/velocity/foot_slip": foot_slip.mean(),
            "/velocity/terrain_normal_error": terrain_normal_error.mean(),
            "/velocity/encoder_bias_abs": self._encoder_bias.abs().mean(),
            "/velocity/push_count": self._push_count.float().mean(),
            "/velocity/command_stage": torch.tensor(
                float(self._current_command_stage), device=self._torch_device
            ),
            "/velocity/run_command_stage": torch.tensor(
                float(self._current_run_command_stage), device=self._torch_device
            ),
            "/velocity/reset_reason": reset_reason.float().mean(),
            "/velocity/command_speed": torch.linalg.vector_norm(
                self._command[:, :2], dim=1
            ).mean(),
            "/velocity/command_yaw_abs": self._command[:, 2].abs().mean(),
            "/velocity/base_clearance": self._arrays["xpos"][:, self._base_body_id, 2].mean(),
            "/velocity/command_vx": self._command[:, 0].mean(),
            "/velocity/command_vy": self._command[:, 1].mean(),
            "/velocity/command_yaw": self._command[:, 2].mean(),
            "/velocity/run_env_ratio": self._is_run_env.float().mean(),
        }
        terrain_level_sums = torch.zeros(
            self._spawn_cols, dtype=torch.float32, device=self._torch_device
        )
        terrain_level_sums.scatter_add_(
            0, self._terrain_types, self._terrain_levels.float()
        )
        terrain_type_counts = torch.bincount(
            self._terrain_types, minlength=self._spawn_cols
        ).clamp(min=1)
        terrain_level_means = terrain_level_sums / terrain_type_counts
        sub_terrains = self._terrain_config.get("sub_terrains", [])
        for index in range(self._spawn_cols):
            entry = sub_terrains[index] if index < len(sub_terrains) else None
            name = (
                str(entry.get("name", f"type_{index}"))
                if isinstance(entry, Mapping)
                else f"type_{index}"
            )
            log[f"/velocity/terrain_level/{name}"] = terrain_level_means[index]
        for index, name in enumerate(self._reward_term_names):
            log[f"reward/{name}"] = self._reward_terms[:, index].mean()
        info["log"] = log
        info["reward_terms"] = {
            name: self._reward_terms[:, index]
            for index, name in enumerate(self._reward_term_names)
        }
        return info

    def last_step_profile_ms(self) -> dict[str, float]:
        return dict(self._last_step_profile_ms)

    def memory_profile(self) -> dict[str, Any]:
        return {
            "backend": "mujoco-warp",
            "device": self.device,
            "graph_capture": self.provider.capabilities.graph_capture,
            "nconmax": self.provider.capacities["nconmax"],
            "njmax": self.provider.capacities["njmax"],
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(self._torch_device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(self._torch_device)),
        }


__all__ = ["WarpVelocityEnv"]
