"""Tracking MDP configuration, pinned to mjlab dcdf6ce5 (Apache-2.0)."""

from dataclasses import dataclass, field
import math
from pathlib import Path

REFERENCE_REVISION = "dcdf6ce519abfd2ea45e9503ae6fb650b75bb869"
G1_JOINTS = tuple(
    [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    ]
    + [f"waist_{a}_joint" for a in ("yaw", "roll", "pitch")]
    + [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
G1_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
REWARD_WEIGHTS = dict(
    motion_global_root_pos=0.5,
    motion_global_root_ori=0.5,
    motion_body_pos=1.0,
    motion_body_ori=1.0,
    motion_body_lin_vel=1.0,
    motion_body_ang_vel=1.0,
    action_rate_l2=-0.1,
    joint_limit=-10.0,
    self_collisions=-10.0,
)


@dataclass
class TrackingConfig:
    project_path: Path = Path("logs/g1-tracking-assets")
    scene_path: str = "res://world.jscn"
    motion_file: Path = Path("motion.npz")  # Relative to project_path.
    motion_layout: Path | None = None  # Explicit names for legacy, unnamed NPZ arrays.
    robot_name: str = "g1"
    base_link: str = "pelvis"
    anchor_link: str = "torso_link"
    joint_names: tuple[str, ...] = G1_JOINTS
    body_names: tuple[str, ...] = G1_BODIES
    end_effectors: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    imu_link: str = "pelvis"
    imu_position: tuple[float, ...] = (0.04525, 0.0, -0.08339)
    imu_orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    backend: str = "mujoco-warp"
    device: str | None = None
    num_envs: int = 4096
    env_spacing: float = 2.0
    seed: int = 42
    random_source: str = "torch_global"
    physics_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 10.0
    sampling_mode: str = "adaptive"
    adaptive_alpha: float = 0.001
    adaptive_uniform_ratio: float = 0.1
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    reset_noise: bool = True
    observation_noise: bool = True
    domain_randomization: bool = True
    pushes: bool = True
    state_estimation: bool = True
    capture_graphs: bool = True
    nconmax: int = 35
    njmax: int = 250
    soft_joint_limit_factor: float = 0.9
    reward_weights: dict = field(default_factory=lambda: dict(REWARD_WEIGHTS))
    solver_settings: dict = field(
        default_factory=lambda: dict(
            solver="Newton",
            integrator="ImplicitFast",
            cone="Pyramidal",
            iterations=10,
            line_search_iterations=20,
            convex_collision_iterations=50,
            impedance_ratio=1.0,
        )
    )

    def validate(self):
        for name in (
            "num_envs",
            "decimation",
            "adaptive_kernel_size",
            "nconmax",
            "njmax",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("physics_dt", "episode_length_s", "adaptive_uniform_ratio"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.backend not in ("mujoco-cpu", "mujoco-warp"):
            raise ValueError("unsupported tracking backend")
        if self.random_source not in ("torch_global", "local_generator"):
            raise ValueError("random_source must be torch_global or local_generator")
        if self.sampling_mode not in ("adaptive", "uniform", "start"):
            raise ValueError("unknown sampling mode")
        if not 0 < self.adaptive_alpha <= 1 or not 0 <= self.adaptive_lambda <= 1:
            raise ValueError("invalid adaptive sampling parameters")
        if not 0 < self.soft_joint_limit_factor <= 1:
            raise ValueError("invalid soft joint limit factor")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint names must be unique")
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("body names must be unique")
        if not self.end_effectors:
            raise ValueError("at least one end effector is required")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if self.body_names[0] != self.base_link:
            raise ValueError("the first tracked body must be the floating base")
        if not set((self.anchor_link, self.imu_link, *self.end_effectors)) <= set(
            self.body_names
        ):
            raise ValueError("anchor, IMU and end effectors must be tracked")
        if (
            len(self.imu_position) != 3
            or len(self.imu_orientation) != 4
            or not all(map(math.isfinite, (*self.imu_position, *self.imu_orientation)))
        ):
            raise ValueError("invalid IMU transform")
        if not math.isclose(
            sum(x * x for x in self.imu_orientation), 1.0, abs_tol=1e-5
        ):
            raise ValueError("IMU orientation must be unit wxyz")
        if set(self.reward_weights) != set(REWARD_WEIGHTS) or not all(
            map(math.isfinite, self.reward_weights.values())
        ):
            raise ValueError("invalid reward weights")
