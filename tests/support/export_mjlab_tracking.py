"""Export an independent numerical oracle using only pinned mjlab code.

Run: PYTHONPATH=mjlab/src .venv/bin/python tests/support/export_mjlab_tracking.py
"""

# ruff: noqa: E402 -- imports follow isolated upstream package setup.
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch

# Avoid importing unrelated task registries and the optional W&B tracking runner.
# All numerical functions below are still imported unmodified from mjlab.
import sys
from types import ModuleType
import mjlab

for name, relative in [
    ("mjlab.tasks", "tasks"),
    ("mjlab.tasks.tracking.config", "tasks/tracking/config"),
    ("mjlab.tasks.tracking.config.g1", "tasks/tracking/config/g1"),
]:
    package = ModuleType(name)
    package.__path__ = [str(Path(mjlab.__file__).parent / relative)]
    sys.modules[name] = package
from mjlab.tasks.tracking.mdp import (
    observations as obs,
    rewards as rew,
    terminations as term,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply, quat_mul

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    "mdp/commands.py",
    "mdp/observations.py",
    "mdp/rewards.py",
    "mdp/terminations.py",
    "tracking_env_cfg.py",
    "config/g1/env_cfgs.py",
    "config/g1/rl_cfg.py",
]
base = ROOT / "mjlab/src/mjlab/tasks/tracking"
g = torch.Generator().manual_seed(234)


def rnd(*shape):
    return torch.randn(*shape, generator=g)


B, J, L, A = 4, 3, 4, 1
p = rnd(B, L, 3)
r = rnd(B, L, 4)
r /= r.norm(dim=-1, keepdim=True)
rp = p + rnd(B, L, 3) * 0.2
rr = rnd(B, L, 4)
rr /= rr.norm(dim=-1, keepdim=True)
v = rnd(B, L, 3)
w = rnd(B, L, 3)
rv = rnd(B, L, 3)
rw = rnd(B, L, 3)
q = rnd(B, J)
qd = rnd(B, J)
action = rnd(B, J)
previous = rnd(B, J)
default = rnd(J)
bias = rnd(B, J) * 0.01
reference_joint = rnd(B, 2 * J)
imu_position = torch.tensor([0.03, 0.01, -0.08])
imu_orientation = torch.tensor([1.0, 0, 0, 0])
names = ("base", "anchor", "foot", "hand")
command = NS(
    cfg=NS(body_names=names),
    anchor_pos_w=rp[:, A],
    anchor_quat_w=rr[:, A],
    robot_anchor_pos_w=p[:, A],
    robot_anchor_quat_w=r[:, A],
    body_pos_w=rp,
    body_quat_w=rr,
    robot_body_pos_w=p,
    robot_body_quat_w=r,
    body_lin_vel_w=rv,
    body_ang_vel_w=rw,
    robot_body_lin_vel_w=v,
    robot_body_ang_vel_w=w,
)
MotionCommand.update_relative_body_poses(command)
robot = NS(data=NS(gravity_vec_w=torch.tensor([0.0, 0.0, -1.0]).expand(B, -1)))
history = rnd(B, 2, 4, 3) * 7
sensor = NS(data=NS(force_history=history))
env = NS(
    num_envs=B,
    device="cpu",
    command_manager=NS(get_term=lambda _: command),
    scene={"robot": robot, "self_collision": sensor},
)
actor = {
    "command": reference_joint,
    "motion_anchor_pos_b": obs.motion_anchor_pos_b(env, "motion"),
    "motion_anchor_ori_b": obs.motion_anchor_ori_b(env, "motion"),
}
critic_body = {
    "body_pos": obs.robot_body_pos_b(env, "motion"),
    "body_ori": obs.robot_body_ori_b(env, "motion"),
}
orientation = quat_mul(r[:, 0], imu_orientation.expand(B, -1))
tail = dict(
    base_lin_vel=quat_apply_inverse(
        orientation,
        v[:, 0]
        + torch.linalg.cross(w[:, 0], quat_apply(r[:, 0], imu_position.expand(B, -1))),
    ),
    base_ang_vel=quat_apply_inverse(orientation, w[:, 0]),
    joint_pos=q - default,
    joint_vel=qd,
    actions=action,
)
critic = {**actor, **critic_body, **tail}
actor = {**actor, **tail, "joint_pos": q - default + bias}
functions = {
    "motion_global_root_pos": ("motion_global_anchor_position_error_exp", 0.3),
    "motion_global_root_ori": ("motion_global_anchor_orientation_error_exp", 0.4),
    "motion_body_pos": ("motion_relative_body_position_error_exp", 0.3),
    "motion_body_ori": ("motion_relative_body_orientation_error_exp", 0.4),
    "motion_body_lin_vel": ("motion_global_body_linear_velocity_error_exp", 1.0),
    "motion_body_ang_vel": ("motion_global_body_angular_velocity_error_exp", 3.14),
}
rates = {k: getattr(rew, f)(env, "motion", std) for k, (f, std) in functions.items()}
rates["action_rate_l2"] = (action - previous).square().sum(-1)
rates["joint_limit"] = ((-1 - q).clamp_min(0) + (q - 1).clamp_min(0)).sum(-1)
rates["self_collisions"] = rew.self_collision_cost(env, "self_collision", 10.0)
asset = SceneEntityCfg("robot")
ends = dict(
    anchor_pos=term.bad_anchor_pos_z_only(env, "motion", 0.25),
    anchor_ori=term.bad_anchor_ori(env, asset, "motion", 0.8),
    ee_body_pos=term.bad_motion_body_pos_z_only(env, "motion", 0.25, ("foot", "hand")),
)
inputs = dict(
    p=p,
    r=r,
    v=v,
    w=w,
    rp=rp,
    rr=rr,
    rv=rv,
    rw=rw,
    q=q,
    qd=qd,
    action=action,
    previous=previous,
    default=default,
    bias=bias,
    reference_joint=reference_joint,
    imu_position=imu_position,
    imu_orientation=imu_orientation,
    self_hits=rates["self_collisions"],
)
cfg = unitree_g1_flat_tracking_env_cfg()
weights = {k: v.weight for k, v in cfg.rewards.items()}
out = dict(
    reference_revision="dcdf6ce519abfd2ea45e9503ae6fb650b75bb869",
    source_sha256={
        str(p.relative_to(ROOT / "mjlab")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (base / s for s in SOURCES)
    },
    inputs={k: v.tolist() for k, v in inputs.items()},
    actor=torch.cat(tuple(actor.values()), -1).tolist(),
    critic=torch.cat(tuple(critic.values()), -1).tolist(),
    rates={k: v.tolist() for k, v in rates.items()},
    relative_position=command.body_pos_relative_w.tolist(),
    relative_orientation=command.body_quat_relative_w.tolist(),
    terminations={k: v.tolist() for k, v in ends.items()},
    weights=weights,
    agent=asdict(unitree_g1_tracking_ppo_runner_cfg()),
    actor_terms=list(cfg.observations["actor"].terms),
    critic_terms=list(cfg.observations["critic"].terms),
)
# Sampling expectations come from invoking the upstream method itself.
s = NS(
    _env=NS(
        termination_manager=NS(terminated=torch.tensor([False, True, True, False]))
    ),
    cfg=NS(adaptive_uniform_ratio=0.1, adaptive_kernel_size=3),
    device="cpu",
    bin_count=4,
    motion=NS(time_step_total=150),
    time_steps=torch.tensor([0, 40, 80, 149]),
    bin_failed_count=torch.tensor([0.0, 0.02, 0.08, 0.0]),
    _current_bin_failed=torch.zeros(4),
    kernel=torch.tensor([1.0, 0.8, 0.64]) / (1 + 0.8 + 0.64),
    metrics={
        k: torch.zeros(4)
        for k in ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin")
    },
)
torch.manual_seed(91)
MotionCommand._adaptive_sampling(s, torch.arange(4))
out["sampling"] = {
    "frames": s.time_steps.tolist(),
    "pending": s._current_bin_failed.tolist(),
    "metrics": {k: v.tolist() for k, v in s.metrics.items()},
}
(ROOT / "tests/fixtures/mjlab_tracking.json").write_text(
    json.dumps(out, indent=2, default=str) + "\n"
)
print("Exported tracking observation, reward, termination and sampler oracle")
