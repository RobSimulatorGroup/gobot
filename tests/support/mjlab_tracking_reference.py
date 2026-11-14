"""Pinned, engine-independent mjlab tracking baseline helpers.

This module deliberately never imports gobot. The reference sources stay intact;
wrappers only persist artifacts, measure completed episodes and check capacity.
"""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import types

REFERENCE = "dcdf6ce519abfd2ea45e9503ae6fb650b75bb869"
VERSIONS = {
    "torch": "2.11.0",
    "mujoco": "3.17.0",
    "mujoco-warp": "3.17.0",
    "warp-lang": "1.17.0",
    "rsl-rl-lib": "5.5.2",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE:
        raise RuntimeError(f"expected mjlab {REFERENCE}; got {revision}")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "src",
            "pyproject.toml",
        ],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("reference numerical sources are modified: " + dirty)
    versions = {name: metadata.version(name) for name in VERSIONS}
    for name, expected in VERSIONS.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {versions[name]}")
    sys.path.insert(0, str(source / "src"))
    import torch

    if torch.version.cuda != "12.8":
        raise RuntimeError("reference requires Torch CUDA 12.8")
    # Avoid unrelated task registration. Imported task/runner implementations
    # below remain the original source at the pinned revision.
    for name, relative in [
        ("mjlab.tasks", "tasks"),
        ("mjlab.tasks.tracking.config", "tasks/tracking/config"),
        ("mjlab.tasks.tracking.config.g1", "tasks/tracking/config/g1"),
    ]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(source / "src/mjlab" / relative)]
            sys.modules[name] = package
    files = [
        p
        for folder in ("tasks/tracking", "envs", "managers", "entity", "sim", "rl")
        for p in (source / "src/mjlab" / folder).rglob("*.py")
    ]
    return dict(
        reference_revision=revision,
        versions=versions,
        source_sha256={
            str(p.relative_to(source)): sha256(p) for p in sorted(set(files))
        },
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def export_motion(source, csv, output, device="cuda:0"):
    identity = reference(source)
    import numpy as np
    import torch
    from mjlab.scene import Scene
    from mjlab.sim.sim import Simulation
    from mjlab.scripts.csv_to_npz import MotionLoader
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = 256
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 500
    scene = Scene(cfg.scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=256, cfg=cfg.sim, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    motion = MotionLoader(str(csv), 30, 50, "cpu")
    # LAFAN's G1 source columns are the native named articulation order.
    names = tuple(robot.joint_names)
    expected = tuple(
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
    if names != expected:
        raise ValueError("LAFAN joint source order differs from native G1 model")
    fields = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    arrays = {k: [] for k in fields}
    for lo in range(0, motion.output_frames, 256):
        count = min(256, motion.output_frames - lo)
        idx = torch.arange(lo, lo + 256).clamp(max=motion.output_frames - 1)
        root = torch.cat(
            (
                motion.motion_base_poss[idx],
                motion.motion_base_rots[idx],
                motion.motion_base_lin_vels[idx],
                motion.motion_base_ang_vels[idx],
            ),
            -1,
        ).to(device)
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(
            motion.motion_dof_poss[idx].to(device),
            motion.motion_dof_vels[idx].to(device),
        )
        sim.forward()
        scene.update(cfg.sim.mujoco.timestep)
        data = robot.data
        values = (
            data.joint_pos,
            data.joint_vel,
            data.body_link_pos_w,
            data.body_link_quat_w,
            data.body_link_lin_vel_w,
            data.body_link_ang_vel_w,
        )
        for key, value in zip(fields, values, strict=True):
            arrays[key].append(value[:count].cpu().numpy().copy())
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    np.savez_compressed(
        output / "motion.npz",
        **arrays,
        fps=np.array([50.0], np.float32),
        joint_names=np.array(names),
        body_names=np.array(robot.body_names),
    )
    scene.write(output / "scene", zip=False) if hasattr(
        scene, "write"
    ) else scene.export(output / "scene", zip=False)
    profile = dict(
        identity,
        csv=str(Path(csv).resolve()),
        csv_sha256=sha256(csv),
        motion_sha256=sha256(output / "motion.npz"),
        input_fps=30,
        output_fps=50,
        frames=motion.output_frames,
        body_names=list(robot.body_names),
        joint_names=list(names),
        reference_conversion="unmodified mjlab MotionLoader + native Scene/Simulation forward; no Gobot import",
    )
    write_json(output / "reference.json", profile)
    print(
        json.dumps(
            profile
            | {
                "source_sha256": f"{len(identity['source_sha256'])} pinned source hashes"
            },
            indent=2,
        )
    )


class TrackingLog:
    """Non-invasive global logging; native env/rewards/runner update unchanged."""

    def __init__(self, env, runner, output, settings):
        import torch

        self.env, self.runner, self.output, self.settings = (
            env,
            runner,
            Path(output),
            settings,
        )
        self.names = tuple(env.reward_manager.active_terms)
        self.acc = torch.zeros(
            env.num_envs, len(self.names), device=env.device, dtype=torch.float64
        )
        self.steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.sums = torch.zeros(
            len(self.names) + 2, device=env.device, dtype=torch.float64
        )
        self.start = time.monotonic()
        self.collection_start = None
        self.peak_memory = 0
        original_reward = env.reward_manager.compute

        def reward(dt):
            value = original_reward(dt)
            self.acc.add_(env.reward_manager._step_reward * dt)
            self.steps += 1
            ids = env.reset_buf.nonzero().flatten()
            if len(ids):
                self.sums[0] += len(ids)
                self.sums[1] += self.steps[ids].sum()
                self.sums[2:] += self.acc[ids].sum(0)
                self.acc[ids] = 0
                self.steps[ids] = 0
            return value

        env.reward_manager.compute = reward
        native_step = env.step

        def step(action):
            result = native_step(action)
            import mujoco_warp as mjw
            import warp as wp

            flags = wp.to_torch(env.sim.wp_data.overflow)
            ignored = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
            if bool((flags & ~ignored).any()):
                raise RuntimeError("reference simulation capacity overflow")
            return result

        env.step = step
        native_act = runner.alg.act

        def act(obs):
            if self.collection_start is None:
                self.collection_start = self.clock()
            return native_act(obs)

        runner.alg.act = act
        native_returns = runner.alg.compute_returns

        def returns(obs):
            self.collection_end = self.clock()
            return native_returns(obs)

        runner.alg.compute_returns = returns
        native_update = runner.alg.update

        def update():
            value = native_update()
            self.update_end = self.clock()
            return value

        runner.alg.update = update
        runner.logger.process_env_step = lambda *a, **kw: None
        self.native_log = runner.logger.log
        runner.logger.log = self.log

    def clock(self):
        import torch

        torch.cuda.synchronize(self.env.device)
        return time.monotonic()

    def log(self, **kw):
        import torch

        sums = self.sums.clone()
        self.sums.zero_()
        times = torch.tensor(
            [
                self.collection_end - self.collection_start,
                self.update_end - self.collection_end,
            ],
            device=self.env.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(times, op=torch.distributed.ReduceOp.MAX)
        count = float(sums[0])
        stats = {}
        if count:
            stats = {
                f"Episode_Reward/{k}": float(v)
                / (count * self.env.max_episode_length_s)
                for k, v in zip(self.names, sums[2:], strict=True)
            }
            stats["Episode/length"] = float(sums[1]) / count
        kw["collect_time"], kw["learn_time"] = times.tolist()
        logger = self.runner.logger
        logger.ep_extras = [dict(stats)] if count else []
        if count:
            logger.rewbuffer.clear()
            logger.rewbuffer.append(float(sums[2:].sum()) / count)
            logger.lenbuffer.clear()
            logger.lenbuffer.append(float(sums[1]) / count)
        self.native_log(**kw)
        if self.runner.gpu_global_rank == 0:
            item = dict(
                state="training",
                iteration=kw["it"],
                elapsed_s=time.monotonic() - self.start,
                samples=(kw["it"] + 1)
                * self.settings["num_envs_per_rank"]
                * self.settings["world_size"]
                * self.settings["rollout_steps"],
                collection_s=kw["collect_time"],
                learning_s=kw["learn_time"],
                completed_episodes=int(count),
                episode=stats,
                learning_rate=kw["learning_rate"],
                losses=kw["loss_dict"],
            )
            with (self.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(item) + "\n")
            write_json(self.output / "status.json", item)
        self.collection_start = None
