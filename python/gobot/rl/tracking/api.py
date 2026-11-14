"""Train one reference motion with one policy, on one or several devices."""

from pathlib import Path


def train_motion(
    project_path,
    motion_file="motion.npz",
    *,
    output,
    scene_path="res://world.jscn",
    num_envs=4096,
    updates=30000,
    seed=42,
    backend="mujoco-warp",
    device=None,
    batch_scope="per_rank",
    nconmax=128,
    njmax=500,
    resume=None,
):
    """Train exactly one named NPZ motion against an authored Gobot scene.

    Run normally for one device, or run the same Python script with torchrun for
    synchronized data parallel PPO. num_envs is per rank by default; use
    batch_scope='global' to split a fixed total evenly. Only rank zero writes the
    shared model and log files. The output directory must be new.
    """
    from gobot.rl.training import launch
    from .training import Config, training_spec

    cfg = Config(
        project_path=Path(project_path),
        motion_file=Path(motion_file),
        scene_path=scene_path,
        output=Path(output),
        num_envs=num_envs,
        updates=updates,
        seed=seed,
        backend=backend,
        device=device,
        batch_scope=batch_scope,
        nconmax=nconmax,
        njmax=njmax,
        resume=None if resume is None else Path(resume),
    )
    return launch(training_spec(), cfg)
