"""Load one tracking policy with the same scene, motion and observation contract."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from gobot.rl.rsl_rl import RslRlVecEnvWrapper
from gobot.rl.policy import scene_bundle_digest
from gobot.rl.training.workflow import update_config
from .env import TrackingEnv
from .training import Config, contract


def load_policy(checkpoint, *, backend="mujoco-warp", device=None, project_path=None):
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["infos"]["tracking_contract"]["schema"] != "gobot_tracking_v1":
        raise ValueError("checkpoint does not use the current tracking contract")
    cfg = update_config(Config(), json.loads(checkpoint.with_name("config.json").read_text()))
    torch.set_num_threads(cfg.torch_threads)
    torch.backends.cuda.matmul.fp32_precision = "tf32" if cfg.allow_tf32 else "ieee"
    torch.backends.cudnn.fp32_precision = "tf32" if cfg.allow_tf32 else "ieee"
    cfg = replace(
        cfg, num_envs=1, backend=backend,
        device=device or ("cuda:0" if backend == "mujoco-warp" else "cpu"),
        project_path=Path(project_path) if project_path else cfg.project_path,
        observation_noise=False, domain_randomization=False, reset_noise=False,
        pushes=False, sampling_mode="start",
    )
    deployment = payload["infos"].get("deployment")
    if deployment is not None:
        if deployment["schema"] != "gobot_tracking_deployment_v1":
            raise ValueError("unsupported tracking deployment schema")
        if deployment["scene_bundle_digest"] != scene_bundle_digest(cfg.project_path, cfg.scene_path):
            raise ValueError("deployment authoring scene or assets differ")
    env = TrackingEnv(cfg, autoreset=False)
    try:
        env.max_episode_length = env.motion.frames + 1
        validate_contract(payload["infos"], env)
        wrapper = RslRlVecEnvWrapper(env, device=env.device)
        runner = OnPolicyRunner(wrapper, deepcopy(cfg.agent), log_dir=None, device=env.device)
        runner.alg.load(payload, {"actor": True}, True)
        return env, wrapper, runner.get_inference_policy(env.device)
    except BaseException:
        env.close()
        raise


def _validate_snapshot(expected, actual, path="snapshot", *, transform=False):
    """Only derived transform values permit a few float32 rounding steps."""
    if isinstance(actual, np.ndarray):
        actual = actual.tolist()
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"deployment physics differs at {path}")
        for key, value in expected.items():
            _validate_snapshot(value, actual[key], f"{path}/{key}",
                               transform=transform or key in ("transform", "global_transform"))
    elif isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError(f"deployment physics differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _validate_snapshot(left, right, f"{path}/{i}", transform=transform)
    elif transform and isinstance(expected, float):
        tolerance = 4 * np.finfo(np.float32).eps
        if not np.isfinite(expected) or not np.isfinite(actual) or not np.isclose(
                expected, actual, rtol=tolerance, atol=tolerance):
            raise ValueError(f"deployment physics differs at {path}")
    elif expected != actual:
        raise ValueError(f"deployment physics differs at {path}")


def validate_contract(info, env):
    expected, actual = info["tracking_contract"], contract(env)
    deployment = info.get("deployment")
    if deployment is None:
        if expected != actual:
            raise ValueError("checkpoint scene, motion or observation contract differs; "
                             "for another machine, export on the training host with "
                             "python -m gobot.rl.tracking.export")
        return
    if deployment["schema"] != "gobot_tracking_deployment_v1":
        raise ValueError("unsupported tracking deployment schema")
    if deployment["scene_bundle_digest"] != scene_bundle_digest(env.project_path, env.scene_path):
        raise ValueError("deployment authoring scene or assets differ")
    # Compiled XML text and float64 FK hashes depend on compiler/BLAS rounding.
    # The exported scene hash and full snapshot replace those two text hashes.
    derived = {"scene_digest", "kinematics_digest"}
    if {k: v for k, v in expected.items() if k not in derived} != {
            k: v for k, v in actual.items() if k not in derived}:
        raise ValueError("deployment motion, observations or controls differ")
    _validate_snapshot(deployment["physics_snapshot"], env.batch.snapshot)
