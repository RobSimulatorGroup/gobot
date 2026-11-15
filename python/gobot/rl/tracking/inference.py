"""Load one tracking policy with the same scene, motion and observation contract."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from gobot.rl.rsl_rl import RslRlVecEnvWrapper
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
    env = TrackingEnv(cfg, autoreset=False)
    try:
        env.max_episode_length = env.motion.frames + 1
        if payload["infos"]["tracking_contract"] != contract(env):
            raise ValueError("checkpoint scene, motion or observation contract differs")
        wrapper = RslRlVecEnvWrapper(env, device=env.device)
        runner = OnPolicyRunner(wrapper, deepcopy(cfg.agent), log_dir=None, device=env.device)
        runner.alg.load(payload, {"actor": True}, True)
        return env, wrapper, runner.get_inference_policy(env.device)
    except BaseException:
        env.close()
        raise
