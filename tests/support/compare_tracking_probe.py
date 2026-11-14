"""Compare Gobot transitions to independently recorded native mjlab outputs."""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from gobot.rl.tracking import TrackingConfig, TrackingEnv
from gobot.rl.training.workflow import update_config


def compare(project, probe, output, backend="mujoco-warp"):
    torch.set_num_threads(4)
    project = Path(project).resolve()
    probe = Path(probe)
    cfg = update_config(
        TrackingConfig(), json.loads((project / "task.json").read_text())
    )
    cfg.backend = backend
    cfg.device = "cpu" if backend == "mujoco-cpu" else "cuda:0"
    cfg.num_envs = 4
    meta = json.loads((probe / "probe.json").read_text())
    randomized = meta.get("randomized", False)
    cfg.domain_randomization = randomized
    cfg.observation_noise = randomized
    cfg.reset_noise = randomized
    cfg.pushes = randomized
    data = np.load(probe / "probe.npz")
    env = TrackingEnv(cfg, autoreset=randomized)
    try:
        initial = env.state if randomized else env.reset(frames=data["frames"])
        values = {
            "initial_actor": initial.obs["actor"].cpu().numpy().copy(),
            "initial_critic": initial.obs["critic"].cpu().numpy().copy(),
        }
        terms = []
        bodies = []
        observations = []
        ends = []
        rngs = []
        reference_frames = []
        values["initial_rng"] = (
            torch.cuda.get_rng_state().cpu().numpy()
            if backend == "mujoco-warp"
            else torch.get_rng_state().numpy()
        )
        for action in data["actions"]:
            state = env.step(torch.tensor(action, device=env.device))
            terms.append(
                torch.stack(
                    [state.info["reward_terms"][k] for k in meta["reward_names"]], -1
                )
                .cpu()
                .numpy()
            )
            bodies.append(torch.cat(env.reward_bodies, -1).cpu().numpy())
            observations.append(
                torch.cat((state.obs["actor"], state.obs["critic"]), -1).cpu().numpy()
            )
            ends.append(
                torch.stack((state.terminated, state.truncated), -1).cpu().numpy()
            )
            rngs.append(
                torch.cuda.get_rng_state().cpu().numpy()
                if backend == "mujoco-warp"
                else torch.get_rng_state().numpy()
            )
            reference_frames.append(env.frame.clone().cpu().numpy())
        values.update(
            rng=np.stack(rngs),
            reference_frames=np.stack(reference_frames),
            rewards=np.stack(terms),
            bodies=np.stack(bodies),
            observations=np.stack(observations),
            terminations=np.stack(ends),
        )
        differences = {}
        for key, value in values.items():
            if key not in data:
                continue
            expected = data[key]
            diff = np.abs(value.astype(float) - expected.astype(float))
            differences[key] = dict(
                max_abs=float(diff.max()),
                rms=float(np.sqrt(np.mean(diff**2))),
                shape=list(value.shape),
            )
        contact_index = meta["reward_names"].index("self_collisions")
        continuous = [i for i in range(len(meta["reward_names"])) if i != contact_index]
        error = float(
            np.abs(
                values["rewards"][..., continuous] - data["rewards"][..., continuous]
            ).max()
        )
        differences["continuous_reward_max_abs"] = error
        differences["self_collision_discrete_mismatches"] = int(
            np.count_nonzero(
                values["rewards"][..., contact_index]
                != data["rewards"][..., contact_index]
            )
        )
        differences["passed_continuous"] = (
            error < 2e-5
            and differences["observations"]["max_abs"] < 2e-3
            and differences["terminations"]["max_abs"] == 0
        )
        if randomized:
            differences["passed_continuous"] &= (
                differences["rng"]["max_abs"] == 0
                and differences["reference_frames"]["max_abs"] == 0
            )
        np.savez_compressed(Path(output).with_suffix(".npz"), **values)
        Path(output).write_text(json.dumps(differences, indent=2) + "\n")
        print(json.dumps(differences, indent=2))
        if not differences["passed_continuous"]:
            raise AssertionError("tracking transition comparison failed")
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--backend", choices=["mujoco-cpu", "mujoco-warp"], default="mujoco-warp"
    )
    compare(**vars(p.parse_args()))
