"""Real two-process PPO: shared updates, per-rank episodes, restore and failures."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import socket
import time
import pytest
import torch
import torch.multiprocessing as mp
from gobot.rl.tracking.training import Config, run, default_agent
from gobot.rl.tracking.sampler import AdaptiveSampler
from gobot.rl.tracking.distributed import DistributedTraining, resolve_config
from test_tracking import tiny_config


def _environment(rank, port):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        GLOO_SOCKET_IFNAME="lo",
    )


def _spawn(worker, path, *args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.spawn(worker, args=(str(path), port, *args), nprocs=2, join=False)
    deadline = time.monotonic() + 100
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("tracking distributed test timed out")
    finally:
        for process in ctx.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)


def _sampler_worker(rank, path, port):
    _environment(rank, port)
    group = DistributedTraining.from_environment("cpu")
    group.initialize()
    try:
        s = AdaptiveSampler(150, 0.02)
        s.failed.fill_(0.05)
        s.defer()
        for step in range(4):
            s.record(torch.tensor([20 + rank * 50]), torch.tensor([True]))
            s.update()
            assert s.failed.eq(0.05).all()
        s.synchronize(group)
        torch.save(s.failed, Path(path) / f"sampler{rank}.pt")
        # All workers report a rollout failure before entering PPO collectives.
        with pytest.raises(RuntimeError, match="rank 1: ValueError: deliberate"):
            group.check_error(
                ValueError("deliberate") if rank == 1 else None, phase="rollout"
            )
        with pytest.raises(RuntimeError, match="shared I/O"):
            group.on_main(lambda: (_ for _ in ()).throw(RuntimeError("shared I/O")))
    finally:
        group.close()


def test_global_failure_ema_and_collective_errors(tmp_path):
    _spawn(_sampler_worker, tmp_path)
    expected = AdaptiveSampler(150, 0.02)
    expected.failed.fill_(0.05)
    for _ in range(4):
        expected.record(torch.tensor([20, 70]), torch.tensor([True, True]))
        expected.update()
    a = torch.load(tmp_path / "sampler0.pt", weights_only=True)
    b = torch.load(tmp_path / "sampler1.pt", weights_only=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a, expected.failed, atol=1e-8, rtol=1e-6)


def _training_worker(rank, path, port, config):
    _environment(rank, port)
    from gobot.rl.training.workflow import update_config

    cfg = update_config(Config(), config)
    result = run(cfg)
    Path(path, f"result{rank}.json").write_text(json.dumps(result))


def test_shared_ppo_normalizers_checkpoint_and_resume(tmp_path):
    env_cfg = tiny_config(tmp_path)
    agent = default_agent()
    agent["num_steps_per_env"] = 4
    agent["algorithm"]["num_mini_batches"] = 2
    agent["algorithm"]["num_learning_epochs"] = 2
    agent["actor"]["hidden_dims"] = [16, 16]
    agent["critic"]["hidden_dims"] = [16, 16]
    cfg = Config(
        **asdict(env_cfg),
        updates=2,
        torch_threads=1,
        output=tmp_path / "train",
        agent=agent,
    )
    cfg.sampling_mode = "adaptive"
    cfg.sampler_sync = "rollout_global"
    cfg.init_at_random_ep_len = False
    cfg.episode_length_s = 0.04
    from gobot.rl.training.workflow import config_dict

    _spawn(_training_worker, tmp_path, config_dict(cfg))
    p = torch.load(cfg.output / "model.pt", weights_only=True)
    assert p["infos"]["world_size"] == 2 and len(p["infos"]["rank_states"]) == 2
    audit = json.loads((cfg.output / "synchronization.json").read_text())
    assert audit["passed"] and len(audit["ranks"]) == 2
    assert audit["ranks"][0]["seed"] + 1 == audit["ranks"][1]["seed"]
    assert audit["ranks"][0]["digests"] == audit["ranks"][1]["digests"]
    assert (
        int(p["actor_state_dict"]["obs_normalizer.count"]) == 32
    )  # 2 updates * 4 ticks * 4 global envs.
    assert int(p["critic_state_dict"]["obs_normalizer.count"]) == 32
    ranks = p["infos"]["rank_states"]
    assert not torch.equal(
        ranks[0]["tracking_state"]["generator"], ranks[1]["tracking_state"]["generator"]
    )
    status = json.loads((cfg.output / "status.json").read_text())
    assert status["samples"] == 32
    metrics = [
        json.loads(x) for x in (cfg.output / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 2 and metrics[0]["completed_episodes"] == 8
    assert status["actor_max_weight_change"] > 0
    from gobot.rl.tracking.evaluate import evaluate

    evaluation = evaluate(cfg.output / "model.pt")
    assert evaluation["steps"] > 2  # Training time limit is two control steps.
    assert evaluation["steps"] <= 11
    assert not evaluation["truncated"]
    assert len(list(cfg.output.glob("events.out.tfevents.*"))) == 1
    resumed = replace(
        cfg, updates=1, resume=cfg.output / "model.pt", output=tmp_path / "resumed"
    )
    _spawn(_training_worker, tmp_path, config_dict(resumed))
    saved = torch.load(resumed.output / "model.pt", weights_only=True)
    assert saved["iter"] == 2
    assert int(saved["actor_state_dict"]["obs_normalizer.count"]) == 48
    assert (
        saved["infos"]["rank_states"][1]["tracking_state"]["common_step_counter"] == 12
    )
    # Same global samples but a changed world-size must not silently resume.
    with pytest.raises(ValueError, match="does not match"):
        run(
            replace(
                cfg,
                output=tmp_path / "bad_resume",
                resume=cfg.output / "model.pt",
                num_envs=4,
            )
        )


def test_batch_scope_and_public_one_motion_api(monkeypatch, tmp_path):
    cfg = Config(project_path=tmp_path, num_envs=16, batch_scope="global")
    group = DistributedTraining("cuda:3", rank=3, local_rank=3, world_size=4)
    local = resolve_config(cfg, group)
    assert local.num_envs == 4 and local.seed == 45 and local.device == "cuda:3"
    assert cfg.num_envs == 16
    local = resolve_config(replace(cfg, batch_scope="per_rank"), group)
    assert local.num_envs == 16
    with pytest.raises(ValueError, match="divisible"):
        resolve_config(replace(cfg, num_envs=15), group)
    from gobot.rl.tracking import train_motion
    import gobot.rl.training

    captured = {}

    def launch(spec, config):
        captured.update(spec=spec, cfg=config)
        return {"ready": True}

    monkeypatch.setattr(gobot.rl.training, "launch", launch)
    assert train_motion(
        tmp_path, "dance.npz", output=tmp_path / "run", num_envs=512
    ) == {"ready": True}
    assert captured["spec"].supports_distributed
    assert captured["cfg"].motion_file == Path("dance.npz")
    assert captured["cfg"].num_envs == 512 and captured["cfg"].batch_scope == "per_rank"


def test_native_mjlab_sampling_defaults(tmp_path):
    cfg = Config(project_path=tmp_path)
    assert cfg.sampler_sync == "rank_local"
    assert cfg.random_source == "torch_global"
    assert cfg.init_at_random_ep_len is True
    assert cfg.allow_tf32 is False
    assert cfg.solver_settings["convex_collision_iterations"] == 50
