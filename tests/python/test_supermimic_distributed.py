"""Global-batch math and actual two-worker WM/BPTT training, including resume."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import os
import signal
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from gobot.rl.supermimic.config import Config
from gobot.rl.supermimic.distributed import Distributed, GradientReducer, resolve_config
from gobot.rl.supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from gobot.rl.supermimic.normalization import EmpiricalNormalization
from gobot.rl.supermimic.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
)
from gobot.rl.supermimic.sampler import ReturnSampler, ReturnSamplerCfg
from gobot.rl.supermimic.state import GenCoordRepresentation
from gobot.rl.supermimic.tasks import EncoderSpec
from gobot.rl.supermimic.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from examples.g1.supermimic._reference.probes import DifferentiableProbeEnv

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/supermimic_official_gencoord_v1.pt"
)


def _tree_close(a, b, *, exact=False):
    if torch.is_tensor(a):
        torch.testing.assert_close(
            a, b, atol=0 if exact else 1e-5, rtol=0 if exact else 1e-4
        )
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _tree_close(a[key], b[key], exact=exact)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _tree_close(x, y, exact=exact)
    else:
        assert a == b


def _math(shard=None):
    torch.set_num_threads(1)
    data = torch.load(FIXTURE, weights_only=True)
    region = slice(None) if shard is None else slice(shard * 2, shard * 2 + 2)
    rep = GenCoordRepresentation(29, data["default"])
    dynamics = Dynamics(rep, 29, 33, data["action_scale"], data["default"], (16, 12, 8))
    dynamics.load_state_dict(data["wm_initial"])
    wm = WorldModelLearner(dynamics, rep, WorldModelLearnerCfg())
    n = 4 if shard is None else 2
    batch = TensorDict(
        {
            "state": data["states"][region, :8],
            "action": data["actions"][region, :8],
            "sim_ctx": data["context"][region, None].expand(-1, 8, -1),
        },
        batch_size=[n, 8],
    )
    # Exercise every WM optimizer step, not just a standalone gradient utility.
    for _ in range(2):
        wm.update([batch, batch])
    torch.manual_seed(123)
    starts = TensorDict(data["policy_starts"], batch_size=[4])[region]
    actor = MultiEncoderAE(
        68,
        29,
        [EncoderSpec("motion", 640), EncoderSpec("velocity", 3)],
        {"motion": 1.0, "velocity": 0.0},
        32,
        (16, 12, 8),
        0.1,
        1,
    )
    critic = build_mlp(741, (12, 8), 1)
    env = DifferentiableProbeEnv(starts)
    policy = ActorCriticPolicyLearner(
        actor,
        critic,
        env,
        ActorCriticPolicyLearnerCfg(horizon=8, batch_size=n),
        compile=False,
        capture=False,
    )
    # Equal time partitions make each distributed critic minibatch exactly a
    # shard of its serial counterpart, isolating math from random permutation.
    with patch("torch.randperm", side_effect=lambda n, **kw: torch.arange(n, **kw)):
        for _ in range(2):
            policy.update()
    assert all(p.grad is None for p in env.dynamics.parameters())
    inactive = list(actor.encoders["velocity"].parameters())
    assert all(p.grad is None for p in inactive)
    assert all(p not in opt.state for p in inactive for opt in policy.actor_optimizers)
    normalizer = EmpiricalNormalization(3)
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    normalizer.update(x if shard is None else (x[:2] if shard == 0 else x[2:]))
    sampler = ReturnSampler(ReturnSamplerCfg(horizon=4, memory=3), 7, 40, n, 0.1)
    for k in range(12):
        frames = (torch.arange(4) * 7 + k) % 39
        rewards = torch.arange(4).float() * 0.01 + k * 0.002
        ended = torch.tensor([k % 5 == 0, False, k % 3 == 0, False])
        sampler.record(
            frames[region],
            rewards[region],
            ended[region],
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
        )
    return {
        "wm": dynamics.state_dict(),
        "wm_optimizers": [o.state_dict() for o in wm.optimizers],
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizers": [o.state_dict() for o in policy.actor_optimizers],
        "critic_optimizers": [o.state_dict() for o in policy.critic_optimizers],
        "normalizer": normalizer.state_dict(),
        "sampler": {key: getattr(sampler, key) for key in ("value", "count", "effort")},
    }


def _init(rank, port):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        GLOO_SOCKET_IFNAME="lo",
    )


def _math_worker(rank, directory, port):
    _init(rank, port)
    group = Distributed.initialize("cpu")
    try:
        assert not group.all_ready(rank == 0)
        assert group.all_ready(True)
        assert group.any(rank == 1)
        torch.save(_math(rank), Path(directory) / f"rank{rank}.pt")
        params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))]
        params[rank].grad = torch.ones(2)
        with pytest.raises(RuntimeError, match="layout differs"):
            GradientReducer()(params)
        with pytest.raises(RuntimeError, match="deliberate I/O failure"):
            group.primary_call(
                lambda: (_ for _ in ()).throw(OSError("deliberate I/O failure"))
            )
    finally:
        group.close()


def _spawn(worker, directory, *args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    children = mp.spawn(
        worker, args=(str(directory), port, *args), nprocs=2, join=False
    )
    deadline = time.monotonic() + 150
    try:
        while not children.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("official distributed workers timed out")
    finally:
        for child in children.processes:
            if child.is_alive():
                child.terminate()
                child.join(timeout=3)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=3)


def test_global_wm_actor_critic_optimizers_normalizers_and_sampler_match_serial(
    tmp_path,
):
    serial = _math()
    _spawn(_math_worker, tmp_path)
    first = torch.load(tmp_path / "rank0.pt", weights_only=True)
    _tree_close(first, torch.load(tmp_path / "rank1.pt", weights_only=True), exact=True)
    # Muon orthogonalizes in BF16: FP32 reduction order can change a BF16
    # rounding boundary on the second update (observed maximum < 8e-6).
    _tree_close(first, serial)


def test_global_batch_split_preserves_ratio_and_learning_rates():
    cfg = Config(device="cuda:0")
    local = resolve_config(cfg, 4, 2)
    assert (local.num_envs, local.wm.batch_size, local.policy.batch_size) == (
        1024,
        512,
        1024,
    )
    assert local.device == "cuda:2"
    assert local.num_envs / local.wm.batch_size == cfg.num_envs / cfg.wm.batch_size
    assert local.wm.lr == cfg.wm.lr and local.policy.lr == cfg.policy.lr
    assert cfg.num_envs == 4096
    per_rank = resolve_config(replace(cfg, batch_scope="per_rank"), 4)
    assert per_rank.num_envs == cfg.num_envs
    with pytest.raises(ValueError, match="divisible"):
        resolve_config(replace(cfg, num_envs=7), 2)
    with pytest.raises(ValueError, match="divisible by 4"):
        resolve_config(
            replace(cfg, policy=ActorCriticPolicyLearnerCfg(batch_size=6, horizon=3)), 2
        )
    with pytest.raises(ValueError, match="choose resume"):
        replace(cfg, resume=Path("a"), initialize_from=Path("b")).validate()


def _training_worker(rank, directory, port, mode, project, library):
    _init(rank, port)
    from gobot.rl.supermimic import training
    from gobot.rl.supermimic.replay import ReplayBuffer

    root = Path(directory)
    cfg = Config(
        project=Path(project),
        library=Path(library),
        backend="mujoco-cpu",
        device="cpu",
        num_envs=4,
        steps=14,
        output=root / "run",
        buffer_size=20,
        replay_ratio=8.0,
        dynamics_hidden_dims=(24, 16, 8),
        actor_hidden_dims=(24, 16, 8),
        critic_hidden_dims=(16, 8),
        wm=WorldModelLearnerCfg(num_batches=2, batch_size=4),
        policy=ActorCriticPolicyLearnerCfg(batch_size=4, horizon=4),
        compile=False,
        capture_graphs=False,
        capture_updates=False,
        torch_threads=1,
        sampling_mode="start",
        reset_noise=False,
        log_interval=2,
        save_interval=12,
        tensorboard=False,
        heartbeat_seconds=0.5,
    )
    trainers = []
    original_run = training.Trainer.run

    def capture(self):
        trainers.append(self)
        return original_run(self)

    training.Trainer.run = capture
    original_sample = ReplayBuffer.sample
    calls = 0

    def sample(self, length, batch):
        nonlocal calls
        calls += 1
        if mode == "resume" and rank == 1 and calls == 1:
            raise RuntimeError("no valid sequences deliberate rank-local shortage")
        if mode == "signal" and rank == 1 and calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return original_sample(self, length, batch)

    ReplayBuffer.sample = sample
    if mode == "budget":
        cfg = replace(cfg, max_seconds=1e-9)
    result = training.run(cfg)
    assert not dist.is_initialized()
    assert result["env_transitions"] == result["control_steps"] * 4
    assert (
        result["stop_reason"]
        == {"resume": "steps", "signal": "signal", "budget": "time_budget"}[mode]
    )
    for name in ("actor", "critic", "dynamics"):
        torch.save(getattr(trainers[-1], name).state_dict(), root / f"{name}{rank}.pt")
    if mode == "resume":
        ck = torch.load(result["checkpoint"], weights_only=True)
        assert ck["counters"]["skipped_updates"] > 0
        resumed = training.run(
            replace(
                cfg,
                steps=24,
                output=root / "resumed",
                resume=Path(result["checkpoint"]),
            )
        )
        assert resumed["learner_updates"] > result["learner_updates"]
        loaded = torch.load(resumed["checkpoint"], weights_only=True)
        torch.testing.assert_close(
            ck["rank_states"][rank]["context"], loaded["rank_states"][rank]["context"]
        )
        assert trainers[-1].dynamics.state_normalizer.count == 24 * 4
        for name in ("actor", "critic", "dynamics"):
            torch.save(
                getattr(trainers[-1], name).state_dict(),
                root / f"resumed_{name}{rank}.pt",
            )


@pytest.mark.parametrize("mode", ["resume", "signal", "budget"])
def test_real_two_rank_training(tmp_path, mode):
    project = os.environ.get("GOBOT_SUPERMIMIC_TEST_PROJECT")
    library = os.environ.get("GOBOT_SUPERMIMIC_TEST_LIBRARY")
    if not project or not library:
        pytest.skip("requires audited official scene and packed motions")
    _spawn(_training_worker, tmp_path, mode, project, library)
    for name in ("actor", "critic", "dynamics"):
        _tree_close(
            torch.load(tmp_path / f"{name}0.pt", weights_only=True),
            torch.load(tmp_path / f"{name}1.pt", weights_only=True),
            exact=True,
        )
        if mode == "resume":
            _tree_close(
                torch.load(tmp_path / f"resumed_{name}0.pt", weights_only=True),
                torch.load(tmp_path / f"resumed_{name}1.pt", weights_only=True),
                exact=True,
            )
    ck = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert len(ck["rank_states"]) == 2
    assert not torch.equal(
        ck["rank_states"][0]["context"], ck["rank_states"][1]["context"]
    )
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["phase"] == "finished"
    assert status["control_steps"] == ck["counters"]["control_steps"]
    if mode == "resume":
        rows = [
            json.loads(line)
            for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
        ]
        assert [row["control_steps"] for row in rows] == list(range(2, 15, 2))
        assert rows[-1]["env_transitions"] == 56
