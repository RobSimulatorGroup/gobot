"""Exercise the actual PPO loop across two Gloo workers without requiring GPUs."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

from gobot.rl.locomotion.distributed import DistributedTraining


def _worker(rank, directory, ports, mode):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(ports[0]), GLOO_SOCKET_IFNAME='lo',
                      OMP_NUM_THREADS='1')
    from gobot.rl.locomotion import training
    from gobot.rl.locomotion.experiment import Experiment
    root = Path(directory)
    algorithms = []
    seeds = []
    before_update = []
    build = training.build_algorithm
    make = training.make_velocity_env
    update = training.update_policy

    def capture_algorithm(*args):
        alg = build(*args)
        algorithms.append(alg)
        return alg

    def capture_environment(*args, **kwargs):
        seeds.append(kwargs['seed'])
        env = make(*args, **kwargs)
        if mode == 'rollout_error' and rank == 1:
            step = env.step
            calls = 0

            def fail_after_checkpoint(action):
                nonlocal calls
                calls += 1
                state = step(action)
                if calls == 5:
                    state.obs['actor'][0, 0] = float('nan')
                return state
            env.step = fail_after_checkpoint
        return env

    def capture_update(alg, env, observation, *args):
        before_update.append((env.training_state_dict()['common_step_counter'], alg.learning_rate))
        result = update(alg, env, observation, *args)
        if mode == 'signal' and rank == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    training.build_algorithm = capture_algorithm
    training.make_velocity_env = capture_environment
    training.update_policy = capture_update
    exp = Experiment(robot='go1', stage='flat', num_envs=4, iterations=2,
                     save_interval=1, eval_interval=2, eval_steps=4, eval_envs=2,
                     hours=1.e-12 if mode == 'budget' else .1)
    config = training.agent_config(exp)
    config['actor']['hidden_dims'] = config['critic']['hidden_dims'] = [16, 8]
    config['num_steps_per_env'] = 4
    config['algorithm'].update(num_learning_epochs=2, num_mini_batches=2)
    run_dir = root / 'run'

    if mode in {'error', 'rollout_error'}:
        try:
            training.run(exp, run_dir, backend='mujoco-cpu', no_eval=True, agent_cfg=config)
        except (ValueError, RuntimeError) as error:
            expected = 'output directory is not empty' if mode == 'error' else "rollout failed: rank 1: ValueError: The observation group 'actor'"
            assert expected in str(error)
        else:
            raise AssertionError('all workers must receive the output error')
        assert not torch.distributed.is_initialized()
        return

    summary = training.run(exp, run_dir, backend='mujoco-cpu', no_eval=mode != 'eval', agent_cfg=config)
    assert summary['world_size'] == 2
    assert summary['total_num_envs'] == 8
    assert not torch.distributed.is_initialized()
    torch.save(algorithms[-1].save(), root / f'algorithm_{rank}.pt')
    saved = torch.load(run_dir / 'model_final.pt', weights_only=False)
    assert saved['infos']['rank_states'][rank]['env_state']['common_step_counter'] == summary['completed_updates'] * 4
    if mode == 'resume':
        initial_lr = algorithms[-1].learning_rate
        os.environ['MASTER_PORT'] = str(ports[1])
        resumed = training.run(replace(exp, iterations=1), root / 'resumed', backend='mujoco-cpu',
                               no_eval=True, resume=run_dir / 'model_final.pt', agent_cfg=config)
        assert before_update[-1] == (8, initial_lr)
        assert resumed['last_iteration'] == 2
        torch.save(algorithms[-1].save(), root / f'resumed_algorithm_{rank}.pt')
        # A changed batch shape cannot reuse per-rank curricula and optimizer progress.
        os.environ['MASTER_PORT'] = str(ports[2])
        try:
            training.run(replace(exp, num_envs=2), root / 'wrong_shape', backend='mujoco-cpu',
                         no_eval=True, resume=run_dir / 'model_final.pt', agent_cfg=config)
        except ValueError as error:
            assert 'same world size and num_envs' in str(error)
        else:
            raise AssertionError('changing num_envs must require initialize')
    assert seeds and all(seed == exp.seed + rank for seed in seeds)


def _compare(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _compare(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _compare(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('mode', ['resume', 'signal', 'budget', 'error', 'eval', 'rollout_error'])
def test_two_worker_ppo(tmp_path, mode):
    if mode == 'error':
        (tmp_path / 'run').mkdir()
        (tmp_path / 'run' / 'keep.txt').write_text('existing run')
    sockets = [socket.socket() for _ in range(3)]
    try:
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        ports = [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    processes = mp.spawn(_worker, args=(str(tmp_path), ports, mode), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('distributed training did not finish within two minutes')
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
    if mode == 'error':
        assert sorted(p.name for p in (tmp_path / 'run').iterdir()) == ['keep.txt']
        return
    if mode == 'rollout_error':
        status = json.loads((tmp_path / 'run' / 'status.json').read_text())
        assert status['state'] == 'failed'
        assert 'rank 1' in status['error'] and 'actor' in status['error']
        checkpoint = torch.load(tmp_path / 'run' / 'latest.pt', weights_only=False)
        assert checkpoint['iter'] == 0
        assert not (tmp_path / 'run' / 'model_final.pt').exists()
        for role in ('actor', 'critic'):
            assert all(torch.isfinite(value).all() for value in checkpoint[f'{role}_state_dict'].values())
        budget = json.loads((tmp_path / 'run' / 'gpu_budget.json').read_text())
        assert budget.get('active_run') is None
        return
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    expected = {'resume': (2, 'iterations'), 'signal': (1, 'signal'),
                'budget': (0, 'budget'), 'eval': (2, 'iterations')}[mode]
    assert (summary['completed_updates'], summary['stop_reason']) == expected
    lines = (tmp_path / 'run' / 'metrics.jsonl').read_text().splitlines()
    assert len(lines) == expected[0]  # Only rank 0 writes metrics.
    for index, line in enumerate(lines):
        metrics = json.loads(line)
        assert metrics['total_steps'] == (index + 1) * 8 * 4
        assert metrics['collection_s'] > 0 and metrics['learning_s'] > 0
        assert 'Episode_Reward/track_linear_velocity' in metrics['log']
        assert sum(value for key, value in metrics['log'].items()
                   if key.startswith('Episode_Termination/')) == metrics['episodes_completed']
    checkpoint = torch.load(tmp_path / 'run' / 'model_final.pt', weights_only=False)
    assert len(checkpoint['infos']['rank_states']) == 2
    states = checkpoint['infos']['rank_states']
    assert not torch.equal(states[0]['rng']['torch'], states[1]['rng']['torch'])
    alg = torch.load(tmp_path / 'algorithm_0.pt', weights_only=False)
    _compare(alg, torch.load(tmp_path / 'algorithm_1.pt', weights_only=False))
    _compare(alg['actor_state_dict'], checkpoint['actor_state_dict'])
    assert alg['actor_state_dict']['obs_normalizer.count'].item() == expected[0] * 2 * 4 * 4
    ledger = json.loads((tmp_path / 'run' / 'gpu_budget.json').read_text())
    assert len(ledger['runs']) == 1
    assert ledger['runs'][0]['allocated_gpus'] == 2
    assert ledger['seconds'] >= summary['elapsed_s'] * 2
    if mode == 'resume':
        _compare(torch.load(tmp_path / 'resumed_algorithm_0.pt', weights_only=False),
                 torch.load(tmp_path / 'resumed_algorithm_1.pt', weights_only=False))
    if mode == 'eval':
        assert (tmp_path / 'run' / 'eval_000001.json').is_file()
        assert (tmp_path / 'run' / 'best.pt').is_file()


def test_torchrun_selects_local_cuda_device(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setenv('RANK', '6')
    monkeypatch.setenv('LOCAL_RANK', '6')
    worker = DistributedTraining.from_environment('cuda:0')
    assert worker.device == 'cuda:6'
    assert worker.algorithm_config == {'global_rank': 6, 'local_rank': 6, 'world_size': 8}
