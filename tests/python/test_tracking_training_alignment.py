"""Compare Gobot's actual learn loop to the installed upstream RSL-RL runner."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gobot.rl.tracking.distributed import DistributedTraining, cpu_state
from gobot.rl.tracking.training import Runner, default_agent


class RecordedEnv:
    num_envs, num_actions, max_episode_length, device = 8, 2, 5, 'cpu'
    cfg = {}
    cfg_obj = SimpleNamespace(reward_weights={}, episode_length_s=.1)
    sampler = SimpleNamespace(synchronize=lambda group: None)

    def __init__(self):
        g = torch.Generator().manual_seed(123)
        self.frames = [TensorDict({'actor':torch.randn(8,4,generator=g),
                                  'critic':torch.randn(8,5,generator=g)}, batch_size=[8]) for _ in range(9)]
        self.rewards = [torch.randn(8,generator=g) for _ in range(8)]
        self.episode_length_buf = torch.zeros(8,dtype=torch.long)
        self.index = 0
        self.actions = []

    def get_observations(self):
        return self.frames[self.index].clone()

    def step(self, action):
        self.actions.append(action.clone())
        reward = self.rewards[self.index].clone()
        self.index += 1
        # Independent physical termination and timeout, including an overlap.
        done = torch.tensor([True,True,False,False,True,False,False,False])
        timeout = torch.tensor([False,True,False,False,True,False,False,False])
        return self.get_observations(), reward, done, {'time_outs':timeout}

    def consume_episode_statistics(self):
        return [0.,0.]


def execute(factory, output):
    env = RecordedEnv()
    cfg = default_agent()
    cfg['actor']['hidden_dims'] = [16,16]
    cfg['critic']['hidden_dims'] = [16,16]
    cfg['num_steps_per_env'] = 4
    torch.manual_seed(42)
    runner = factory(env, deepcopy(cfg), output)
    runner.save = lambda *a,**k: None
    runner.logger.init_logging_writer = lambda: None
    runner.logger.log = lambda **kw: None
    gradient, returns = [], []
    original_step, original_returns = runner.alg.optimizer.step, runner.alg.compute_returns
    def step(*a,**kw):
        gradient.append([p.grad.clone() for p in list(runner.alg.actor.parameters())+list(runner.alg.critic.parameters())])
        return original_step(*a,**kw)
    def compute(obs):
        original_returns(obs)
        returns.append(cpu_state(dict(rewards=runner.alg.storage.rewards, returns=runner.alg.storage.returns,
                                      advantages=runner.alg.storage.advantages)))
    runner.alg.optimizer.step = step
    runner.alg.compute_returns = compute
    runner.learn(2,init_at_random_ep_len=True)
    return dict(models=cpu_state(runner.alg.save()), gradients=gradient, targets=returns,
                actions=env.actions, rng=torch.get_rng_state(), clock=env.episode_length_buf)


def test_recorded_rollouts_produce_identical_gae_gradients_optimizer_and_normalizers(tmp_path):
    torch.set_num_threads(1)
    group = DistributedTraining('cpu')
    native = execute(lambda env,cfg,out: OnPolicyRunner(env,cfg,str(out),'cpu'),tmp_path/'native')
    (tmp_path/'gobot').mkdir()
    gobot = execute(lambda env,cfg,out: Runner(env,cfg,out,'cpu','probe',{},group),tmp_path/'gobot')
    def exact(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:exact(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):exact(x,y)
        else:assert a==b
    exact(native,gobot)
    assert len(native['gradients'])==40  # two updates x five epochs x four minibatches
    assert int(native['models']['actor_state_dict']['obs_normalizer.count'])==64


def test_maxforce_self_contact_reduction_is_independent_of_pair_order():
    from gobot.rl.articulation import ArticulationBatch
    env = object.__new__(ArticulationBatch)
    env._closed=False;env._provider=None;env.num_envs=1;env.self_contact_threshold=10.
    env._self_contact_hits=torch.zeros(1,1);env._self_contact_link_ids=np.array([1,2])
    def sample(forces,mode):
        env.self_contact_reduction=mode
        env._raw_cpu={'contact_link_index':np.array([[[1,2],[1,2]]]),
                      'contact_count':np.array([2]),'contact_force':np.array([[ [0,0,f] for f in forces]])}
        env._sample_self_contacts(0)
        return env.read_self_contact_substeps().item()
    assert sample([0,20],'none')==0 and sample([20,0],'none')==1
    assert sample([0,20],'maxforce')==sample([20,0],'maxforce')==1
    assert sample([0,9],'maxforce')==0


def test_warp_determinism_flag_enables_all_bits_not_just_contacts():
    from enum import IntFlag
    from gobot.sim.providers.mujoco_warp import _determinism_option
    class Flags(IntFlag):
        CONTACTS=1;CONSTRAINT=2;ATOMICS=4;ALL=7
    runtime=SimpleNamespace(DeterminismType=Flags)
    assert _determinism_option(runtime,True)==7
    assert _determinism_option(runtime,False) is False
    assert _determinism_option(SimpleNamespace(),True) is True


def test_alignment_recipe_is_explicit_and_preserved_by_configuration(tmp_path):
    from gobot.rl.tracking.training import Config
    from gobot.rl.training.workflow import config_dict,update_config
    from gobot.rl.tracking.training import contract
    from test_tracking import tiny_config
    from gobot.rl.tracking import TrackingEnv
    from dataclasses import asdict,replace
    config=Config(**asdict(tiny_config(tmp_path)))
    assert config.self_contact_reduction=='none' and not config.deterministic_physics
    original=TrackingEnv(config)
    try:baseline=contract(original)
    finally:original.close()
    assert 'self_contact_reduction' not in baseline and 'deterministic_physics' not in baseline
    variant=replace(config,self_contact_reduction='maxforce')
    loaded=update_config(Config(),config_dict(variant))
    assert loaded.self_contact_reduction=='maxforce'
    env=TrackingEnv(loaded)
    try:assert contract(env)['self_contact_reduction']=='maxforce'
    finally:env.close()
