"""Compare Gobot's actual learn loop to the installed upstream RSL-RL runner."""
from copy import deepcopy
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gobot.rl.tracking.distributed import DistributedTraining, cpu_state
from gobot.rl.tracking.training import Runner, default_agent


class RecordedEnv:
    num_envs, num_actions, max_episode_length, device = 8, 2, 5, 'cpu'
    cfg = {}
    cfg_obj = SimpleNamespace(reward_weights={}, episode_length_s=.1)

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
