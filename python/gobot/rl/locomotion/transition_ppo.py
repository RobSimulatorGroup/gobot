"""PPO with final-state timeout bootstrap, independent of any motion dataset."""
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.utils import check_nan


class TransitionPPO(PPO):
    def __init__(self, *args, fall_penalty=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.fall_penalty = float(fall_penalty)

    def process_env_step(self, obs, rewards, dones, extras):
        timeouts = extras.get('time_outs', torch.zeros_like(dones)).bool()
        terminated = extras.get('terminated')
        if terminated is None:
            terminated = dones & ~timeouts
        combined = rewards - self.fall_penalty * terminated.bool()
        with torch.no_grad():
            if bool(timeouts.any()):
                if 'final_observation' not in extras:
                    raise ValueError('timeout bootstrap requires pre-reset final_observation')
                final = TensorDict(extras['final_observation'], batch_size=[len(rewards)], device=self.device)[timeouts]
                # Fallen worlds may already have reset to valid observations.
                # Their unused final states must not enter the critic: NaN * 0
                # would otherwise contaminate the stored transition reward.
                check_nan(final, combined[timeouts], dones[timeouts])
                combined[timeouts] += self.gamma * self.critic(final).squeeze(-1)
        super().process_env_step(obs, combined, dones, {k: v for k, v in extras.items() if k != 'time_outs'})
