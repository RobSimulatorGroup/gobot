import argparse
from dataclasses import dataclass

from gobot_gym_adapter import GobotGymEnv


@dataclass
class PPOConfig:
    total_steps: int = 4096
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 1


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "gobot_ppo requires PyTorch. Install torch in the active Python environment "
            "before running PPO training."
        ) from error
    return torch, nn


class ActorCritic:
    def __init__(self, observation_size, action_size, hidden_size=128):
        torch, nn = _require_torch()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(observation_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, action_size),
                )
                self.critic = nn.Sequential(
                    nn.Linear(observation_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                )
                self.log_std = nn.Parameter(torch.zeros(action_size))

            def forward(self, observations):
                mean = self.actor(observations)
                std = self.log_std.exp().expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                value = self.critic(observations).squeeze(-1)
                return dist, value

        self.model = Model()


class PPORunner:
    def __init__(self, env, config=None, device="cpu"):
        self.config = config or PPOConfig()
        self.env = env
        self.device = device
        self.torch, _ = _require_torch()
        self.torch.manual_seed(self.config.seed)

        observation_size = int(env.env.get_observation_size())
        action_size = int(env.env.get_action_size())
        if observation_size <= 0 or action_size <= 0:
            observation, info = env.reset(seed=self.config.seed)
            if not info.get("ok", True):
                raise RuntimeError(info.get("error", "failed to reset Gobot environment"))
            observation_size = len(observation)
            action_size = int(env.env.get_action_size())
        if observation_size <= 0 or action_size <= 0:
            raise RuntimeError("Gobot PPO requires non-empty observation and action spaces.")

        self.agent = ActorCritic(observation_size, action_size).model.to(device)
        self.optimizer = self.torch.optim.Adam(self.agent.parameters(), lr=self.config.learning_rate)

    def train(self):
        torch = self.torch
        cfg = self.config
        observation, info = self.env.reset(seed=cfg.seed)
        if not info.get("ok", True):
            raise RuntimeError(info.get("error", "failed to reset Gobot environment"))

        obs = torch.tensor(observation, dtype=torch.float32, device=self.device)
        completed_steps = 0
        episode_return = 0.0
        episode_count = 0
        last_loss = 0.0

        while completed_steps < cfg.total_steps:
            rollout = self._collect_rollout(obs)
            obs = rollout["next_obs"]
            episode_return += float(rollout["reward_sum"])
            episode_count += int(rollout["episode_count"])
            completed_steps += len(rollout["rewards"])
            last_loss = self._update(rollout)
            if completed_steps % max(cfg.rollout_steps, 1) == 0:
                mean_reward = episode_return / max(episode_count, 1)
                print(f"steps={completed_steps} episodes={episode_count} mean_reward={mean_reward:.6f} loss={last_loss:.6f}")

        return {"steps": completed_steps, "episodes": episode_count, "last_loss": last_loss}

    def _collect_rollout(self, obs):
        torch = self.torch
        cfg = self.config
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        reward_sum = 0.0
        episode_count = 0

        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                dist, value = self.agent(obs.unsqueeze(0))
                action = dist.sample()
                clipped_action = action.clamp(-1.0, 1.0)
                log_prob = dist.log_prob(action).sum(-1)

            next_observation, reward, terminated, truncated, info = self.env.step(
                clipped_action.squeeze(0).cpu().tolist()
            )
            if info.get("error"):
                raise RuntimeError(info["error"])

            done = bool(terminated or truncated)
            observations.append(obs)
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            rewards.append(float(reward))
            dones.append(done)
            values.append(value.squeeze(0))
            reward_sum += float(reward)

            if done:
                next_observation, reset_info = self.env.reset()
                if not reset_info.get("ok", True):
                    raise RuntimeError(reset_info.get("error", "failed to reset Gobot environment"))
                episode_count += 1
            obs = torch.tensor(next_observation, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_value = self.agent(obs.unsqueeze(0))

        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
            "values": torch.stack(values),
            "next_value": next_value.squeeze(0),
            "next_obs": obs,
            "reward_sum": reward_sum,
            "episode_count": episode_count,
        }

    def _update(self, rollout):
        torch = self.torch
        cfg = self.config
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"].detach()
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        next_value = rollout["next_value"].detach()

        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            next_values = next_value if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + cfg.gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values

        observations = rollout["observations"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"].detach()
        batch_size = len(rewards)
        minibatch_size = min(cfg.minibatch_size, batch_size)
        indices = torch.arange(batch_size, device=self.device)
        last_loss = 0.0

        for _ in range(cfg.update_epochs):
            shuffled = indices[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, minibatch_size):
                minibatch = shuffled[start:start + minibatch_size]
                dist, new_value = self.agent(observations[minibatch])
                new_log_prob = dist.log_prob(actions[minibatch]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = new_log_prob - old_log_probs[minibatch]
                ratio = log_ratio.exp()
                mb_advantages = advantages[minibatch]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)
                policy_loss = -torch.min(
                    mb_advantages * ratio,
                    mb_advantages * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()
                value_loss = 0.5 * (returns[minibatch] - new_value).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                last_loss = float(loss.detach().cpu())

        return last_loss


def train(scene_path="", robot="robot", backend="null", project_path=None, config=None, device="cpu"):
    if project_path:
        import gobot

        gobot.set_project_path(project_path)
    env = GobotGymEnv(scene_path=scene_path, robot=robot, backend=backend)
    runner = PPORunner(env, config=config, device=device)
    return runner.train()


def main():
    parser = argparse.ArgumentParser(description="Train a single Gobot RL environment with PPO.")
    parser.add_argument("--project", default=None)
    parser.add_argument("--scene", default="")
    parser.add_argument("--robot", default="robot")
    parser.add_argument("--backend", default="null")
    parser.add_argument("--total-steps", type=int, default=PPOConfig.total_steps)
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps)
    parser.add_argument("--seed", type=int, default=PPOConfig.seed)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = PPOConfig(total_steps=args.total_steps, rollout_steps=args.rollout_steps, seed=args.seed)
    result = train(
        scene_path=args.scene,
        robot=args.robot,
        backend=args.backend,
        project_path=args.project,
        config=config,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
