"""Training statistics and console output, independent of PPO and physics state."""
from datetime import timedelta

import torch


def step_statistics(xp, *, momentum, foot_velocity, contact, first_contact, peak_height,
                    foot_force, linear_velocity, angular_velocity, command):
    """Per-world sums and counts, sampled before reset with the reward state.

    Columns: angular momentum, landing peak height, contact slip speed, landing
    force, landing count, contact count, velocity error, yaw error, sample count.
    Supports NumPy and Torch; no device-to-host transfers.
    """
    norm = lambda x: (x * x).sum(-1) ** .5
    contacts = contact > 0
    landings = first_contact > 0
    angular = norm(momentum)
    return xp.stack((angular, (peak_height * landings).sum(-1),
                     (norm(foot_velocity[..., :2]) * contacts).sum(-1),
                     (norm(foot_force) * landings).sum(-1),
                     landings.sum(-1), contacts.sum(-1),
                     norm(linear_velocity[:, :2] - command[:, :2]),
                     abs(angular_velocity[:, 2] - command[:, 2]),
                     xp.ones_like(angular)), 1)


class TrainingStatistics:
    """Accumulate episode statistics across rollouts, then reduce sums across ranks."""

    def __init__(self, count, actions, reward_names, cfg, device):
        self.reward_names = tuple(reward_names)
        self.reward_enabled = [float(getattr(cfg.rewards, name)) != 0 for name in reward_names]
        self.dt = cfg.physics_dt * cfg.decimation
        self.episode_seconds = cfg.episode_length_s
        self.action_clip = cfg.action_clip
        # Return, step count, weighted reward integrals, action acceleration, velocity/yaw error.
        self.episodes = torch.zeros(count, 2 + len(reward_names) + 3, device=device)
        self.previous_action = torch.zeros(count, actions, device=device)
        self.previous_previous_action = torch.zeros_like(self.previous_action)
        self.completed = torch.zeros(self.episodes.shape[1], device=device)
        self.step_sums = torch.zeros(9, device=device)
        self.terminations = torch.zeros(5, device=device)
        self.begin_update()

    def begin_update(self):
        self.completed.zero_()
        self.step_sums.zero_()
        self.terminations.zero_()

    def record(self, action, reward, done, stats):
        action = action.detach()
        if self.action_clip is not None:
            action = action.clamp(-self.action_clip, self.action_clip)
        acceleration = (action - 2 * self.previous_action + self.previous_previous_action).abs().mean(-1)
        self.previous_previous_action.copy_(self.previous_action)
        self.previous_action.copy_(action)
        self.episodes[:, 0].add_(reward)
        self.episodes[:, 1].add_(1)
        self.episodes[:, 2:-3].add_(stats['reward_terms'] * self.dt)
        self.episodes[:, -3].add_(acceleration)
        self.episodes[:, -2:].add_(stats['metrics'][:, 6:8])
        self.step_sums.add_(stats['metrics'].sum(0))
        finished = self.episodes[done].clone()
        finished[:, -3:] /= finished[:, 1:2].clamp(min=1)
        self.completed.add_(finished.sum(0))
        reasons = stats['reset_reason']
        # First entry is the total; the remaining entries are mutually exclusive reasons.
        self.terminations.add_(torch.stack((done.sum(), *((done & (reasons == code)).sum() for code in range(1, 5)))))
        self.episodes[done] = 0
        self.previous_action[done] = 0
        self.previous_previous_action[done] = 0

    def summarize(self, distributed, env):
        device = self.episodes.device
        levels, types = (torch.as_tensor(a, device=device) for a in env.terrain_assignments)
        names = env.terrain_type_names
        terrain_sums = torch.zeros(len(names), device=device).scatter_add_(0, types.long(), levels.float())
        terrain_counts = torch.bincount(types.long(), minlength=len(names))
        packed = torch.cat((self.completed, self.step_sums, self.terminations, terrain_sums, terrain_counts))
        values = distributed.reduce(packed)
        end = len(self.completed)
        completed, steps, terminations = values[:end], values[end:end+9], values[end+9:end+14]
        sums, counts = values[end+14:end+14+len(names)], values[end+14+len(names):]
        episodes = int(terminations[0])
        mean = lambda value, count: value / count if count else None
        log = {
            'Metrics/angular_momentum_mean': mean(steps[0], steps[8]),
            'Metrics/peak_height_mean': mean(steps[1], steps[4]),
            'Metrics/slip_velocity_mean': mean(steps[2], steps[5]),
            'Metrics/landing_force_mean': mean(steps[3], steps[4]),
        }
        for i, name in enumerate(self.reward_names):
            if self.reward_enabled[i]:
                log[f'Episode_Reward/{name}'] = mean(completed[2+i], episodes * self.episode_seconds)
        log['Episode_Metrics/mean_action_acc'] = mean(completed[-3], episodes)
        log['Curriculum/terrain_levels/mean'] = mean(sum(sums), sum(counts))
        log['Curriculum/terrain_levels/max'] = distributed.reduce([float(levels.max())], maximum=True)[0]
        for name, total, count in zip(names, sums, counts, strict=True):
            log[f'Curriculum/terrain_levels/{name}'] = mean(total, count)
        ranges = env.cfg_obj.command.ranges
        for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
            low, high = getattr(ranges, name)
            log[f'Curriculum/command_vel/{name}_min'] = low
            log[f'Curriculum/command_vel/{name}_max'] = high
        log['Metrics/twist/error_vel_xy'] = mean(completed[-2], episodes)
        log['Metrics/twist/error_vel_yaw'] = mean(completed[-1], episodes)
        for code, name in ((2, 'time_out'), (1, 'fell_over'), (3, 'out_of_terrain_bounds'), (4, 'numerical_failure')):
            log[f'Episode_Termination/{name}'] = int(terminations[code])
        return {'episode_reward': mean(completed[0], episodes),
                'episode_steps': mean(completed[1], episodes), 'episodes_completed': episodes,
                'terrain_level': log['Curriculum/terrain_levels/mean'], 'log': log}


def format_training_log(metrics, *, target_iteration, remaining_iterations, average_iteration_s):
    """Render one global PPO update. Iteration numbering matches saved checkpoints."""
    rows = []

    def add(label, value, spec='.4f', suffix=''):
        rows.append((label, '—' if value is None else format(value, spec) + suffix))

    add('Total steps', metrics['total_steps'], ',d')
    add('Steps per second', metrics['steps_per_second'], '.0f')
    add('Collection time', metrics['collection_s'], '.3f', 's')
    add('Learning time', metrics['learning_s'], '.3f', 's')
    for key, label in (('value', 'Mean value loss'), ('surrogate', 'Mean surrogate loss'),
                       ('entropy', 'Mean entropy'), ('episode_reward', 'Mean reward'),
                       ('episode_steps', 'Mean episode length (steps)'), ('action_std', 'Mean action std')):
        add(label, metrics.get(key))
    add('Learning rate', metrics['learning_rate'], '.6g')
    add('GPUs', metrics['world_size'], 'd')
    add('Environments', metrics['total_num_envs'], ',d')
    add('Completed episodes', metrics['episodes_completed'], 'd')
    for key, value in metrics['log'].items():
        suffix = {'Metrics/twist/error_vel_xy': ' m/s', 'Metrics/twist/error_vel_yaw': ' rad/s'}.get(key, '')
        if key.startswith('Episode_Termination/'):
            suffix = ' episodes/update'
        add(key, value, 'd' if key.startswith('Episode_Termination/') else '.4f', suffix)
    add('Action clip fraction', metrics['action_clip_fraction'])
    add('Joint effort saturation fraction', metrics['joint_effort_saturation_fraction'])
    width = max(40, *(len(label) for label, _ in rows))
    line_width = max(80, width + 22)
    lines = ['#' * line_width,
             f"Learning iteration {metrics['iteration']}/{target_iteration}".center(line_width), '']
    lines.extend(f'{label:>{width}}: {value}' for label, value in rows)
    lines.append('-' * line_width)
    lines.append(f"{'Iteration time':>{width}}: {metrics['iteration_s']:.2f}s")
    lines.append(f"{'Time elapsed':>{width}}: {timedelta(seconds=int(metrics['elapsed_s']))}")
    eta = int(average_iteration_s * remaining_iterations)
    lines.append(f"{'ETA':>{width}}: {timedelta(seconds=eta)}")
    return '\n'.join(lines)
