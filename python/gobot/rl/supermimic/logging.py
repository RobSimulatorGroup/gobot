"""Headless JSON/TensorBoard logging; no simulator or terminal-UI dependency."""
import json
from pathlib import Path


class SuperMimicLogger:
    TAGS = {'wm_loss': 'Loss/world_model', 'wm_err8': 'Loss/world_model_err8',
            'pi_loss': 'Loss/policy', 'v_loss': 'Loss/value', 'imag_rew': 'Loss/imagined_reward',
            'imag_survival': 'Loss/imagined_survival', 'pi_gradnorm': 'Train/grad_norm',
            'lr': 'Train/learning_rate', 'sim_rew': 'Reward/sim_step_norm',
            'ep_len': 'Episode/length', 'ep_ret_norm': 'Episode/return_norm',
            'term_frac': 'Episode/terminated_frac', 'iteration_time': 'Perf/iteration_time',
            'collect_time': 'Perf/collect_time', 'learn_time': 'Perf/learning_time'}

    def __init__(self, *, log_dir, log_backend='tensorboard', **metadata):
        self.backend = log_backend
        self.writer = None
        self.stream = (Path(log_dir)/'metrics.jsonl').open('a', buffering=1)
        if log_backend == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)

    def log_status(self, status):
        if self.backend != 'no_print':
            print(status, flush=True)

    def start(self, status='Starting'):
        self.log_status(status)

    def log_buffer_fill(self, count, capacity):
        self.log_status(f'Replay: {count}/{capacity} control steps')

    def start_training_timer(self):
        pass

    def log_step(self, iteration, **metrics):
        self.stream.write(json.dumps({'iteration': iteration, **metrics}, allow_nan=False)+'\n')
        if self.writer:
            for name, tag in self.TAGS.items():
                if metrics.get(name) is not None:
                    self.writer.add_scalar(tag, metrics[name], iteration)
            duration = max(metrics['iteration_time'], 1.e-9)
            self.writer.add_scalar('Perf/it_per_s', 1/duration, iteration)
            self.writer.add_scalar('Perf/sim_steps_per_s', metrics['iter_sim_steps']/duration, iteration)

    def log_save(self, path):
        self.log_status(f'Saved {path}')

    def finish(self):
        self.close()

    def close(self):
        if self.writer:
            self.writer.close(); self.writer = None
        if not self.stream.closed:
            self.stream.close()
