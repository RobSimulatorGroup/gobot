"""Video previews retain failures and use explicit fixed motion windows."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
pytest.importorskip("unisupermimic", reason="Run with the isolated SuperMimic reference interpreter")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.locomotion.supermimic_video import CLIPS, CLIP_START_SECONDS, collect_clips, latest_checkpoint
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY


def test_select_atomic_numbered_checkpoint_and_final(tmp_path):
    for name in ('ckpt_000250.pt', 'ckpt_000500.pt', 'last.pt', 'ckpt_000750.pt.tmp'):
        (tmp_path / name).touch()
    assert latest_checkpoint(tmp_path, {'status': 'training'}) == (tmp_path / 'ckpt_000500.pt', 500)
    assert latest_checkpoint(tmp_path, {'status': 'finished', 'iteration': 612}) == (tmp_path / 'last.pt', 612)


def test_video_keeps_pre_reset_failure_and_fixed_start():
    count = 4000
    motion = SimpleNamespace(names=list(CLIPS), clip_start=torch.tensor([0, count, 2 * count]),
        clip_len=torch.full((3,), count), root_p=torch.zeros(3 * count, 3),
        root_R=torch.eye(3).expand(3 * count, 3, 3), q=torch.zeros(3 * count, 29))
    mdp = SimpleNamespace(motion=motion, obs=lambda *args: torch.zeros(3, 4),
                          goal=lambda frames, length: torch.zeros(3, 4))

    class Env:
        supermimic_mdp = mdp
        steps = 0
        def reset_to_frames(self, frames):
            self.first_frames = frames.clone()
            self.current = {'s': torch.zeros(3, 71), 'f': frames, 'a_prev': torch.zeros(3, 29),
                            'xi': torch.zeros(3, 33)}
            self.current['s'][:, 3] = 1
            return {}, {SUPERMIMIC_STATE_KEY: self.current}
        def step(self, action):
            self.steps += 1
            pre_reset = self.current['s'].clone()
            pre_reset[:, 0] = 123 + self.steps
            transition = {'s1': pre_reset, 'f': self.current['f'].clone(),
                          'term': torch.tensor([True, False, False]),
                          'done': torch.tensor([True, False, False])}
            self.current['f'] += 1
            self.current['s'][:, 0] = 999  # This reset state must not appear in video.
            return SimpleNamespace(info={SUPERMIMIC_STATE_KEY: self.current,
                                          SUPERMIMIC_TRANSITION_KEY: transition})
    env = Env()
    with patch('unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate.load_policy',
               return_value=lambda *args: torch.zeros(3, 29)):
        records = collect_clips(env, {'args': {'hist': 1, 'goal_frames': 1}}, .04)
    expected = motion.clip_start + torch.tensor([round(CLIP_START_SECONDS[name] * 50) for name in CLIPS])
    torch.testing.assert_close(env.first_frames, expected)
    first = records[CLIPS[0]]
    assert first['failed'] and first['done']
    assert len(first['policy']) == 2 and first['policy'][-1, 0] == 124
    assert first['simulated_seconds'] == .02
    assert len(records[CLIPS[1]]['policy']) == 3
