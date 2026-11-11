Motion clip interpolation in rotations.py adapts the small-angle convention from
https://github.com/xbpeng/MimicKit, commit fbe1a36e878f036d6bf5521defb8eb3b8ab4170d.
Copyright 2025 Xue Bin Peng. Apache-2.0; see LICENSE-MimicKit.

The feature layout and BCE motion discriminator follow the reference G1 task.
PPO, rollout storage, policy normalization, optimizer, runner and training logger
are provided by the installed RSL-RL package through Gobot's existing integration.
No MimicKit simulator, PPO implementation, logger or Gymnasium dependency is used.
