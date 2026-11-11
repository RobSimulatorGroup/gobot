# G1 AMP 对齐版本

`g1-motion-amp` 使用 Gobot 场景、`ArticulationBatch` 和 MuJoCo Warp 仿真，
复用 RSL-RL PPO、OnPolicyRunner 和 TensorBoard 日志。对齐配方是默认配置，
显式配置保存在 `rsl_reference_aligned.json`。

准备好的资源目录为 `examples/g1/assets/mimickit/gobot`：
`world.jscn` 是场景来源，`motion_amp/motions.npz` 是 walk/run 动作库，
`profile.json` 包含布局、控制参数和数据来源。部署时复制整个资源目录，保留网格和许可证。
运行训练和回放使用 Gobot 主环境 `.venv`，无需原 MimicKit 环境。

## 训练

在 118 的仓库目录执行：

```bash
cd /home/h3c-r5300-g6/wqq/gobot
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gobot.rl.train g1-motion-amp \
  --backend mujoco-warp --device cuda:0 --num-envs 4096 --seed 1 \
  --set updates=2000 --output logs/g1-motion-amp/walk_s1
```

输出目录必须不存在。`--dry-run` 检查配置；`--set 'motion="run"'` 切换跑步数据。
使用 `--config examples/g1/mimickit/rsl_reference_aligned.json` 可显式选择相同的对齐配置。

每轮 32 步，4096 环境 × 2000 次更新共 262144000 个样本。物理频率 240 Hz，控制频率 30 Hz。
Actor/critic 为 1024/512 ReLU，actor 237 维输入、29 个绝对关节角输出；
判别器读取 10 帧、2390 维动作历史。当前任务使用纯 AMP 风格奖励，没有速度跟踪奖励。

Actor 使用按关节动作范围缩放的固定标准差（归一化值 0.05）；PPO 使用 SGD、
momentum=0.9、固定学习率 1e-4。判别器使用 SGD、学习率 2.5e-4、batch 8192，
每轮更新 32 次，整轮结束后更新归一化统计。继续使用 RSL 的联合 actor/critic 更新，
没有额外复制 MimicKit PPO 或引入 Gymnasium。

## 保留的模型与回放

最终版本位于 `logs/g1-motion-amp/walk_s1_aligned_recover`：

- `model.pt`：118 上完整 checkpoint，可恢复训练；SHA256
  `10e2030f9eb5c6d468d4a9c28dc9c3c77e6f70d14b3e16531683f443b3a07d52`。
- `actor_eval.pt`：本机、jump-wqq 和 118 上的精简推理模型。
- `config.json`、`status.json`、`run.json`（118）：该训练的配置、完成状态和模型协议记录。
- `evaluation.json`：最终策略的评测数据；`walk.mp4`：该策略的单独回放。

```bash
.venv/bin/python -m gobot.rl.motion_amp.evaluate \
  --project examples/g1/assets/mimickit/gobot \
  --checkpoint logs/g1-motion-amp/walk_s1_aligned_recover/actor_eval.pt \
  --backend mujoco-warp --num-envs 64 \
  --output /tmp/g1_amp_evaluation.json --video /tmp/g1_amp_walk.mp4
```

评测 seed 1001/1002/1003，每个 64 个 10 秒首回合；两组匹配物理初态共 384/384 完成。
COM 配方初态下，前进速度中位数 0.971 m/s、末端横漂中位数 18.1 cm、回合平均骨盆倾角
中位数 3.43°。这只覆盖平地短回合，不代表推力或复杂地形鲁棒性。

## 恢复训练

完整 checkpoint 包含 actor、critic、优化器、判别器、replay 和 RNG。
它可独立恢复，不依赖已经删除的历史 `model_500.pt`；`config.json` 中的 resume 字段
是训练来源记录。默认每 100 次更新保存 checkpoint，单份完整模型约 2 GB。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gobot.rl.train g1-motion-amp \
  --backend mujoco-warp --device cuda:0 --num-envs 4096 --seed 1 \
  --resume logs/g1-motion-amp/walk_s1_aligned_recover/model.pt \
  --set updates=1000 --output logs/g1-motion-amp/walk_continue
```

恢复要求相同任务、控制、数据、算法及 batch 配置；`updates` 表示追加次数。
恢复后从新的物理回合开始，不是逐步物理状态恢复。
数值异常环境按失败终止，不进入 AMP 判别器或 replay；异常计数和首个现场写入运行日志。

## 数据与接口约定

保留已训练模型的 COM-v2 任务定义：实时根线速度使用骨盆刚体质心速度，
参考动作根速度来自链接原点位置差分，示范保留原任务数值，重置时按原任务的 COM 解释转换。
这是与已训练模型绑定的语义，不把示范数值称为物理上的 COM 速度。
Gobot 通用 `ArticulationBatch` 始终提供链接原点速度，转换位于任务特征层：
`v_com = v_origin + omega_world × (R * local_com_offset)`。

模型 manifest 检查场景、关节顺序、控制参数和观测协议。不得只改速度定义后直接复用该模型。
代码的来源声明和许可证保留在 `python/gobot/rl/motion_amp/NOTICE.md` 与 `LICENSE-MimicKit`。

## 开发验证

```bash
.venv/bin/python -m pytest tests/python/test_motion_amp.py tests/python/test_articulation_batch.py -q
GOBOT_RUN_MOTION_AMP_GPU_TEST=1 .venv/bin/python -m pytest tests/python/test_motion_amp.py -q
```

保留特征/重置、终态 bootstrap、replay 隔离、推理一致性与数值异常恢复测试。
GPU 故障注入验证单环境失稳后的物理重置和正常训练数据完整性。
