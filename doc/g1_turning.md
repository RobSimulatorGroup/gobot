# G1 转向跟踪对照

原 Gobot / mjlab 基线已经使用相同的三轴角速度奖励，但它们的独立训练结果仍有持续偏航。
本实验只修改角速度奖励项：跟踪身体 z 轴角速度，不在同一项中混合滚转和俯仰速度，并将 std 从 sqrt(0.5) 改为 0.5。
奖励权重仍为 2，姿态、身体摆动、动作平滑度等奖励保留。

- 原基线：`2 * exp(-(wx² + wy² + (wz-command_wz)²) / 0.5)`。
- 转向对照：`2 * exp(-(wz-command_wz)² / 0.25)`。

CPU、Warp 和 mjlab 对照入口调用同一个角速度误差函数。默认 `ang_vel_axes="xyz"` 保留原基线；
`examples/g1/turning.json` 显式选择新奖励。它是一个双方一致的新实验，不是 mjlab 上游原任务已经修好了直行。
本改动不添加航向控制器，也不把评测中的零转向偷偷换成纠偏指令。现有 checkpoint 的动作不会因此改变，需要训练新策略。

## 从头训练

Gobot（118）：

```bash
cd /home/h3c-r5300-g6/wqq/gobot
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --config examples/g1/turning.json \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_training_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 \
  --hours 8 --budget-hours 64 --save-interval 500 \
  --output logs/g1_turning_seed42 \
  --budget-file logs/g1_turning_seed42/budget.json
```

相同奖励设置的 mjlab 对照，完成 Gobot 后再跑：

```bash
cd /home/h3c-r5300-g6/wqq/gobot/mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_training_terrain.json \
  --ang-vel-axes z --ang-vel-std 0.5 \
  --num-envs 1024 --iterations 18000 --seed 42 --save-interval 500 \
  --output logs/g1_turning_seed42
```

两边都从头训练以保证相同预算。新目标不 resume 旧目标；训练配置摘要会拒绝这种混用。
如只想判断 Gobot 是否改善，先完成 Gobot 一轮及下面的评测，不必同时占用两轮完整训练时间。

## 固定命令验收

在空闲卡上检查低速、中速、高速直行，以及行进和原地左右转。所有命令按身体坐标给出，
每组运行 20 秒，不添加航向反馈。旧模型和新模型使用相同的评测入口；这些评测比较行为指标，不比较不同奖励下的 return。

```bash
cd /home/h3c-r5300-g6/wqq/gobot
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m examples.locomotion.assessment \
  --checkpoint logs/g1_turning_seed42/model_final.pt \
  --output logs/g1_turning_review/commands_71.json \
  --backend mujoco-warp --terrain flat --num-envs 32 --steps 1000 --seed 71 \
  --commands 0.8 0 0 --commands 1.5 0 0 --commands 2.5 0 0 \
  --commands 0.8 0 0.5 --commands 0.8 0 -0.5 \
  --commands 0 0 0.5 --commands 0 0 -0.5 \
  --hours .5 --budget-hours 2 --budget-file logs/g1_turning_review/budget.json
```

随后把 seed 改为 72、73，输出文件名也分别更换。粗糙地形仍使用既有的共同地形/重置评测，避免仅优化平地而损失跨地形能力。

- 直行看 `heading_drift_abs_deg`、`cross_track_max_m` 和水平速度误差。
- 转弯看 `yaw_error`、`yaw_rate_bias_rad_s`，实际平均身体 yaw 速度为指令值加该偏差。
- 同时报告跌倒、数值失败和地形通过率。
- 原地转弯不能通过“站着不动”验收；行进中转弯不能只看前进速度。
- 短路线穿越成功不能代替 20 秒连续直行的航向检查。

改变目标的短微调仅用于筛查，不能替代从头、同预算的 Gobot/mjlab 对照，也不能证明多个训练 seed 的稳定性。

## 同 seed 的初始化核对

用 118 上两边的实际训练构造流程、每卡 1024 环境、seed 42 做未更新策略的核对：
Actor 和 Critic 的初始参数哈希分别完全一致；CPU 随机数状态也相同。
但第一批观测、随机 episode 长度和首次动作采样前的 CUDA 随机数状态已经不同。
Gobot 环境使用独立的 Torch generator，mjlab 环境事件与策略采样的随机数消耗顺序不同。

因此当前证据排除了网络初始权重差异，但没有对齐训练轨迹。数据路径不同能够使 PPO 得到不同策略，
不能仅凭这一对结果推断 Gobot 存在系统性训练劣势，也不能断言偏航差距完全由随机性造成。
原始初始化核对保存在 `logs/g1_turning_audit/initial_gobot.json` 和 `initial_mjlab.json`。
