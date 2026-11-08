# G1 直行退步排查与重训

2026-11-08。检查旧的 `g1_29dof_8gpu` 与新的 `g1_shared_gobot_seed42` 保存的源码，
并在 118 的 RTX 4090 上运行短程诊断。未启动新的完整训练。

确认了一个 Gobot 实现问题并修复：**Warp 环境在保存 transition 诊断观测时再次抽取观测噪声，
推进了命令、重置和推力共用的随机数生成器。** 诊断开关因此会改变之后的训练样本。
这不是给同一个策略观测叠加两倍噪声，而是多消耗了一组随机数。

这个问题在新旧两个训练版本中都存在，不能据此宣称找到了新版本左转的唯一原因。
本次修复让 transition 记录与 CPU 路径一样使用不附加随机噪声的观测；正常 actor 观测仍保留原噪声配置。
奖励、PPO 参数、地形和接触计时不变。恢复训练的配置摘要包含这个变化，避免静默续接旧训练。

## 已验证的证据

1. **诊断开关确实改变数据。** 修复前，16 个环境使用相同 reset seed，初始观测完全一致；
   开启/关闭 transition 记录后第一步 actor 观测最大差 2.967，环境 RNG 状态不再相同。
   将命令周期临时缩短为 0.04–0.08 秒以复现后，第三步命令最大差达到 1.831。
   这些短周期仅用于测试，不修改正式训练配置。
2. **修复后 CPU/GPU 回归检查通过。** 测试包含观测噪声、域随机化、推力、命令重采样、自动重置，
   并比较开启/关闭诊断和调用日志统计的三个路径。RNG、结束标志完全一致，噪声差小于 2e-6；
   GPU 并行接触求解的连续物理量用绝对 1e-3、相对 1e-4 容差比较。
3. **未发现 PPO 缓存覆盖或命令错位。** 原代码 64 个环境、10 次 PPO 更新、15,360 个样本、231 次重置：
   旧观测在 step 后的最大变化为 0，PPO 存储观测误差为 0，奖励读到的命令与采样动作所依据的命令差为 0；
   日志统计没有修改动作/奖励/观测，也没有推进环境或 CUDA 全局 RNG。损失有限，normalizer count 正确。
4. **不能将长时钟问题直接套到这次 20 秒训练。** 在同一段旧策略轨迹上同时计算新旧接触计时，
   初始时钟为 0、连续运行 20 秒的 8 个环境，两种计时均识别 1005 次落地，摆脚高度与落地力奖励完全一致。
   人为将时钟设为 32 秒时，旧计时确实漏掉 273/1006 次落地；但实测 Gobot 的 provider reset 会清零时钟，
   正式训练 episode 最多 20 秒，不能将人工大时钟结果视为实际训练中的发生频率。
   固定 dt 的实现仍应保留。计时浮点值和 critic 输入仍可能有微小差异，本测试没有证明训练逐位相同。

本地相关回归 19 项通过、2 项 GPU 检查按配置跳过；双进程 PPO 回归 7 项通过。
118 候选修复的 CPU 和 GPU 两项检查均通过。

## 版本差异与尚未证明的部分

实际保存的源码表明，旧版到共同地形版本的主要变化是接触计时、初始地形分配和训练日志。
PPO 算法本身沿用相同的 RSL-RL 实现。共同地形入口使用新的初始难度分配，并额外执行一次带 seed 的 reset；
这会改变出生位置、朝向、命令和后续随机数序列。两轮都填 seed 42 不等于两轮取得相同训练样本。

目前没有发现命令符号接反、奖励读错命令、PPO 旧观测被新状态覆盖或统计代码污染训练数据。
这不是对所有路径的无 bug 保证。累计航向没有直接进入奖励、转向跟踪较弱的问题仍然存在；
这些因素能允许偏置保留下来，但尚不能证明是哪一次更新产生了左偏。

优先做一轮只包含观测采样修复的重训。先不要同时调整 yaw 权重、加航向控制器或更换地形，
否则无法判断改进来自哪项变化。该轮仍使用原共同地形，和失败的 `g1_shared_gobot_seed42` 比较。
如果结果恢复，再补训练 seed 检查稳定性；单次恢复不等于已经排除训练随机性。

## 在 118 重训

从头训练，不 resume / initialize 旧 checkpoint；输出目录使用新名字。
这次保留每 500 更新的编号 checkpoint，便于回查偏航何时出现。

```bash
cd /home/h3c-r5300-g6/wqq/gobot
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_training_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --no-eval \
  --hours 8 --budget-hours 64 --save-interval 500 \
  --output logs/g1_observation_fix_seed42 \
  --budget-file logs/g1_observation_fix_seed42/budget.json
```

完成后对最终模型做平地检查：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m examples.locomotion.assessment \
  --checkpoint logs/g1_observation_fix_seed42/model_final.pt \
  --output logs/g1_observation_fix_review/flat_71.json \
  --backend mujoco-warp --terrain flat \
  --speeds -0.3 0.8 1.5 2.5 --steps 1000 --num-envs 32 --seed 71 \
  --hours .5 --budget-hours 2 \
  --budget-file logs/g1_observation_fix_review/budget.json
```

对中间模型检查时，将 checkpoint 换成 `model_004999.pt` 等已有文件，输出文件也用不同名字。
评测 GPU 应空闲，避免与八卡训练争用。重点看 `heading_drift_abs_deg`、`yaw_rate_bias_rad_s`、
速度误差和路线通过率，不用单一 reward 判定恢复。

原始短诊断脚本和 JSON 保存在 `logs/g1_regression_audit/`；正式回归测试为
`tests/python/test_transition_observations.py`。此前两边最终模型的交叉评测报告为
`logs/g1_shared_comparison/review.md`。
