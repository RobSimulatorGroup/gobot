# G1 与 mjlab 的训练采样对齐

2026-11-09。参考仓库内固定版本 `dcdf6ce519abfd2ea45e9503ae6fb650b75bb869`。
这次保留原奖励、PPO 参数和共同地形，修改 Gobot 的 MuJoCo Warp 训练数据采样流程。

## 改动

- 环境与策略使用当前进程对应 GPU 的同一 Torch RNG，与参考流程一致。一个训练进程负责一个环境批次。
- 启动顺序为初始地形抽样、推力计时器、脚摩擦、编码器偏置、身体质心随机化。
- 训练入口先构造环境、应用共同地形分配，再 reset 一次；删除构造后重复 reset(seed) 导致的随机数重启。
- 重置姿态采用一次 `[N,6]` 抽样；零范围速度和关节重置仍按参考调用形状抽样。
- 显式 reset 也走地形课程采样；初始 reset 冻结等级，但仍消费同样的随机数。
- 禁用的跑步命令不额外抽样。命令使用参考的 uniform_，事件和噪声使用 rand 后缩放。
- 推力计时按秒累计，使用参考的触发阈值；先抽下一次触发时间，再一次性抽六轴速度扰动。
- 高度观测噪声在米单位下加入，然后缩放；transition 诊断不消费额外噪声。
- 基线重置时驱动输入清零，与 mjlab 一致；CPU 基线也同步这一项。机器人关节姿态仍恢复默认姿态，
  下一次策略动作照常转换成 PD 目标。AMP 等扩展继续使用自己的重置目标。
- 恢复签名记录采样流程变化，旧训练不能静默 resume 成这轮对照；需从头训练。

Gobot 不导入 mjlab 的环境实现，场景与机器人仍由 Gobot 编译。参考代码只用于测试和独立对照进程。
CPU 环境仍使用 NumPy 的随机数流程，本次与 mjlab GPU 逐样本对齐的是 MuJoCo Warp 路径，不能声称 CPU 的随机样本也逐位一致。

## 实测证据

118 / RTX 4090，seed 42：

1. **每卡 1024 环境的正式初始化流程**：Actor、Critic 初始参数哈希一致；初始随机 episode 长度一致；
   首次策略采样前的 CUDA RNG 状态一致。初始观测仍有浮点差异，因此动作哈希不要求逐位相同。
2. **64 环境的逐步诊断**：启动阶段脚摩擦、编码器偏置、COM、推力计时器和 RNG 完全一致。
   初始 reset 的 qpos 最大误差约 6e-8，Actor 观测约 4e-7；qvel 与清零驱动输入一致。
3. 为覆盖事件，把命令/推力间隔临时设为 0.04–0.08 秒、episode 设为 0.2 秒。
   15 个控制步经过命令重采样、推力和自动重置，每一阶段的 RNG 状态、采样计时器、目标 heading、
   命令模式和结束标志一致。正式任务配置仍是原来的间隔和 episode 长度。
4. **物理不逐位一致**：部分粗糙高度场上的接触力和随后动力学仍不同。重复运行 mjlab 自己时，
   同样的 reset qpos/RNG 下，第一控制步 qvel 最大差也达到约 4.58。
   因此不能将所有接触后的状态差异归因于 Gobot 的采样错误；也不能由此保证两边碰撞实现完全无差异。

这里验证的是采样调用、输入分布与有限步序列。真实训练中，一旦物理或策略造成不同的结束时刻，
重置的环境数量就可能不同，后续采样轨迹也会分开。不能承诺两轮训练学出相同权重，或这项改动已解决偏航。

## 复核

两项捕获分别使用自己的环境依赖。输出 `.pt` 是未训练环境状态，不包含训练策略：

```bash
cd /home/h3c-r5300-g6/wqq/gobot
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/support/capture_mjlab_sampling.py gobot \
  --terrain-reference logs/g1_training_terrain.json --output logs/g1_sampling_check
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 mjlab/.venv/bin/python tests/support/capture_mjlab_sampling.py mjlab \
  --terrain-reference logs/g1_training_terrain.json --output logs/g1_sampling_check
.venv/bin/python tests/support/compare_mjlab_sampling.py logs/g1_sampling_check
```

比较器要求采样字段一致，物理状态的误差单独报告；若因实际结束时刻不同而出现 RNG 分歧，会退出失败，
需要定位第一处分歧，不能只看最终 hash。

## 重训

先做 Gobot 新一轮；与已有 mjlab seed42 比较时仍只有各一个训练 seed，不能据此估计多次训练的稳定性。
共同地形、奖励、命令课程、PPO 预算全部保留。编号 checkpoint 每 500 次更新保存一次。

```bash
cd /home/h3c-r5300-g6/wqq/gobot
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_training_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --no-eval \
  --hours 8 --budget-hours 64 --save-interval 500 \
  --output logs/g1_sampling_seed42 \
  --budget-file logs/g1_sampling_seed42/budget.json
```

不从上轮 checkpoint resume/initialize。跑完用相同固定命令、连续 20 秒、三个评测种子，以及多地形路线评测，
同时报告速度误差、累计偏航、左右转向响应、跌倒和地形通过率。不要仅凭 reward 判断是否修好。
