# mjlab 速度训练对照

参考固定为 mjlab `dcdf6ce519abfd2ea45e9503ae6fb650b75bb869`，
RSL-RL 5.5.2、Torch 2.11.0/CUDA 12.8、MuJoCo/MuJoCo Warp 3.17.0、Warp 1.17.0。
对照入口检查源码与依赖，使用原生 mjlab 环境和 PPO runner。
Gobot 的训练入口和任务行为见 [训练指南](gobot_training.md)。

## 共同地形

在 Gobot 仓库根目录导出一次：

```bash
.venv/bin/python tests/support/export_training_terrain.py \
  --robot g1 --output logs/g1_velocity_terrain.json
```

文件保存碰撞几何、高度场、出生点和 SHA256。Gobot 校验自己的编译场景，
mjlab 从同一文件构造地形。仅相同 seed 不能保证不同地形生成器产生相同几何。
两边保留原生 Torch 地形分配和事件顺序，按 `seed + rank` 初始化。
输出目录的 `terrain.json` 记录地形和每卡初始分配哈希，`training_terrain.json` 保存完整输入。
导出器拒绝覆盖已有文件。

## 配对训练

两边都从头训练，依次使用相同八张卡。Gobot：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --config examples/locomotion/configs/g1_velocity_8gpu.json
```

mjlab：

```bash
cd mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_velocity_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --save-interval 500 \
  --output logs/g1_velocity_mjlab_seed42
```

每卡 1024 环境，总计 8192；每次更新采样 24 步，共 3,538,944,000 个控制步。
两边使用相同 PPO 配置、Newton + ImplicitFast、nconmax=70、njmax=1500、EPA horizon=64，
关闭 TF32。这些容量设置是明确的实验参数，不能描述为 mjlab 所有默认设置原样运行。

短测试可将环境数改为每卡 64，更新次数改为 10，使用独立输出目录。
mjlab 入口的 `--check` 只检查配置和依赖。训练目录必须为空，
恢复已有 mjlab 实验时显式传入 `--resume`，其 `--iterations` 表示最终累计更新数。
Gobot 的 `--iterations` 表示本次追加更新数。

## 评测和判读

训练结束后使用最终 checkpoint 和相同的独立评测协议。
固定地形、初始状态、指令、回合长度、噪声和随机化设置；
比较速度/角速度误差、跌倒率、路线完成率、训练时间和显存。
跨后端回放工具为 `tests/support/export_locomotion_comparison.py` 和
`tests/support/evaluate_mjlab_locomotion.py`；导出输入记录观测、关节顺序、初始状态和参考版本。

两个训练器调用相同日志统计代码。奖励与 episode 长度按所有卡本轮完成的回合加权，
奖励项按最长回合秒数归一化。速度误差是回合内时间平均，终止项是结束次数。
不能混用上游原训练入口的不同统计口径，也不能按训练 reward 选择对某一边更有利的模型。

Gobot 场景编译的浮点精度与独立 CUDA 接触轨迹存在差异。
配置和短测试一致不证明最终策略等价；比较稳定性需要同一组多个训练 seed。
先确认基线，再在独立实验中一次只改变一个因素。
