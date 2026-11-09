# G1 与 mjlab 对照训练

对照使用固定的 Gobot 场景地形、同一组初始地形分配和同一份统计代码。
两边沿用固定参考版本的奖励权重和 PPO 优化参数。Gobot 的通用训练器与机器人预设分离，
G1 只是 `examples/locomotion/presets.py` 提供的一项任务；机器人资产仍由场景编译。

当前 Gobot 明确采用以下训练约定，与固定 mjlab 参考的部分行为不同：

- 最后物理子步完成后先刷新状态，再计算奖励、终止和最终观测；没有旧时序开关。
- `TransitionPPO` 用重置前的最终状态进行 timeout bootstrap，PPO 优化仍由 RSL-RL 执行。
- reset 恢复配置的默认关节目标，环境随机数与策略采样分离。
- 恢复训练保留地形等级并开启新 episode；不会在加载时再次升降地形等级。
- 数值失败中止训练并保存诊断，不生成零奖励重置样本。

因此这是相同地形和预算下的训练结果对照，不是逐步或逐位复现。奖励公式与 PPO 更新检查
仍有意义，但不能据此宣称完整环境语义相同。报告必须记录这些差异。

接触计时每个物理子步累计 `physics_dt`（当前 0.005 秒），reset 清空相应世界的计时。
不再用 float32 仿真时钟相减，避免长时间回放误判首次触地。旧计时状态已删除。

## 生成一次训练地形

从 Gobot 仓库根目录执行一次；mjlab 使用仓库下独立的 `mjlab/.venv` 环境：

```bash
.venv/bin/python tests/support/export_training_terrain.py \
  --robot g1 --output logs/g1_velocity_v2_terrain.json
```

这个文件包含全部碰撞几何、高度场、出生点、地形名称/比例及 SHA256。文件来自普通 Gobot
场景的 CPU 编译结果，不需要额外安装 mjlab。导出器不会覆盖已有文件；新实验地形请用新的文件路径。

Gobot 启动时核对自己场景的实际几何是否与文件一致；mjlab 从同一文件构造原 terrain entity，
保留 terrain curriculum 与边界检查。不能只复制 seed，因为不同生成器未必生成相同几何。
两边按 `seed + rank` 调用同一个初始分配函数，地形类型数量和初始难度逐卡相同，
每个输出目录的 `terrain.json` 保存共同文件哈希和每卡分配哈希，`training_terrain.json` 保存完整输入。

## 先做八卡短测试

Gobot（在 Gobot 根目录）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_velocity_v2_terrain.json \
  --num-envs 64 --iterations 10 --seed 42 --no-eval \
  --hours .1 --budget-hours 1 --save-interval 10 \
  --output logs/g1_velocity_v2_gobot_smoke \
  --budget-file logs/g1_velocity_v2_gobot_smoke/budget.json
```

完成后测试 mjlab：

```bash
cd mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_velocity_v2_terrain.json \
  --num-envs 64 --iterations 10 --seed 42 --save-interval 10 \
  --output logs/g1_velocity_v2_mjlab_smoke
cd ..
```

两边 `terrain.json` 应完全相同。短测试输出目录要与正式实验分开。

## 正式对照命令

在 tmux 中分别启动，两项使用相同的八张卡，所以依次运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_velocity_v2_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --no-eval \
  --hours 8 --budget-hours 64 --save-interval 500 \
  --output logs/g1_velocity_v2_gobot_seed42 \
  --budget-file logs/g1_velocity_v2_gobot_seed42/budget.json
```

```bash
cd mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_velocity_v2_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --save-interval 500 \
  --output logs/g1_velocity_v2_mjlab_seed42
cd ..
```

每卡 1024 环境，共 8192；每次更新采样 24 控制步，预算相同。
建立基线时从头训练，两边使用相同预算。
Gobot 的恢复检查包含任务语义、模型、接触计时实现和地形哈希；旧 checkpoint 不兼容当前任务，
不应使用 `--resume` 或 `--initialize` 带入本次基线。旧结果仅保留为历史记录。mjlab 对照入口也拒绝
缺少共同地形信息的旧实验恢复。新实验使用上面的新输出目录。

上述命令关闭 Gobot 的训练内评测，结束后对两边的最终策略执行相同独立评测。训练时不能根据日志 reward
选出对某一边更有利的模型。单个训练种子不足以比较稳定性；应对两边使用同一组多个种子。

## 日志含义

两边调用 `python/gobot/rl/locomotion/training_log.py` 中相同的统计和格式化代码；mjlab 对照脚本
只加载这些独立 Python 帮助文件，不导入 Gobot 引擎，不修改 mjlab 的奖励或优化器。

- `Mean reward` / `Mean episode length`：所有卡在本轮完成的 episode 均值，未完成的累计值保留到后续更新。
- `Episode_Reward/*`：加权奖励率积分除以配置的最长 episode 秒数，然后按实际完成数量平均。
- `Metrics/twist/error_vel_xy` / `error_vel_yaw`：完成 episode 内的时间平均误差，再跨 episode 平均，单位 m/s、rad/s。
- `Episode_Termination/*`：全卡本轮实际结束次数，失败优先于 timeout，原因互斥；不是百分比。
- 足滑按接触脚样本加权，落地高度/力按落地次数加权。没有样本显示 `—`，JSON 为 `null`。
- 地形等级均值按真实环境数量加权；空类型显示 `—`。
- 两边都写 `metrics.jsonl`；mjlab 同时写 TensorBoard。上游原始 `mjlab.scripts.train` 的日志统计未改。
- mjlab 的 solver 超限仍为所有卡求解调用的比例，参数保持 10/20，容量/数值错误没有被隐藏。
- Gobot 数值失败会中止采样并写 `numerical_failure_rankN.pt`，保留上次成功 checkpoint。
  失败的运行应单列报告，不能把中断前日志当作完成训练的结果。

单卡与多卡日志都是同一口径；无法取得的诊断项不伪填数值，例如 mjlab 的关节力矩饱和度显示 `—`。
