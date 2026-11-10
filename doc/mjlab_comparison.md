# G1 与 mjlab 对照训练

对照使用固定的 Gobot 场景地形、同一组初始地形分配和同一份统计代码。
两边沿用固定参考版本的奖励权重和 PPO 优化参数。Gobot 的通用训练器与机器人预设分离，
G1 只是 `examples/locomotion/presets.py` 提供的一项任务；机器人资产仍由场景编译。

标准速度基线使用未修改的 `rsl_rl.algorithms.PPO`。固定参考版本为
`dcdf6ce519abfd2ea45e9503ae6fb650b75bb869`，训练任务版本为 `velocity_v3`。

- 每个物理子步更新接触计时；奖励/终止读取最后子步留下的派生状态，射线读取上次 sense 的结果。
- 推力和重置后只 forward 一次，随后更新指令、sense 和观测。
- 接触观测沿用奖励读取的缓存；任意 scene reset 会使整批接触缓存失效，与参考相同。
- 超时回填由 RSL PPO 使用采样动作时保存的价值；失败和 timeout 掩码独立，允许同时为真。
- reset 清零 actuator targets；Warp 环境事件与策略共用 Torch 设备随机数流。
- 启动事件顺序由任务配置表达，框架没有机器人名称判断。G1 与 Go1 的原版事件顺序不同。
- 初次 reset 也执行参考课程调用；恢复 checkpoint 保留已保存的等级，不评判恢复前的临时状态。
- 标准基线关闭 transition diagnostics；AMP 的终态价值扩展只留在显式 AMP 算法中。

Newton + ImplicitFast、奖励非有限值处理和 RSL 输出检查保持参考行为。
CPU 采用相同奖励/传感器时序，但仍使用 NumPy 随机数；固定种子的参考采样对齐对象是 Warp 训练路径。

配置相同、优化器相同和短测通过，均不能证明最终策略质量一致。独立编译的模型存在 float32/XML
精度差异；CUDA 接触求解也并非逐位确定。对照必须同时记录数值误差、原版自身重复运行的误差和
相同预算下的独立评测。先完成本基线的训练和评测，再在新目录中一次只改变一项改进。

## 行为验证

`tests/support/record_velocity_baseline.py` 在两个独立进程中各自初始化，不复制对方的状态或 RNG。
默认 16 环境、64 步，包含部分/整批重置、推力、指令重采样、失败与超时重叠、课程切换。
整批重置发生在第 16/32/48 步，限制探针中的接触轨迹分岔；这不是长 episode 收敛测试。
`--updates 2` 改为真实 PPO 采样/优化，不添加上述探针干预。输出保留模型、全部观测/动作、
奖励、结束掩码、课程和 RNG。`compare_velocity_baseline.py` 逐项报告误差，严格检查随机与离散状态，
不会把连续误差隐藏成一个“全部对齐”的结论。

```bash
.venv/bin/python tests/support/record_velocity_baseline.py --runtime gobot \
  --terrain logs/g1_velocity_v3_terrain.json --output logs/baseline_check/gobot.pt
mjlab/.venv/bin/python tests/support/record_velocity_baseline.py --runtime mjlab \
  --terrain logs/g1_velocity_v3_terrain.json --output logs/baseline_check/mjlab.pt
.venv/bin/python tests/support/compare_velocity_baseline.py \
  logs/baseline_check/gobot.pt logs/baseline_check/mjlab.pt \
  --output logs/baseline_check/comparison.json
```

`--robot go1` 使用相同流程及 Go1 地形导出文件。PPO 回归测试还覆盖真实超时和失败重叠，
在相同 transitions 上逐项比较参数、归一化和 Adam 状态。

## 生成一次训练地形

从 Gobot 仓库根目录执行一次；mjlab 使用仓库下独立的 `mjlab/.venv` 环境：

```bash
.venv/bin/python tests/support/export_training_terrain.py \
  --robot g1 --output logs/g1_velocity_v3_terrain.json
```

这个文件包含全部碰撞几何、高度场、出生点、地形名称/比例及 SHA256。文件来自普通 Gobot
场景的 CPU 编译结果，不需要额外安装 mjlab。导出器不会覆盖已有文件；新实验地形请用新的文件路径。

Gobot 启动时核对自己场景的实际几何是否与文件一致；mjlab 从同一文件构造原 terrain entity，
保留 terrain curriculum 与边界检查。不能只复制 seed，因为不同生成器未必生成相同几何。
两边按 `seed + rank` 使用参考的 Torch 初始地形分配及相同事件顺序，不额外覆盖分配，
每个输出目录的 `terrain.json` 保存共同文件哈希和每卡分配哈希，`training_terrain.json` 保存完整输入。

## 先做八卡短测试

Gobot（在 Gobot 根目录）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_velocity_v3_terrain.json \
  --num-envs 64 --iterations 10 --seed 42 --no-eval \
  --hours .1 --budget-hours 1 --save-interval 10 \
  --output logs/g1_velocity_v3_gobot_smoke \
  --budget-file logs/g1_velocity_v3_gobot_smoke/budget.json
```

完成后测试 mjlab：

```bash
cd mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_velocity_v3_terrain.json \
  --num-envs 64 --iterations 10 --seed 42 --save-interval 10 \
  --output logs/g1_velocity_v3_mjlab_smoke
cd ..
```

两边 `terrain.json` 应完全相同。短测试输出目录要与正式实验分开。

## 正式对照命令

在 tmux 中分别启动，两项使用相同的八张卡，所以依次运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --terrain rough \
  --terrain-reference logs/g1_velocity_v3_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --no-eval \
  --hours 8 --budget-hours 64 --save-interval 500 \
  --output logs/g1_velocity_v3_gobot_seed42 \
  --budget-file logs/g1_velocity_v3_gobot_seed42/budget.json
```

```bash
cd mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_velocity_v3_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --save-interval 500 \
  --output logs/g1_velocity_v3_mjlab_seed42
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
- Gobot 与参考均由 RSL-RL 检查返回的非有限观测/奖励；检查失败时训练退出。
  失败的运行应单列报告，不能把中断前日志当作完成训练的结果。

单卡与多卡日志都是同一口径；无法取得的诊断项不伪填数值，例如 mjlab 的关节力矩饱和度显示 `—`。
