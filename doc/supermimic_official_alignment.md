# 官方 SuperMimic → Gobot：GenCoord 对齐

2026-11-11 起，以本地 `supermimic/` 的 **HybridRobotics/supermimic** 为新对齐目标。
它与现有 **CuboiLeo/SuperMimic-Reproduced** 是不同实现。旧实验的数值对照、checkpoint、
BONES 分数不能作为官方版本的验收证据。旧实现仍支撑当前 `g1-tracking` 训练入口；
`gobot.rl.supermimic.official` 已接通真实场景训练，入口为 `g1-supermimic`。
运行方式见 [BONES-SEED 操作指南](../examples/g1/official/README.md)。现已完成一次单卡 BONES 训练及全量评估；历史成绩与多卡进展见下方，
尚无同预算、多 seed 的算法质量结论。

## 固定来源与验证范围

`examples/g1/official/reference.lock.json` 固定：

- 官方源码：`abc124c1bd5976341034668af3d12c087f2f1f55`。
- 配套 `HybridRobotics/mjlab_research`：`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`。
  这是本次获取的 companion 提交；官方 lock 只有 editable 路径，**未公布训练时的 Git revision**。
- 上游锁定版本：Python 3.13、Torch 2.10.0、rsl-rl 5.4.0、TensorDict 0.11.0、
  MuJoCo 3.10.0、MuJoCo Warp 3.10.0.1、Warp 1.14.0。
- 本轮实际版本写入 fixture metadata。当前为 Torch 2.11.0+cu128、rsl-rl 5.5.2、
  TensorDict 0.13.0、MuJoCo / MuJoCo Warp 3.17.0、Warp 1.17.0。

独立 oracle 首先验证 **同一软件栈、CPU FP32、关闭 compile/TF32 的数学与学习更新**；
后续真实场景和 CUDA 联通检查见下方记录。
源码读取前检查 commit、工作树状态和 Git 对象字节；fixture 记录实际运行模块、导出器、
合成环境和张量文件的 SHA256。参考进程不导入 Gobot，Gobot 测试也不需要参考 checkout。
绕过参考 package 的环境注册 initializer，但直接执行原始数学函数，不替换数值实现。

Gobot 的运行统计模块来自本轮同栈 rsl-rl 5.5.2，保留许可证。它与官方的 5.4.0 原栈结果
是否完全一致还需要独立验证；当前测试不做该声明。Gobot 数值核心只依赖 Torch/TensorDict，
复用已有旋转基元，不把 mjlab manager、MuJoCo 指针、编辑器对象带入算法接口。

## 已查明的语义差异

| 边界 | 现有 Reproduced 配方 | 官方 `abc124c1` GenCoord |
|---|---|---|
| full state | 71D，根角速度为 body frame | 71D，根线/角速度均为 world frame |
| WM representation | 71D heading-invariant，rot6d | 68D：z、gravity/body linear/body angular、29 q、29 dq |
| WM prediction | 70D 速度增量和位置修正 | 35D `[dv_body,dw_body,djoint_vel]` |
| 积分 | 显式积分加位置修正 | 当前根坐标系速度增量；位置系数 `(decimation+1)/(2*decimation)` |
| motion goal | 680D；anchor 版本另扩展 | 10 帧、stride=5；WM 670D，actor 去掉相对位置后 640D |
| actor 默认输入 | 10×93D 本体历史 | 单帧 68D rep_state；proprio/history 为显式配置 |
| 网络 | SiLU、固定缩放 | Linear/LayerNorm/GELU；运行统计归一化 |
| actor | 单 motion encoder | motion/velocity encoders + shared decoder；32D tanh latent |
| context 顺序 | friction、29 encoder bias、3 CoM | friction、3 CoM、29 encoder bias |
| WM 默认窗口 | 8 步预测 | 8 行状态，最后一行动作不评分，实际 7 步预测 |
| WM loss | 多状态通道 + 终止 BCE | root position/rotation matrix、joint position/velocity，按通道加权平均 |
| termination | 学习得到的终止头 | 高度/倾斜阈值的 sigmoid 软终止 |
| actor loss | `-J/H` | `(-J + latent_BC).mean()`，不除 H |
| optimizer | Adam | 隐藏 Linear 权重用 Muon，其余用 fused/capturable Adam |
| reward anchor | torso | pelvis |
| 调度 | 固定 UTD=.4、40 步 warmup | 根据 envs、WM batch/horizon、replay_ratio=16 计算 Ratio |

来源文件：官方 `representations.py`、`models.py`、`learners/{wm_learner,policy_learner,learner_utils}.py`、
`runners/supermimic_runner.py`、`tasks/env_cfgs.py`、`sim_context.py`。以上是代码契约，不是成绩。

## 本轮可执行交付

- `official/state.py`：GenCoord 表示、世界系速度往返、可微积分、损失通道、目标窗口与历史布局。
  FK 通过 callable 注入场景服务，未绑定参考环境。
- `official/models.py` / `normalization.py`：Dynamics、MultiEncoderAE、稳定 buffer 地址的统计更新。
- `official/wm_learner.py` / `policy_learner.py` / `optim.py`：WM、BPTT/actor-critic、GAE、Muon/Adam。
  world-model 环境以 tensor protocol 注入；CPU 默认 eager，CUDA capture 保留参考执行顺序。
- `official/replay.py`：参考的环形窗口采样、终止边界、因果历史 padding。
- `official/rewards.py`：参考的 pelvis re-anchor 跟踪奖励。
- 独立 fixture 与测试：1/8/32 步预测值、loss、输入/参数梯度、一次 WM 更新、
  actor + 20 次 critic 更新及 optimizer state；actor 更新测试使用明确标注的合成可微环境。
  它能验证学习器顺序/梯度，不能验证真实运动目标、接触、终止或场景 physics。
- 重放窗口测试涵盖环绕覆盖、terminated/truncated、相邻 clips 的目标窗口。

普通数学比较容差 `atol=1e-5, rtol=1e-4`；更新参数容差 `atol=1e-6, rtol=1e-4`。
布尔、索引和 replay row 精确比较。GPU 峰值、完整训练和全量评估记录见下方；
长期动力学对拍及同预算多 seed 质量比较仍待完成。

2026-11-12 新增工程验证：

- 官方模型导入 `.jscn` 后 1161 项静态物理检查通过。
- 从 118 的完整 BONES 库取 4 个片段 / 5382 帧；官方 MotionLoader 独立读取六组字段逐值一致。
- 真实 CPU、CUDA scene 各完成 16 控制步 / 9 learner updates / 180 critic updates 并保存。
- CUDA `torch.compile + graph capture` 完成 12 控制步 / 5 learner updates；冷编译包含在计时中。
- 完整官方网络宽度、H=32，以 64 环境 / 64 学习 batch 完成一次真实 BONES GPU 更新。
- 配置入口、数据完整性、clip 边界、启动 DR、checkpoint 恢复与计时停止有回归测试。
- 118 完整转换：129943 clips / 47475680 frames，六组状态保留源 fp16 值，全部 SHA256 校验通过；全库加载与 CPU 单控制步通过。
- 本机合并当前 master 后：111 tests passed，1 个可选 GPU 测试跳过；实际 GPU 联通检查另行通过。
- 已完成单卡 1024 env / WM512 / policy1024 的完整训练与全量评估（见下方）；未修改 C++ 引擎代码。

完整数据 manifest SHA256：`f3ec3ea8025f39e827c3c86ea9a5736c9764cb48a492dc9aa7c2ac384980b422`。
本次代码在本机 `codex/supermimic-bones-seed` 分支提交，jump-wqq 拉取后同步到 118。
正式长训练由用户执行；后续只自动运行短验证与评估。

生成独立参考数据（不安装或改写参考源码；需要已安装 Torch/RSL/TensorDict 的 Python）：

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m examples.g1.official.math_fixture \
  --reference supermimic --mjlab mjlab_research \
  --output tests/fixtures/supermimic_official_gencoord_v1.pt

.venv/bin/python -m pytest -q tests/python/test_supermimic_official.py
```

检查固定来源但不生成张量：

```bash
.venv/bin/python -m examples.g1.official.reference \
  --reference supermimic --mjlab mjlab_research \
  --output logs/supermimic_official/source_audit.json
```

## 后续质量验收（接通不等于复现成绩）

1. 固定两边实际训练配置、同一数据 manifest、相同硬件/时间预算和最终评估协议。
2. 做真实 CPU/Warp 固定动作和冻结策略轨迹对照，定位第一处分歧；静态审计不能替代动力学对拍。
3. 正式 batch 的显存/吞吐验收，随后用户从头训练和多 seed 评估。
4. 原栈 Torch 2.10/MuJoCo 3.10 的独立验证、MaxCoord 仍为后续工作。同步多卡已接通，验收记录见下方。

已有实装：官方 G1 导出/审计、packed 数据转换、场景 FK 和物理接口、startup DR、
return sampler、WM imagination、回放、训练、按时停止、保存恢复与有限 lane 评估。
官方 motion-only 配方保留 velocity encoder 参数，但其 slot 为 0；没有实现 velocity 任务训练。


## 已完成的 BONES 训练与旧 Reproduced 对比

两边均由 Gobot 场景执行，使用同名的 129943 clips / 47475680 source frames，均为 seed0。
官方算法运行 `bones_s0` 已完成，checkpoint SHA256
`4114c37de7c1a09b7dd99fba4e2847439296f5c99abb8e5f60a1a95b55c4df57`。
全量评估用 8 卡分片、每片 512 lanes，合并检查 clip 无缺失/重复/截断。

| 项目 | 旧 Gobot Reproduced | Gobot 官方 GenCoord |
|---|---:|---:|
| 全量平均归一化回报 | 0.935562 | 0.917437 |
| 全量失败 | 429 / 129943 | 652 / 129943 |
| 固定过滤集平均回报 | 0.936911 | 0.918559 |
| 固定过滤集失败 | 90 / 129491 | 551 / 129491 |
| 训练硬件 | 8 × RTX 4090 | 1 × RTX 4090 |
| 全局 transitions | 3276800000 | 102400000 |
| learner updates | 39984 | 39997 |
| 循环墙钟 | 38981.79 s | 12359.86 s |
| GPU·小时 | 86.626 | 3.433 |

旧版样本量是 32 倍、GPU 时间约 25.23 倍；不能据此认定算法或样本效率优劣。
除上表算法差异，旧评估还使用 dense FP32、torso anchor、N 帧执行 N 步；官方接入使用
保留源 fp16 的参考、pelvis anchor、N 帧执行 N−1 步。两份评估各自检验对应任务。
逐 clip：共同失败 122；旧失败/新成功 307；旧成功/新失败 530。
新版本在名字包含 flip/handstand/cartwheel 的片段上失败更少，在 jump/stand_up/dance 上更多；
关键词有重叠，不是人工标注类别。原始结果存于
`logs/supermimic_official/bones_s0_evaluation_20261112/`。

## 同步多卡实现与比较口径

`official/distributed.py` 使用 torchrun，一张 GPU 一个进程；不把 RSL-RL PPO runner 接入
官方学习器。RSL-RL 的 Logger/writer 仅负责 TensorBoard，JSONL 和实时状态使用官方控制步时钟。

- 默认全局 batch 均分：固定 1024 env / WM512 / policy1024，1/2/4/8 卡的 Ratio 均为 .4；
  WM、actor、critic 都在梯度全局平均后裁剪，再执行原 Muon/Adam。没有 sqrt(world) 学习率缩放。
- dynamics/actor/critic 初始参数广播；归一化合并全局 moments；ReturnSampler 先合并新样本计数和
  回报总和，再应用 memory cap，避免各卡自行改变课程。每卡环境、回放和 RNG 独立。
- 所有 rank 同步判断回放是否有有效窗口、非有限学习指标、停止信号与时间预算；rank0 独占公共
  日志/原子 checkpoint，保存所有 rank 的 RNG 与 startup context。
- CUDA graph 中包含 NCCL 梯度归约；首次 eager warmup 校验各卡 active-gradient 布局，保留
  inactive velocity encoder 的 grad=None。退出时先释放通信 graph，再销毁 process group。
- 同卡数/同来源的 resume 恢复优化器和全局 sampler，物理回合重置、回放重填；改变卡数或代码
  使用显式 `--initialize-from` 权重迁移，记录来源 SHA256 并清零本阶段训练计数。

CPU/Gloo 双进程验证涵盖真实 WM/actor/20-step critic 更新、Muon/Adam 状态、全局归一化、
ReturnSampler memory cap、单 rank 回放不足、停止信号、时间预算和保存恢复。
各 rank 参数/优化器逐位相同；固定样本与单进程全局 batch 比较容差 atol=1e-5、rtol=1e-4
（Muon 内部 BF16 正交化会放大归约顺序的舍入差别，测试观测最大差小于 8e-6）。

稳定吞吐、编译成本和最终质量分别报告。缩短一次同等计算量训练，使用固定全局 batch 的
`official_bones_1024.json`；扩大到 `official_bones.json` 属于增加四倍样本量的实验。
对比旧 Reproduced 时至少同时记录：卡数、墙钟、GPU·小时、全局 transitions、更新次数、
输入/奖励/评估协议。新多卡正式训练仍由用户启动，短跑不代表最终质量已经提升。


### 118 实测吞吐与验收

2026-11-12，完整 BONES、正式网络、H32、WM horizon8、compile+CUDA graph；每组200控制步，
77次learner update。下表稳定速度取第51–200步。相同全局配方内，batch、学习率、Ratio和总样本量固定。

| 全局 env / WM / policy | 1卡 transitions/s | 2卡 transitions/s | 4卡 transitions/s | 4卡相对1卡 |
|---|---:|---:|---:|---:|
| 1024 / 512 / 1024 | 8484 | 9052 | 9034 | 1.065× |
| 2048 / 1024 / 2048 | 12445 | 15257 | 16479 | 1.324× |

2048配方四卡稳定训练区间耗时约减少24%；1024配方只减少约6%，不应期待8卡线性加速。
2048配方的1/2/4卡每卡Torch峰值分别为20.57/16.73/14.81 GiB（不含CUDA/NCCL外部分配）。
初次编译约数分钟，且缓存命中不一致；200步总墙钟不能直接作为加速比。上述为单次短跑，
不代表100k长训总耗时或最终质量。所有测试的每卡模型和优化器都经过逐字节SHA256一致性检查。

额外验收：95项本机回归；双卡CUDA训练24步并恢复至48步；已完成的bones_s0可显式初始化四卡。
按用户要求，八卡以完整网络/H32、小全局batch128进行32步/10更新验证，完成100次WM、10次actor、
200次critic更新，启用全库、DR、ReturnSampler和CUDA graph，关闭torch.compile以减少启动时间。
八卡模型/优化器一致，正常保存退出；八卡正式batch的稳定加速比尚未测量。
详细报告和原始日志位于 `logs/supermimic_official/distributed_validation_20261112/`。


## 后续统一评估

已在原始FP32参考、同一Gobot场景、128/500容量、N−1步时序、相同失败阈值下重新评估
Reproduced与官方八卡模型，并在同条轨迹上同时计算骨盆/躯干奖励。
全量失败为429→19；共用骨盆回报0.931829→0.950115，共用躯干回报0.935604→0.945823。
这组结果可用于同评估协议比较，替代上方历史各自协议的跨算法分数对照。
训练预算和策略输入信息仍不同；完整配置、校验与结果见
[统一评估基准](supermimic_common_evaluation.md)。
