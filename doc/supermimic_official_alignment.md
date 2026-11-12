# 官方 SuperMimic → Gobot：GenCoord 对齐

2026-11-11 起，以本地 `supermimic/` 的 **HybridRobotics/supermimic** 为新对齐目标。
它与现有 **CuboiLeo/SuperMimic-Reproduced** 是不同实现。旧实验的数值对照、checkpoint、
BONES 分数不能作为官方版本的验收证据。旧实现仍支撑当前 `g1-tracking` 训练入口；
`gobot.rl.supermimic.official` 已接通真实场景训练，入口为 `g1-supermimic`。
运行方式见 [BONES-SEED 操作指南](../examples/g1/official/README.md)。此次验证是工程联通，
不等于完整训练质量验收。

## 固定来源与验证范围

`examples/g1/official/reference.lock.json` 固定：

- 官方源码：`abc124c1bd5976341034668af3d12c087f2f1f55`。
- 配套 `HybridRobotics/mjlab_research`：`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`。
  这是本次获取的 companion 提交；官方 lock 只有 editable 路径，**未公布训练时的 Git revision**。
- 上游锁定版本：Python 3.13、Torch 2.10.0、rsl-rl 5.4.0、TensorDict 0.11.0、
  MuJoCo 3.10.0、MuJoCo Warp 3.10.0.1、Warp 1.14.0。
- 本轮实际版本写入 fixture metadata。当前为 Torch 2.11.0+cu128、rsl-rl 5.5.2、
  TensorDict 0.13.0、MuJoCo / MuJoCo Warp 3.17.0、Warp 1.17.0。

目前只验证 **同一软件栈、CPU FP32、关闭 compile/TF32 的数学与学习更新**。
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
布尔、索引和 replay row 精确比较。正式宽度模型、GPU compile/capture、物理对照尚未验收。

2026-11-12 新增工程验证：

- 官方模型导入 `.jscn` 后 1161 项静态物理检查通过。
- 从 118 的完整 BONES 库取 4 个片段 / 5382 帧；官方 MotionLoader 独立读取六组字段逐值一致。
- 真实 CPU、CUDA scene 各完成 16 控制步 / 9 learner updates / 180 critic updates 并保存。
- CUDA `torch.compile + graph capture` 完成 12 控制步 / 5 learner updates；冷编译包含在计时中。
- 完整官方网络宽度、H=32，以 64 环境 / 64 学习 batch 完成一次真实 BONES GPU 更新。
- 配置入口、数据完整性、clip 边界、启动 DR、checkpoint 恢复与计时停止有回归测试。
- 仍需在目标硬件验证全库正式 batch 峰值，以及从头训练的最终质量。未修改 C++ 引擎代码。

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
4. 原栈 Torch 2.10/MuJoCo 3.10 的独立验证、MaxCoord、同步多卡仍为后续工作。

已有实装：官方 G1 导出/审计、packed 数据转换、场景 FK 和物理接口、startup DR、
return sampler、WM imagination、回放、训练、按时停止、保存恢复与有限 lane 评估。
官方 motion-only 配方保留 velocity encoder 参数，但其 slot 为 0；没有实现 velocity 任务训练。
