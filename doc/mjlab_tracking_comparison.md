# dance2_subject1 原生 mjlab / Gobot 对照

基线版本固定为 mjlab `dcdf6ce519abfd2ea45e9503ae6fb650b75bb869`，
RSL-RL 5.5.2、Torch 2.11.0/CUDA 12.8、MuJoCo/MuJoCo Warp 3.17.0、Warp 1.17.0。
只训练 LAFAN `dance2_subject1`，原始 6771 帧、30Hz，约 225.67 秒。

先运行原生 mjlab。基线入口不导入 Gobot；使用未修改的
`unitree_g1_flat_tracking_env_cfg`、`MotionTrackingOnPolicyRunner` 和 RSL-RL PPO。
仅指定批量数、随机种子、迭代数、128/500 接触约束容量、关闭 TF32，以及本地日志路径。
日志包装器记录全卡已结束回合的加权均值、同步计时及容量错误，不修改奖励、终止或动作。

## 唯一共同输入

使用原生 mjlab `MotionLoader` 的 FP32 插值及原生 Scene/Simulation FK 转换一次，
输出完整刚体顺序及名称。两边必须使用这个 NPZ，而非各自独立转换 CSV。
`reference.json` 保存 CSV/NPZ SHA256、依赖版本与上游源码文件校验值。
`scene/` 保存原生名义模型，供后续 Gobot 场景导入及参数审计。
准备器不上传 W&B、不训练策略。

在准备好的 mjlab 环境中，从 Gobot 仓库根目录执行：

```bash
mjlab/.venv/bin/python tests/support/train_mjlab_tracking.py prepare \
  --csv /path/to/lafan1/g1/dance2_subject1.csv \
  --output logs/tracking-comparison/dance2-reference
```

## 先训练 mjlab

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
mjlab/.venv/bin/torchrun --standalone --nproc-per-node=8 \
  tests/support/train_mjlab_tracking.py train \
  --reference logs/tracking-comparison/dance2-reference \
  --output logs/tracking-comparison/mjlab-dance2-s42 \
  --num-envs 1024 --iterations 30000 --seed 42
```

每卡 1024 环境，8 卡共 8192；rollout 24，30000 更新共 5,898,240,000 个控制步。
`--iterations` 为目标累计更新数；恢复到新目录使用 `--resume .../model_*.pt`。
未训练完成前不能把中间 checkpoint 当作最终比较结果。

## Gobot 对齐验收

旧的 `logs/dance2_subject1-8gpu` 运行保留作参考，不纳入严格对照。
严格对照必须先检查以下项目：

- 相同动作 NPZ、名称顺序、帧率、root/anchor/IMU 定义。
- 实际编译出的质量、惯量、几何、碰撞排除、接触参数、关节限位、执行器、初始状态与地面。
- 相同积分器、求解器参数、子步、物理状态读取时点、重置和参考帧推进次序。
- 原生按 rank 独立维护的自适应采样，控制步更新；不使用 Gobot 的 rollout 后全局汇总变体。
- 相同观测布局、噪声、启动随机化、随机推力及 Torch 随机数调用顺序。
- 相同 PPO 超参数、初始参数、观测归一化、随机回合长度初始化、每卡批量与 minibatch。
- 物理状态批量读取复用后端计算，消除重复 Python FK；性能测试保持采样语义。
- 独立数值探针、相同动作序列短回放、完整片段评估及吞吐/显存记录。

物理编译精度和 GPU 接触排序仍可能导致长期轨迹分歧；必须报告参数差异及容差。
通过配置检查或短训练不代表训练收敛结果严格相同。训练结果使用相同最终更新数和固定评估协议。

## 已落地的对齐和可复现探针

`tests/support/prepare_gobot_tracking_reference.py` 将原生名义模型导入 Gobot，
保留 `init_state` 默认关节角，并比较 1092 项物理属性。`ctrlrange` 在两边
`ctrllimited=false` 时不参与有效物理参数判定；其余差异不被静默忽略。
默认随机化使用 Torch 全局 RNG、各卡独立 sampler、原生初始随机回合时钟，
二维网格环境原点和 solver 参数。连杆位置/朝向/速度读取后端已有结果，
保持 pre-integration 奖励和 post-forward 观测时序。

原生探针与 Gobot 对照：

```bash
mjlab/.venv/bin/python tests/support/probe_mjlab_tracking.py \
  --motion logs/tracking-comparison/dance2-reference/motion.npz \
  --output logs/tracking-comparison/native-probe

.venv/bin/python tests/support/prepare_gobot_tracking_reference.py \
  --reference logs/tracking-comparison/dance2-reference \
  --output logs/tracking-comparison/gobot-aligned-scene

.venv/bin/python tests/support/compare_tracking_probe.py \
  --project logs/tracking-comparison/gobot-aligned-scene \
  --probe logs/tracking-comparison/native-probe \
  --output logs/tracking-comparison/probe-comparison.json
```

原生探针加 `--randomized` 可比较启动随机化、观测噪声、采样帧和 CUDA RNG。
固定帧探针使用 0、1200、6000、10000，针对这条长舞蹈，不适用于任意短动作。
连续奖励、观测、终止和帧/RNG 都有明确阈值检查；自碰撞计数另行列出。

必须保留的一项限制：上游 `reduce=none, num_slots=1` 的自碰撞传感器依赖
GPU 原子并行匹配到的第一个接触。原生 mjlab 自身重复同 seed 的短回放，
也观察到单步 0.2 的自碰撞奖励差异。不能将这项离散误差描述为零，或将
配方对齐描述为长期逐位确定性。对照不修改上游接触槽语义。

最终原生整段评估（关闭随机化、噪声和推力）：

```bash
mjlab/.venv/bin/python tests/support/evaluate_mjlab_tracking.py \
  --checkpoint logs/tracking-comparison/mjlab-dance2-s42/model.pt \
  --motion logs/tracking-comparison/dance2-reference/motion.npz \
  --output logs/tracking-comparison/mjlab-full-evaluation.json
```

正式 Gobot 对照入口会拒绝未完成的原生基线、不同进程数、动作哈希或已改动的编译模型：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  tests/support/train_gobot_tracking_comparison.py \
  --baseline logs/tracking-comparison/mjlab-dance2-s42 \
  --project logs/tracking-comparison/gobot-aligned-scene \
  --output logs/tracking-comparison/gobot-dance2-s42
```

它从 baseline 的 comparison.json 取卡数、每卡环境数、seed、总更新数和容量，
从通过审计的 task.json 取场景配置。`--preflight-updates N` 仅用于显式短验证，
会标注 preflight，不能代替正式训练。正式运行必须等 mjlab 的 status.json 为 trained。

可在单独 tmux 中运行完整顺序流水线（前提是已导入并通过物理审计的 Gobot 项目）：

```bash
.venv/bin/python tests/support/finish_tracking_comparison.py \
  --baseline logs/tracking-comparison/mjlab-dance2-s42 \
  --project logs/tracking-comparison/gobot-aligned-scene \
  --output logs/tracking-comparison/gobot-dance2-s42 \
  --report logs/tracking-comparison/final-comparison.json
```

流水线等原生基线成功结束后，才占用相同八卡训练 Gobot；然后依次执行两个模型的
完整片段评估并输出对照报告。任一步失败都停止后续步骤。运行期间报告是阶段状态，
只有 `state=completed` 才包含最终结果。使用全新输出目录，不能与已有训练并发竞争 GPU。

## 在 Gobot Editor 中实时运行原生 mjlab 模型

`examples/g1/tracking/mjlab_live_runtime.py` 在 Gobot 的异步运行线程中加载
原生 mjlab 环境和 checkpoint，每 20ms 重新推理动作，推进四个 5ms 物理步。
`mjlab_live.py` 只在场景线程消费已完成的位姿快照，更新 Play 的 runtime clone。
此模式的物理后端是原生 mjlab/MuJoCo Warp；它可视化原生模型效果，不代表
该模型已经通过 Gobot 物理后端的迁移验证。

```bash
.venv/bin/python tests/support/prepare_mjlab_live.py \
  --scene logs/tracking-comparison/gobot-aligned-scene/world.jscn \
  --checkpoint logs/tracking-comparison/mjlab-dance2-s42/model.pt \
  --motion logs/tracking-comparison/dance2-reference/motion.npz \
  --output logs/tracking-comparison/mjlab-dance2-live-editor

.venv/bin/gobot_editor --path logs/tracking-comparison/mjlab-dance2-live-editor
```

生成器要求新的输出目录。Python 环境须满足已固定的数值版本及 mjlab 依赖；
`--extra-python-paths` 可显式追加已有可选依赖目录，运行器先加载并验证数值栈。
本机复用了 `mjlab/.venv/lib/python3.13/site-packages` 的非数值依赖。

点击顶部 Play：蓝色为策略控制的物理机器人，绿色为运动参考。
模型分别使用独立 PBR 材质；可见地面的上表面和物理地面均为 z=0。
两台机器人仅在显示时平移分开，并用相同的参考 pelvis XY 平移保持取景，
该显示变换不改动物理状态或跟踪误差。

- 顶部工具栏支持播放、暂停、停止和单步。
- 视口控制焦点内 `R` 重置，`F` 施加 x 方向 0.8m/s 的根部速度扰动。
- Physics 面板显示 LIVE、推理步数、误差及动作幅度；`live-status.json` 每秒更新。
- 接触调试由引擎内置：后端适配器在最后一个实际执行的 5ms 子步结束后、
  forward/reset 前采集 SDK 求解结果；C++ 将接触对转换为逐连杆受力，随不可变快照
  交给 SimulationServer、Physics 面板和原生调试渲染。示例不再生成 Python 箭头。
  `Contact Forces (N)` 面板按机器人/连杆汇总 Fx/Fy/Fz 和合力，包含自碰撞的两侧反力；
  这表示连杆的净接触力，不能一概当作地面支撑力。绿色运动参考不产生物理接触力。
  `Contact points / normals` 和 `Contact force arrows` 可独立开关，暂停时立即生效；
  原生力箭头长度使用 `min(max_length, force_scale * log1p(force_norm))`，真实数值仍以 N 显示。
  `contact_roots` 只声明各环境的显示根节点，坐标变换由 C++ 完成；重置、失败、停止后
  清除旧接触快照，已被读者持有的旧快照保持只读。
- 关闭随机化、观测噪声和自动推力；首次失败或动作结束后保持最终姿态，按 R 重来。

如需保存一次评估的实际运动，可给 `evaluate_mjlab_tracking.py` 增加
`--trajectory trajectory.npz`；配套 `.mjb` 只用于离线可视化。
`render_tracking_trajectory.py` 可生成并排 MP4；`prepare_tracking_replay.py`
可生成纯记录回放项目。记录回放与上述实时推理项目是不同的入口。

## 训练流程定位与稳定对照配方（2026-11-15）

旧结果是原生 `reduce=none, num_slots=1` 配方，必须保留这一实验标记。
首次更新探针确认：初始网络权重及CPU/CUDA RNG相同，24步内采样帧、终止标记和
RNG仍相同，但自碰撞奖励可差0.2～0.4，随后GAE、梯度和权重出现明显分歧。
原生mjlab自身重跑也出现相同量级的奖励差异。暂时屏蔽自碰撞项的诊断消融中，
连续奖励/return差下降到约1e-6；这不是用于正式训练的修复。

用原生记录的完整rollout和外部RNG流重放，Gobot与原生RSL-RL学习循环连续两次更新的
GAE、40个minibatch梯度、学习率、优化器、观测归一化和参数更新逐位相同。
所以当前证据将明显分歧定位到上游单接触槽选择，不支持“Gobot PPO公式不同”的解释。
这不证明旧单种子最终3.19mm差距的全部因果来源，也不保证长训练轨迹逐位相同。

显式的新稳定对照使用 `maxforce`，让两边每个物理子步读取最大接触力，再使用原来
10N阈值、4子步计数和-10权重。它消除任意首接触选择；由于改变了原生接触归约，
必须两边都从头训练新目录，不能和旧baseline直接混为同一配方。

```bash
# 原生baseline的新配方；其余参数保持原对照设置。
mjlab/.venv/bin/torchrun --standalone --nproc-per-node=8 \
  tests/support/train_mjlab_tracking.py train \
  --reference logs/tracking-comparison/dance2-reference \
  --output logs/tracking-comparison/mjlab-dance2-maxforce-s42 \
  --num-envs 1024 --iterations 30000 --seed 42 \
  --self-contact-reduction maxforce
```

`train_gobot_tracking_comparison.py` 自动继承baseline的归约及确定性设置；检查点contract
记录非默认归约，恢复训练和评估不能悄悄换回另一种归约。旧配方默认仍为 `none`。
评估显式恢复训练配置中的IEEE/TF32设置。正常评估不导入mjlab。

另修复MuJoCo Warp 3.17位掩码兼容：`deterministic=True`现在启用
`CONTACTS | CONSTRAINT | ATOMICS`，而非Python布尔值1对应的CONTACTS一位。
可用 `--deterministic-physics` 对两边显式开启全部确定性选项并记录到实验元数据。
它不能单独消除首接触槽问题，且开销明显，不是默认训练设置。

诊断入口（都只运行短更新，不修改上游源码）：

```bash
.venv/bin/python tests/support/probe_tracking_training.py \
  --engine mjlab --num-envs 64 --updates 2 \
  --self-contact-reduction maxforce --output logs/probe-native
.venv/bin/python tests/support/probe_tracking_training.py \
  --engine gobot --num-envs 64 --updates 2 \
  --self-contact-reduction maxforce --output logs/probe-gobot
.venv/bin/python tests/support/compare_tracking_training.py \
  logs/probe-native/trace.pt logs/probe-gobot/trace.pt --output logs/probe-differences.json

# 同输入重放专门隔离学习循环，不是物理一致性测试。
.venv/bin/python tests/support/probe_tracking_training.py \
  --engine gobot --num-envs 64 --updates 2 --self-contact-reduction maxforce \
  --replay logs/probe-native/trace.pt --output logs/probe-replay
.venv/bin/python tests/support/compare_tracking_training.py \
  logs/probe-native/trace.pt logs/probe-replay/trace.pt \
  --output logs/probe-replay-differences.json --require-exact-ppo
```

原生探针需要匹配的数值库及mjlab依赖；本机可显式追加
`--extra-python-path mjlab/.venv/lib/python3.13/site-packages`。支持torchrun：
各rank写入独立 `rankN/trace.pt`，重放参数可指向含这些子目录的共同输出目录。
完整trace包含模型、梯度和rollout，体积较大，仅用于少量更新的验收。

### 八卡验收边界

同输入重放已在118上通过：8卡、每卡1,024环境、rollout 24、一次PPO更新的
20个minibatch梯度、GAE、优化器、normalizer和更新后权重逐位一致（各rank都通过）。
实际仿真的maxforce配方仍会在少数环境发生接触阈值/终止分叉；原生自身重复八卡运行
也出现同类差异。因此不能把maxforce称为全流程逐位确定性，只能说它移除了任意首接触
选槽造成的主要离散奖励噪声，学习循环本身已有严格同输入验收。

`--deterministic-physics` 在本机64环境能运行，但在118的每卡1,024环境时，原生Warp
确定性内核临时数组申请超过32位shape上限，运行失败。当前八卡生产配方不要开启该开关；
它保留为诊断选项，且不会静默回退。最终3.19mm误差差距仍需新配方的多种子长训练才能
判断是否系统性改善，本次没有启动长期训练。
