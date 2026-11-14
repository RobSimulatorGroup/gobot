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
