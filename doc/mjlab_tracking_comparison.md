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
