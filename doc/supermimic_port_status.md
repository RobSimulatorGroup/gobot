**SuperMimic → Gobot 状态：2026-11-03**

当前训练工程链路已具备正式配方所需的随机化、checkpoint、续训/fork 和集群入口。完整迁移验收仍有一个新增失败片段，尚未启动正式长训练，也未提交任何远程集群作业。之前展示的走跑/舞蹈预览使用独立参考 checkpoint；本轮新生成的 Gobot checkpoint 来自短训练功能验证，不能当作已学会走路的策略。

集群操作说明在 [cluster/README.md](../examples/locomotion/cluster/README.md)。实际提交还需要集群入口、调度器和 GPU 型号/数量；当前不假设这些信息。

**已实现**

- 参考兼容的 29DoF G1 `.jscn` 与平地场景。修复 MJCF 网格碰撞、固定链接、位置执行器 kv、无质量固定链接、接触摩擦模式及静态平面的导入/编译语义。
- `ArticulationBatch` 复用 CPU typed batch 与 Warp RobotBatchView，统一命名 PD 目标、固定步进、掩码重置、接触聚合及场景同步。CPU 使用场景执行器参数，不叠加普通场景步进的默认阻尼。
- `SceneKinematics` 从 Gobot physics snapshot 生成可微 FK，支持固定、转动/连续和移动关节及偏移关节。
- 独立 `gobot.rl.motion_tracking` 和 `gobot.rl.supermimic`。运行时不依赖 UniLab；网络、replay、可微奖励和 BPTT 保留参考结构。
- 真实的每环境随机化：启动时采样机器人摩擦 `[0.3,1.2]`、躯干质心偏移 x±2.5 cm / yz±5 cm；每次 episode reset 采样关节目标偏置 ±0.01 rad。地面训练摩擦保留 0.3，名义评估为 1.0。context 与实际物理参数绑定，不做“仅修改观测”的随机化。
- Warp 同时扩展输入参数和派生惯量/加速度常量，保证每个世界独立；CPU 通过已有批量物理参数请求应用相同配置。重复设置偏移不会叠加。
- 普通状态读取与掩码重置使用运动学刷新，保留已完成子步的接触和求解器状态；不对其他环境额外做完整动力学求解。
- `TrackingTrainingSession` 保存网络、optimizer、normalizer、task sampler、物理 context、随机数和阶段信息。resume 保留阶段、开始新 episode 并重新填充 replay；fork 保留 actor/normalizer、重建 optimizer/critic/replay，并冻结 actor。
- 迁移预热的独立评估：1/8/32 步世界模型误差、32 步真实回报 critic 残差、物理 rollout 与冻结 actor 一致性。没有预声明阈值时只报告指标，不能解冻 actor；promotion 必须对应同一个 checkpoint 和训练 contract。
- 每 15 分钟和训练结束时保存不可变的预览 checkpoint/请求。可由独立图形 worker 渲染；`render` 模式可由 GPU trainer 启动同卡 worker，`queue` 模式供计算集群留存后处理。渲染失败不会被标记成成功或无限自动重启。
- 普通 Python 与 Slurm 入口：完整网络 smoke、512/1024/2048 吞吐扫描、参考/Gobot 配对评估、种子 42/43 的独立训练。集群配方支持 per-job 墙钟限制和使用量记录，不把旧本地追加窗口复制成集群总额度。

正式训练入口为 `examples.g1.train_tracking`；此前 `examples.g1.tracking smoke-train` 仍是固定物理的小网络测试。正式配方要求完整网络、32 步 BPTT 和匹配机器人签名的完整迁移验收报告，不能用 smoke 的正常退出替代验收。

**本地实测**

| 检查 | 结果 |
|---|---|
| C++ MJCF / PhysicsServer / SimulationServer / 场景序列化 / native batch | 12 / 17 / 8 / 23 / 8 项通过，共 68 项 |
| Python 综合回归 | 61 项通过，1 个 MPS 专用测试跳过；新增 checkpoint/promotion/return-target 检查另行通过 |
| 参考算法固定 replay 对照 | loss、梯度、一次 optimizer 更新后的参数逐项一致 |
| G1 随机姿态 FK | 位置/旋转 ≤1e-5，线角速度 ≤1e-4；有限差分覆盖位置、旋转和速度 |
| 静态物理对照 | 机器人 1,076 项、含地面/solver 的世界 1,103 项通过 |
| 最终名义 CPU/Warp 短程对照 | 状态最大差 1.812e-5；接触力差单独记录，最大约 0.0185 N |
| 随机化 CPU/Warp 对照 | 状态最大差 2.027e-5；GPU 实际参数读回、派生常量按环境独立、掩码重置保持其他求解器状态均通过 |
| 参考/Gobot 同动作 32 步 | 未来目标完全一致，状态最大差 4.33e-5，奖励最大差 5.22e-8 |
| 完整网络 GPU 短训练 | 64 环境、完整参考网络、32 步 BPTT、BF16、编译开启、真实随机化，6 个 iteration 完成；含首次编译约 293 s |
| 完整网络 GPU resume | 从 iteration 6 恢复并完成 iteration 7；约 32 s |
| 完整网络 GPU fork | 新种子运行 3 个 iteration；actor 及其 normalizer 逐位不变，世界模型发生更新，新 critic 可训练；约 52 s |
| 独立预热评估 | 已生成模型/critic/物理指标；本次为 metrics-only，不授权解冻 |

这些短训练 checkpoint 确实由 Gobot 环境中的训练产生，但更新次数很少，不能据此声称策略已具备可用步态。全网络 smoke 与旧参考的 512 环境长训练也不是相同规模实验。

**完整冻结 checkpoint 迁移对照**

所有下列完整评估使用同一个原始 `ckpt_012000.pt`，没有重新训练：

| 实现 | 完整片段 | 平均归一化 return | 走/跑/冲刺 |
|---|---:|---:|---:|
| 原始独立参考，3.11 | 33/40 | 0.731845 | 18/18 |
| 独立参考，3.17 | 34/40 | 0.746414 | 18/18 |
| Gobot，初次 full-forward 刷新 | 34/40 | 0.751171 | 18/18 |
| Gobot，仅修改步后运动学刷新 | 32/40 | 0.726629 | 18/18 |
| Gobot，继续修正掩码重置读取（最新） | 34/40 | 0.735782 | 18/18 |

最新平均差 -0.010632 在 0.02 范围内，但仍新增 `dance2_subject4` 失败，因此迁移门槛仍未通过。此前新增的 `fallAndGetUp3_subject1` 失败已不再出现。不能因总完成数相同或走跑全部完成，把全库门槛改判为通过。

单环境舞蹈窗口可越过之前失败时刻，而完整批量评估仍有差异；短窗口不是完整片段验收。短程同动作结果也尚未唯一确定长闭环差异的来源。场景/模型序列化允许 float32 与 XML 表示误差；不宣称位级相同的动力学模型。

完整确定性参考/Gobot 配对仍待完成。确定性模式的本地 100 步筛查约 44 s（初始化约 8 s），长评估单控制步约 0.5–0.6 s，原本地两小时任务上限偏短，未形成完整结果。已提供两张同型号 GPU 并行运行、每任务六小时的集群脚本，不再以该本地小窗口作为最终计算安排。

**关键产物**

本地目录 `examples/locomotion/logs/tracking_port/`：

- `g1_masked_reset_full40/` 与 `masked_reset_comparison.json`：最新完整对照和未通过判定。
- `r1_reference_317_compat/`：独立参考 3.17 完整结果与隔离兼容补丁身份。
- `reference_trace/`、`final_readback_parity/`、`randomization_parity_v2/`：同动作、名义/随机化与重置诊断。
- `gpu_full_network_randomized_smoke/`、`gpu_full_network_resume/`、`gpu_full_network_fork/`：真正的 Gobot 短训练、恢复和 fork checkpoint。
- `gpu_full_network_fork/validation_metrics_v2.json`：独立预热测量，metrics-only。
- `videos/comparison.mp4`：旧参考 checkpoint 在 Gobot 执行/渲染的走、跑、冲刺、舞蹈各 8 秒预览；不是本轮短训练 checkpoint 的视频。
- `g1_tracking_cluster_bundle.tar.gz`：当前源代码、原生实现、场景、运动数据、冻结参考 checkpoint、独立参考源/资产、配方和提交脚本。标记为未获策略准入，实际任务尚未提交集群。

新的训练入口保存逐文件校验的源码归档、运行 contract、硬件/依赖和数据身份；历史早期评估只保存源码哈希，不能把当前包宣称为每份历史运行的逐字节代码副本。

后续先在集群完成同协议配对并定位剩余舞蹈差异；通过后启动正式训练或显式 fork 适应。命令学生蒸馏、PPO/AMP 地形适应与 SkillController 仍属于后续里程碑。
