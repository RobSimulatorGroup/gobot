# SuperMimic-Reproduced 对齐实施记录

更新至 2026-11-04。新实现已经接入唯一训练入口，旧 Uni-SuperMimic learner、任务语义、
fork/promote、旧配方、对照工具和对应测试/文档已删除。
**已完成训练链路迁移和本地诊断；尚未完成 BONES 从头训练及全基准成绩复现。**

路线见 [对齐计划](supermimic_reproduced_alignment_plan.md)，使用方法见
[训练框架](gobot_training.md) 和 [参考工具](../examples/g1/reproduction/README.md)。

| 阶段 | 当前事实 |
|---|---|
| P0 来源与发布物 | 两份源码固定；发布配置、223 MB checkpoint、完整 benchmark 均通过固定 SHA256；源码已归档 |
| P0 数据身份 | 公开完整评估清单和固定过滤子集已恢复；原始训练库的字节与清单仍未取得 |
| P0 软件栈 | 原始 Python 3.12 / Torch 2.14.1+cu130 / MuJoCo 3.16 / Warp 1.18 环境安装已完成；Gobot 配对栈为 3.17 |
| P1 场景与物理 | 1161 项静态检查、G1 FK/梯度及 CPU→CPU、Warp→Warp 同动作对照已通过；跨 CPU/Warp 失败保留 |
| P2 motion library | 参考重采样 + Gobot 场景 FK 转换已实现，dense/compact、清单和哈希已接入；实际 LAFAN1 走路 CSV 对照通过 |
| P3 学习算法 | replay、terminal rows、WM、BPTT、GAE critic、梯度裁剪/Adam 均已接入；独立完整更新 fixture 回归通过 |
| P4 环境时序 | clean/executed action、bias、startup/reset DR、return sampler、参考帧与 history 顺序已替换；64 步 dense/compact 同策略 GPU 对照完成 |
| P5 训练和恢复 | 完整网络 CPU/CUDA/torch.compile 冒烟、CPU 恢复、2-rank CPU/Gloo 参数同步已验证；实际 2/4 GPU NCCL 仍待验证 |
| P6 正式成绩 | 未执行匹配数据和训练量的从头配对训练；不能用 teacher 或冒烟结果代替 |

**2026-11-04 新增验证**

- 全部 40 条 LAFAN1 已通过 Gobot 场景 FK 转换，共 **441140 帧**，dense NPZ 和 fp16 packed library 均有哈希清单。
- 同一 e8 发布 teacher、相同转换数据、40 个环境、128/500 诊断容量下，完整 LAFAN1 配对结果：

| 运行 | 平均 return | 步数加权 return | 失败 |
|---|---:|---:|---:|
| 参考 + mjlab，3.17 配对栈 | 0.6235608319 | 0.5972900168 | 21/40 |
| Gobot，3.17 配对栈 | 0.6266288506 | 0.6016140894 | 19/40 |

平均差 **+0.0030680186**、失败数差 -2；逐片段 return 最大绝对差 **0.0381211136**。
参考独有失败：dance2_subject1、run2_subject4、sprint1_subject4、walk1_subject1；
Gobot 独有失败：fight1_subject5、run1_subject5。保留所有差异，不用汇总接近宣称逐片段等价。
这使用 BONES teacher 在 LAFAN1 上评估，不是 LAFAN1 从头训练结果或 BONES 全库成绩。

- 4 个 lane 使用独立参考实际采样的 COM/足部摩擦/encoder bias 和执行噪声，24 步同动作比较：
  state/observation 最大差 **3.8207e-5**、reward 最大差 **4.0978e-8**、执行动作最大差 **5.9605e-8**；
  goal、frame、终止/超时一致。输入为注入的相同随机样本，不承诺两套框架 RNG 消耗完全相同。
- compact、每 10 步超时、共 64 步的实际 GPU 对照：replay 的 frame、clip、ep_id、terminal、ep_start、valid、ptr、
  ep_counter 和 done 标签全部一致；观测历史布局、重置后的 target frame 和 sampler 状态一致。
  连续数值仍有物理累计差异，记录在 `reset_parity.json`。
- 真实 G1 replay 和独立 G1 FK，**完整网络 H32 一次更新**：WM 参数/梯度误差 **0**，
  actor 参数最大差 **3.7253e-9**、critic 参数最大差 **1.2573e-8**；全部预设数值门槛通过。
  WM/actor/critic 参数分别为 8815111 / 8891197 / 235009；batch=4，WM 一次 batch、critic 5×4 次更新。
- 修复 friction-only 随机化误触发 startup COM 常量重算；reset DR 在写入参考姿态前应用。
  checkpoint 现在显式保存/恢复各 rank 的 startup COM，并检查恢复 shards 的训练计数一致性。
  配置在初始化物理环境前校验，避免无效 batch/horizon/调度值运行到 GPU 阶段才失败。

原始发布软件栈的相同 40 条 LAFAN1 同权重评估已完成：Torch 2.14.1+cu130、MuJoCo/MuJoCo Warp 3.16、
Warp 1.18 下，平均 return **0.6195513959**、步数加权 **0.5986617315**、**19/40** 失败。
将同一参考换到配对栈后为 0.6235608319、21/40 失败。四个片段失败状态改变，
单片段 return 最大差 0.20719608。此为整套软件栈的影响，不能归因于 MuJoCo 单一版本。
这同样使用 128/500 诊断容量，不是 BONES 的发布 benchmark。

本轮新增实测：

- 独立 fixture 执行固定参考的 replay wrap/terminal/history、两次 WM minibatch、完整 actor BPTT、critic epochs，
  比较 loss、梯度、参数及三套 Adam 状态。使用解析测试骨架；真实 G1 FK 另有对照，不能把它称为完整环境等价证明。
- 实际 `walk1_subject2.csv` 转换共 **13066 帧**。joint pos/vel 误差为 **0**；body position 最大误差
  **9.54e-7 m**，quaternion（对齐符号）**1.79e-7**，线速度 **7.15e-7 m/s**，角速度 **3.22e-6 rad/s**。
- 完整网络 CPU、CUDA、torch.compile 各执行 **45 control steps / 2 learner updates**；缩小 batch 和 H=4，仅作功能检查。
- compact + DR + 观测/执行噪声 + **H=32** 的完整网络 CPU 测试执行 **50 control steps / 4 updates**，所有学习指标有限。
- CPU 恢复从 45 到 90 control steps，总计 4 updates；重新 warmup/refill，不是逐位连续恢复。
- 2 个 CPU/Gloo rank 各 45 步、2 updates，WM/actor/critic 参数差均为 **0**；这不是多 GPU NCCL 测试。
- 64 步 compact 同策略 Warp 对照：目标帧/goal 一致，初始 action 一致，终止/超时标志一致；
  action 最大差 **0.002456**，reward 最大差 **3.14e-6**，混合状态分量最大差 **0.007446**。
  未为长闭环轨迹宣称逐位一致。

完整走路片段同权重试验（`walk1_subject2`，13066 帧/步）：

| 配对运行，容量 128/500 | 失败 | normalized return |
|---|---:|---:|
| 固定参考 + mjlab，3.17 配对栈 | 0 | 0.6029567138 |
| Gobot，同 checkpoint、同片段、同栈 | 0 | 0.5995487229 |

return 绝对差 **0.0034079909**。这是一个完整长片段的 teacher 移植诊断，不是从头训练成绩。
首个 Gobot 35/250 单世界完整片段运行触发 broadphase 容量溢出，异常日志保留为
`gobot_walk_capacity35_failure.log`；参考默认屏蔽溢出警告。128/500 配对用于排除接触截断影响，
不把它改判为正式 35/250 通过。nconmax 是多世界共享容量预算，正式 4096 env/rank 需另行检查容量。

新增实现细节：mjlab 奖励使用最后子步积分前的链接状态、上一步 command 缓存的 re-anchored reference，
而 replay 和 joint-limit 使用积分后的状态。Gobot 在子步之间只刷新运动学，保持接触/solver 状态。
compact fp16 quaternion 在场景重置时归一化，但保留参考的原始 reset quaternion 供 replay，
并保留对应 body angular velocity；没有放宽引擎单位四元数约束。

默认评估强制采用参考 benchmark 的 dense 模式，不能从训练 checkpoint 继承 compact=true。
正式 e8 配方的 policy batch 为 2048（不是仓库默认 4096）。
不中断的 100k 重建配方会得到约 39984 updates；发布 checkpoint 因 92k 恢复历史为 39968。
恢复时 Gobot 保存每 rank RNG/sampler，属于计划中明确的恢复增强，不声称恢复了缺失的原始多卡状态。

本轮诊断实际环境是 Python 3.13、Torch 2.11.0+cu128、MuJoCo/MuJoCo Warp 3.17、Warp 1.17。
它们不是原始发布的 Torch 2.14 / MuJoCo 3.16 / Warp 1.18 栈；同版本的算法/模型对照与
发布栈复现不能混为一谈。所有原始版本影响和完整配对训练仍需单独测量。

**已确认并修正的问题**

- mjlab G1 足部碰撞优先级为 1，Gobot 原先统一编译为 0。新增可存储的
  `PhysicsMaterial3D.contact_priority`，覆盖 MJCF 导入、场景资源、physics snapshot、Python 数据快照及
  MuJoCo 编译，测试覆盖导入→PackedScene→物理 artifact。
- mjlab 世界有一个独立静态 terrain body。`SceneKinematics` 现在按选定基座构建 articulation FK，
  排除无运动关节的世界静态兄弟分支；断开的运动关节仍报错，不能静默丢失控制关节。
- 增加 `SceneStateKinematics`，明确把学习状态的 body-frame 根角速度转换到引擎 FK 的 world frame，
  使用稳定关节/链接名字映射。
- 导出工具调用真正的 reference 配置与 Scene builder，覆盖配置层设置的 PD、armature、碰撞和 solver。
  XML 名称整理经过内存模型字段对照；没有用原始 XML 默认值冒充实际运行模型。
- 静态审计单独列出 plane 的显示网格 size 差异：0.01 vs 0.05。MuJoCo plane 的 size 不决定碰撞边界；
  仍比较平面的类型、变换、材料、接触参数，实体几何体尺寸不豁免。该分类有回归测试。

**已测量结果**

| 测量 | 结果 |
|---|---|
| CMake 构建 | `test_resource_format_mjcf`、`test_physics_server`、`test_packed_scene`、`gobot_python` 成功 |
| 原生相关回归 | 67 项通过 |
| Python 相关回归 | 25 项综合回归通过；补强近目标奖励 fixture 后，12 项定向检查也通过 |
| G1 FK 位置最大误差 | 1.8444e-7 m |
| G1 FK 旋转矩阵最大元素误差 | 3.3440e-7 |
| G1 FK 线速度 / 角速度最大误差 | 6.6936e-7 m/s / 1.1145e-6 rad/s |
| CPU 同动作 32 步，关节位置 / 速度最大误差 | 2.6053e-7 rad / 1.2943e-5 rad/s |
| Warp 同动作 32 步，关节位置 / 速度最大误差 | 9.1630e-6 rad / 2.9530e-4 rad/s |
| CPU、Warp 掩码重置 | 两种同后端对照中，未重置 lane 的读回状态逐位不变 |
| 发布 actor 固定输入对照 | 16 组 930 维历史 + 680 维目标，动作误差 0 |

Warp 对照使用独立 `mujoco_warp` 模型，未复用 Gobot provider 作为参考。
这组诊断使用 2 个环境、deterministic Warp、128/500 容量，不等于正式配方的 35/250。

跨 CPU/Warp 的 32 步比较中，关节位置差达到 0.01832 rad、速度差达到 1.1024 rad/s。
同后端 Warp 参考对照误差小得多，支持“差异主要不来自 Gobot 场景/控制适配”的判断，
但尚未定位具体求解器原因。跨后端长轨迹和完整策略表现不能据此判为通过。

第一份数值 fixture 由独立进程运行固定参考源码产生，记录导入源文件 SHA256，且不导入 Gobot。
它覆盖观测与历史布局、策略/critic、8 项奖励、8 步世界模型 loss、动作/参数梯度和一次 Adam 更新。
完整 BPTT/replay 更新现由第二份 training fixture 覆盖；完整环境/训练成绩另行验证。

**发布结果与配置核实**

Checkpoint：`e8_boneseed_ckpt_0100000.pt`。
SHA256：`0527bab674556f48eaa7d3a27d650d1d14986019fc982d8acecbf6da75bc5ea6`。
实际计数为 `iteration=100000`、`num_updates=39968`。

从发布的逐片段 JSON 重算：

| 集合 | 数量 | 失败 | clip 平均 return | 实际步数加权 return |
|---|---:|---:|---:|---:|
| 原始完整评估 | 129943 | 495 | 0.9301884730 | 0.9247524822 |
| 固定代码名字过滤后的子集 | 129491 | 145 | 0.9314684656 | 0.9250606483 |

后续以这些数据和过滤规则的哈希为准。参考文档的 0.932 不能替代逐片段数据的精确重算。
发布配置来自 92k 后的恢复段：0.0004 是已缩放学习率，0.8 小时是该段时限，
`compact_library=false` 会在提供 library 路径后被代码改成 true，320k 调度与 100k 停止要分别表达。

本机产物根目录：`examples/g1/logs/reproduced_alignment/`。

- `baseline/baseline.json`：源码/发布物身份、精确汇总、训练清单缺失说明。
- `baseline/evaluation_clips.json`、`filtered_evaluation_clips.json`：两套评估集合。
- `model_export_317/`：实际参考模型导出及配置/文件哈希。
- `project_317/assets/supermimic/world_physics_audit.json`：1161 项静态物理检查。
- `project_317/fk_audit.json`：场景 FK 与学习状态适配验证。
- `project_317/control_audit.json`、`warp_reference_control_audit.json`：同后端对照。
- `project_317/warp_control_audit.json`：保留的跨后端失败记录。
- `math/e8_actor_parity.json`：发布 actor 固定输入对照。

下一步扩展完整片段同权重评估，以实际集群资源验证 NCCL 2/4 卡及正式训练吞吐，
使用固定训练/评估清单执行多 seeds 的从头配对训练。需要集群连接、GPU/调度器资源和 BONES 训练库位置。

本轮可追溯产物（均位于 `examples/g1/logs/reproduced_alignment/`）：

- `motion_conversion_parity.json`：真实 CSV、参考 CPU MuJoCo 与 Gobot 场景 FK 转换对照。
- `reference_task.pt` / `gobot_task.pt`：初版 dense 64 步记录；`compact_task_parity.json`：修正 reset 精度后的 compact 对照。
- `train_cpu/`、`train_warp/`、`train_compiled/`：完整网络小批次 smoke；`train_randomized/`：compact + DR + H32。
- `validated_train_cpu/`、`validated_resume_cpu/`：最终源码下的 CPU smoke/resume，含显式优化器计数、源码身份及日志区间聚合。
- `train_gloo/synchronization.json`：两 rank CPU/Gloo 参数同步检查。
- `reference_walk_capacity128.json`、`gobot_walk_capacity128.json`、`full_clip_alignment.json`：完整片段配对及局限。

上一轮 Python 定向回归 **27 项通过**，包含真实 G1 项目、compact/packed 数据一致性、DR 和 timeout terminal rows；无需资产的纯数学/数据边界测试独立运行。

本地验证使用 `build/python` 的新绑定与 Python 包；没有覆盖主 `.venv` 内的旧原生安装。
普通安装/集群使用须按项目构建流程安装新包。本地复验可设置
`PYTHONPATH=/home/wqq/gobot/build/python:/home/wqq/gobot` 并使用诊断解释器。

2026-11-04 产物补充：

- `data_lafan1/manifest.json` / `library.pt`：完整本地 LAFAN1 身份和转换结果。
- `lafan1_alignment.json`：40 条完整 teacher 片段配对，保留六条失败状态分歧。
- `random_context_reference.pt` / `random_context_parity.json`：独立随机物理输入及 24 步比较。
- `reset_reference.pt` / `reset_gobot.pt` / `reset_parity.json`：多次自动重置和 replay 边界。
- `real_update_reference.pt` / `real_update_parity.json`：独立真实 G1 replay 全网络更新。

这些日志/数据仍位于 ignored 实验目录。正式多 GPU 从头训练尚未提交；需要实际训练服务器连接信息、
GPU/调度器配额和 BONES 数据目录，不能把本地有限诊断当作完成结果复现。

本轮 2-rank CPU/Gloo 恢复已完成：45→90 control steps，总计 4 learner updates，
两个 rank 的 WM/actor/critic 参数逐位一致，各优化器计数一致；startup COM 保存在各自 shard 中。
检查报告见 `resume_audit_gloo_continued/synchronization.json`。
软件栈配对报告见 `reference_stack_comparison.json`。

2026-11-04 最终定向回归：**36 项通过**。覆盖独立数学/完整更新 fixture、真实场景 DR/compact/恢复、
子步观察与 friction-only 更新、完整片段报告校验和任务入口。`git diff --check`、43 个相关 Python 模块语法检查
及两个 Slurm 脚本语法检查通过。没有启动集群长训练。
