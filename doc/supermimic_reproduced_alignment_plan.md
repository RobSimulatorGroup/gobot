# SuperMimic-Reproduced → Gobot：G1 训练与结果对齐计划

计划制定于 2026-11-03。2026-11-06 已完成单 seed 八卡训练与全量评测，
见 [100k 实验报告](g1_bones_100k_results.md)；同栈同预算多 seed 复现仍按本计划推进。

目标是用 **Gobot 自己的场景、物理批处理和 Python 训练接口**，从头训练 G1，复现
SuperMimic-Reproduced + mjlab 的指定实验。SuperMimic-Reproduced 是此次算法与任务语义的主基线。
不保留 Gobot 旧 SuperMimic 算法、配置、checkpoint 或命令的兼容路径。

保留并复用已验证的引擎边界：`.jscn`、PhysicsSceneCompiler、ArticulationBatch、SceneKinematics、
MuJoCo CPU/Warp provider、任务注册、独立视频与运行记录。Gobot 训练进程不依赖 mjlab、UniLab，
不绕过 Gobot 场景另建一份训练用机器人。参考进程只负责导出测试数据、运行基线和比较结果。

这份计划不把已完成的入口整理、旧 Uni-SuperMimic 对照或短训练测试计为新基线对齐。
以下是实验设计与验收门槛，不是所有项目均已完成的声明。
最新单 seed 训练与完整评测记录在 [正式结果](g1_bones_100k_results.md)。发布成绩比较不能代替同配置、同训练量的多 seed 配对实验。

**1. 固定基线与验收对象（P0）**

固定本地参考仓库 `ba6105349069bf040f1fdad60873d13138d505a2`，mjlab
`8bb2bc9e187797689ec82c78c35019afd2fd0376`。保存源码归档、lock、原始实验配置、
模型资产、数据来源 revision、转换/过滤脚本和 SHA256。采用参考实现代码时保留原作者署名与许可文本。

先取得发布的 e8 checkpoint 及其嵌入配置、对应训练数据清单、逐片段评估结果。
发布的 e8 配置、checkpoint 和完整逐片段报告现已下载并通过固定 SHA256 校验；
原始训练库清单与动作数据仍需独立准备，不能由评估片段名称推定训练库身份。
原始配置或训练清单无法取得时，明确记录“重建配方”，不能声称逐项复现原实验。

建立三个独立实验目标：

| 实验 | 训练 / 评估口径 | 参考自报结果 | Gobot 的目的 |
|---|---|---|---|
| LAFAN1 default | 全部 40 条；93 维观测、680 维目标；整条播放 | 0.56–0.59，3–9 / 40 失败 | 小规模完整算法对齐 |
| LAFAN1 anchor | 全部 40 条；目标额外加 9 维锚点位姿 | 0.904，3 / 40 失败 | 单独的增强观测实验 |
| BONES-SEED e8 | default 观测；匹配 e8 训练集合；全库与过滤子集分别评估 | 全库 129943 条：0.930、495 失败；过滤后 129491 条：0.932、145 失败 | 最终主要结果复现 |

上述成绩来自本地参考 README / EXPERIMENTS，不是 Gobot 已有成绩。论文数字不是此次直接验收基线。
参考 README 把主结果概括为约 0.93；实验日志明确写着，e8 用 129943 条训练，
新增杂技关键词过滤只改变评估集合。必须分别固定 `train_manifest` 与 `eval_manifest`。
如果未来两边都改用 129491 条训练，另立新实验，不能继续把它称为原 e8 的精确复现。

依赖分成两条明确的基线：原始发布环境保留 Python 3.12、Torch 2.14.1+cu130、
MuJoCo/MuJoCo Warp 3.16、Warp 1.18，用于核验发布结果。Gobot 原生批处理已依赖
MuJoCo 3.17 的 mjBatch，CMake 明确拒绝缺少该 API 的 SDK，不能只降级 Python 包。
因此配对实验采用参考实现与 Gobot 同为 MuJoCo/MuJoCo Warp 3.17，并单独测量参考实现
从 3.16 到 3.17 的版本影响。配对实验应统一 Torch/Warp 和原生 Python ABI，记录新的环境锁。
目前完成的数值及资产诊断使用现有 Torch 2.11 / MuJoCo 3.17 / Warp 1.17，
不能将它们写成已完成原始发布栈或最终配对训练栈的验证。

精度也固定：审计 reference 的实际 dtype、TF32、compile、物理解算选项和设备映射。
当前 Gobot 默认 BF16，参考训练源码没有对应的 autocast 路径；首轮不继承 BF16 默认值。
关闭 TF32/compile 做数值测试，训练使用冻结配方中的原设置；性能优化另做消融。

产物：`reference.lock.json`、`experiment_specs/`、数据与资产清单、原始 checkpoint 身份。
通过条件：每个目标实验有唯一、可重建的配置与数据集合；未知项有明确记录。

**2. 替换 G1 资产来源，建立场景与物理对照（P1）**

重写 `examples/g1/prepare_tracking.py`：删除对旧 Uni-SuperMimic checkout 的依赖。
从固定 mjlab 版本的 G1 资产和最终运行配置导出机器人、碰撞、驱动与场景参数，导入 Gobot `.jscn`。
只读原始 XML 不够：mjlab 配置层覆盖的 PD、armature、碰撞、默认姿态、动作缩放和 solver 也必须导出。

按稳定名称比较，而不是假定两边内部数组索引相同：

- 29 个动作关节、自由基座、14 个跟踪链接；预期状态维度 nq=36、nv=35。
- 质量、惯量及惯性坐标系、关节轴/零位/限位、armature、阻尼、摩擦损失。
- 所有有效碰撞形状、足部碰撞代理、接触掩码与 exclusions、地面材料和接触参数。
- 默认关节位置、action scale、kp/kd、effort/velocity limits、执行器实际表达式。
- 200 Hz 物理、4 子步 / 50 Hz 控制、积分器、solver、迭代数、容差和容量。

测试零姿态、随机合法姿态、静止站立、单关节阶跃、同一动作序列和包含接触的短 rollout。
检查 reset 是否清理目标 lane 的速度、控制和 solver 状态，且不改变其他 lane。
区分积分后的 qpos/qvel 与派生链接状态的更新时间，记录每一步读回时点。

复用 `SceneKinematics`，与 reference FK / MuJoCo 链接原点的位姿和速度比较，覆盖梯度。
不把质心速度当成链接原点速度；不把世界系角速度直接填入模型所需的 body-frame 角速度。
任务层使用 Gobot 命名的 typed batch，不引入原生 MuJoCo 指针或编辑器对象。
若缺少形状选择、有效材料读回等能力，只向物理批处理边界增加通用功能。

产物：新 G1 场景、robot profile、compiled-model 对照、FK/控制/掩码 reset 报告。
通过条件：离散拓扑与控制映射完全一致；连续量误差在预先记录的表示精度范围内。
意义明确的 C++ 改动必须 CMake 构建并运行场景导入、物理批处理与序列化测试。

**3. 数据转换、紧凑动作库与评测口径先对齐（P2）**

替换仅支持 LAFAN1 CSV 和 subject 名字正则的路径，采用显式数据清单。
保留数据版本和训练/验证/评估用途，不把 LAFAN1 的 subject1–3 留出实验当作全 40 条复现。

- LAFAN1：30 Hz，位置米、xyzw 四元数、关节弧度；转换为与 reference 一致的 50 Hz 序列。
- BONES-SEED：120 Hz，位置厘米、xyz 欧拉角度、关节角度；显式转换单位与旋转约定。
- 对齐帧数取整、时间端点、slerp 小角度处理、速度差分、关节顺序和 clip 起止边界。
- 分别保存原始 CSV、转换结果、过滤结果的哈希；过滤失败要保存具体规则与证据。
- 实现 71 维 fp16 紧凑状态库，按批次转换 fp32，在线使用 Gobot SceneKinematics 求跟踪刚体。
- 对齐 fp16 量化及四元数归一化语义；不能以“数学等价”为由改变 reference 的输入。
- 全量库常驻每个 rank 的设备；检查启动内存、临时拷贝和编译峰值，不重复展开全库 body FK。

同时写独立评测 scorer。已确认的关键差异：

```
reference benchmark_return = sum(reward) / (0.1 * executed_steps)
当前 Gobot 评分             = sum(reward) / (0.1 * expected_full_clip_steps)
```

新对齐主指标采用 reference 的 executed_steps 分母；另记录完整片段 coverage、失败时刻和
expected-length return 作为诊断，分别命名。提前失败不能因主指标较高而被计为完成。
实际末帧参与哪一次控制步、超时标志何时出现，要用 reference trace 验证，不能只假设 frames-1。

评估使用确定性策略、首帧初始化、无观测/动作/reset 噪声、无 DR、足部摩擦 1.0，
评估时去掉训练的 10 秒 episode 截断。物理解算确定性以实际配置与重复实验确认。
实现固定数量 lane 的 clip 队列，不能为十几万条 clip 创建十几万个环境。
支持按 clip ID 分片；合并检查 checkpoint、协议与清单一致，并拒绝重复、缺失或额外 clip。
非有限值、未完成分片、人为步数上限均不能输出完整 benchmark 的成功状态。

产物：LAFAN1、BONES-SEED 转换与库构建工具；固定清单；独立 scorer 和分片评测器。
通过条件：相同逐步轨迹产生相同分项与汇总指标；合成短片段覆盖所有结束/失败/队列边界。

**4. 按新基线替换算法（P3）**

替换 `python/gobot/rl/supermimic/` 当前旧实现，不加 legacy 开关。
将网络、state spec、world-model trainer、imagination、policy trainer、replay 和调度职责分开。
不机械复制 mjlab manager 依赖；纯 Torch 数学与训练代码可以移植，仿真和 FK 接口注入 Gobot 服务。

必须对齐的具体项目：

| 项目 | 目标 |
|---|---|
| Actor 输入 | 每帧 93；10 帧、term-major、旧→新；默认移除旧 6 维参考旋转 |
| Motion goal | 每帧 68，10 帧 680；anchor 实验单独追加 9，合计 689 |
| 输入统计 | 固定 scale，删除 actor/critic/world model 的旧 RunningNorm 路径 |
| MLP | 固定每层宽度、SiLU、bias、输出初始化；移除旧 world model LayerNorm，不只匹配层数 |
| State | 明确 71 维字段顺序、wxyz、局部原点、世界系线速度与基座系角速度 |
| Dynamics | 70 维速度增量与位置修正、显式积分、固定输入/输出/误差 scale |
| Termination head | 4 输入、64×64 隐藏层、BCE 标签；取代旧解析 sigmoid 软终止 |
| World-model loss | 8 步自回归；逐状态维度求和、再按窗口平均，含终止 BCE |
| Actor loss | H=32；存活权重和 critic bootstrap；按 reference 使用 `-J/H` |
| Critic | 相同特征、漂移项、固定 scale、TD(λ)，gamma=.99、lambda=.95 |
| Optimizer | Adam 参数、actor betas、grad norm/value clipping、更新顺序逐项一致 |

测试分两层：

1. 相同输入张量、权重、采样索引和噪声，比较输出、每项 loss、梯度，以及一次 optimizer 更新后的参数。
2. 比较 1/8/32 步 rollout、survival、bootstrap 和 TD targets；测试冻结世界模型参数时仍保留
   对预测状态的梯度，防止意外 detach。

单元测试先禁用 compile / TF32；再在训练精度与 compile 设置下验证一次。
参考输出由独立参考进程生成，Gobot 测试不能调用自身函数制造“期望结果”。
参考参数导入仅作为数值对照工具，不是保留旧 Gobot checkpoint 兼容。
正式训练首版只提供从头训练与同版本续训；删除旧迁移用的冻结 actor 适应 / promote 流程及其参数。
旧 Uni-SuperMimic 启动器和对照 fixtures 从活动工作流移除，由固定新参考的独立 oracle 工具替代。

产物：新 SuperMimic 算法、固定 replay/权重 fixtures、loss/梯度/更新逐项报告。
通过条件：同一数学输入下的一步学习更新对齐，而不只是 loss 有限或下降。

**5. 对齐环境动作、奖励、噪声和采样（P4）**

重写 `motion_tracking/mdp.py`、`env.py`、任务配置和 replay 的接口契约。

- **动作噪声**：同时保存 clean action 与 executed action；上一动作观测和 action-rate penalty
  使用 clean action，世界模型拟合 executed action。移除旧 learner 的 ±20 额外裁剪，除非
  冻结 reference 的实际配置证明存在同样裁剪。
- **Encoder bias**：reference 观测是 `q + bias - default_q`；当前 Gobot 是减 bias，并将其加入 target。
  逐层审计 mjlab action/actuator 的有效 set-point，分别测试观测和实际执行力矩，不只改符号。
- **奖励锚点**：跟踪、xy/yaw re-anchoring 与根部奖励使用 `torso_link`；
  高度/倾斜硬终止仍使用 `pelvis`。保留 8 项奖励和 0.02 秒缩放，并逐项对拍。
- **随机化**：足部形状摩擦与 encoder bias 在 reset 时采样，躯干 CoM 在 startup 采样。
  删除当前“全机器人碰撞形状摩擦 startup 采样”的任务语义；地面参数按实际参考模型导出。
- **Reset**：相同位置/速度/关节噪声、欧拉角扰动乘法顺序、soft joint limits；
  不用旋转向量噪声或 hard limits 冒充相同 reset。
- **动作采样**：50% uniform / 40% hard / 10% memory；相同 1 秒 bins、初始值、温度、
  EMA、memory 更新率和截断处理。移除旧的仅回写 episode 起始 bin 的估计方式。
- **Replay**：terminal row、terminated/truncated、frame/clip ID、历史 backfill、环形覆盖边界
  与 clean/executed action 全部明确；采样窗口不得跨 episode/clip，终止头必须取得正标签。
- **时序**：记录 action → substeps → reward/termination → reference advance → reset → history
  的实际参考顺序，逐时点对齐。特别检查预重置最终状态、引用帧 t/t+1、链接派生量是否滞后一子步。

物理读回具有时间差异时，应在后端明确采样时点；不能为靠近一个分数而随意额外调用 forward，
改变已完成子步的接触/solver 状态。reference 确认存在的行为和决定修复的行为分别记录；
修复行为先在两边一致验证，原复现与修正实验分开报告。

产物：观测/动作/奖励/噪声/采样状态规范，step trace 对照和真实 CPU/Warp 回归。
通过条件：名义状态、相同注入随机样本、结束边界下的输入/奖励/终止/重置都对齐。

**6. 统一计数器，完成同步多卡与恢复（P5）**

删除旧 `iteration = learner update`、64 步 warmup、500 次 world-model-only 默认阶段。
使用含义明确的计数器：`control_steps_per_rank`、`global_env_transitions`、
`learner_updates`、`world_model_optimizer_steps`、`policy_optimizer_steps`、`critic_optimizer_steps`。

按参考行为设置：40 个采样控制步 warmup，UTD=0.4，每轮 learner update 的 10 次 world-model batch、
一次 actor 更新和 5 epochs × 4 minibatches critic。恢复后按恢复点重新 warmup/refill replay。
所有数据量比较使用实际计数，不从“100k iteration”字面推测。

调度必须分离：训练计划总步数、学习率衰减终点、实际停止步数。
e8 按 320k control steps 配置调度，约 250k 结束余弦衰减，却在 100k 停止；
把计划总步数改成 100k 会提前衰减，构成不同实验。
100k control steps 在 UTD=.4 下约 40k learner updates，具体值受 warmup/恢复影响，以记录为准。

实现真正的同步数据并行：

- 每 rank 独立环境、replay、sampler 和 RNG；使用 `seed + 1000*rank` 等冻结参考规则。
- 同步 world model、actor、critic 初始参数和固定 buffers；backward 后平均梯度，再裁剪、更新。
- 记录 per-rank/global batch 与 samples；按参考 lr×sqrt(world_size)，恢复时不能重复缩放。
- 在 CUDA/Warp 初始化前完成 rank 到 GPU 的映射，尊重 Slurm 的可见设备 / UUID，
  不假定 LOCAL_RANK 就是物理卡号；正式训练不允许多个 rank 意外共享 GPU。
- 所有 rank 的 update 次数和 collective 顺序一致；非有限值、停止信号与异常要统一退出。
- 用 CPU/Gloo 小型数值测试验证平均梯度，再用真实 2 卡、4 卡验证 NCCL、CUDA graph 和吞吐。
- checkpoint 原子保存共享权重/optimizer/计数器，以及每 rank 的 RNG、sampler 和任务状态；
  不把 rank0 sampler 广播成所有 rank 的状态。保存相同 world size 下的恢复测试。
- 沿用 reference 的新 episode / replay refill 恢复语义，不声称是物理轨迹逐位连续。
  world size 改变视为新实验，首版不做弹性恢复或跨旧格式兼容。

参考 e8 有 92k checkpoint 续训历史，可能缺少非 rank0 sampler/RNG 的精确保存。
优先做两边同配方的不中断从头训练；另测两边一致的恢复流程。
历史 e8 恢复状态缺失属于复现偏差，不能靠伪造随机数状态消除。

产物：分布式 launcher、清晰计数/调度、新 checkpoint schema、Slurm 同步训练与独立种子任务。
通过条件：固定合并 batch 的单卡更新与多卡平均梯度更新一致；实际 2/4 卡不死锁、停止与恢复完整。

**7. 同一权重在两套环境中评估（P6）**

在从头大规模训练前，用发布 e8 权重和可取得的 LAFAN1 权重做冻结策略对照。
缺少 LAFAN1 权重时，先训练一个可重复的 reference 检查点；不为缺失来源编造 checkpoint 身份。

先做三个诊断：固定观测的 action 对照；固定动作序列的物理对照；闭环完整片段对照。
闭环分歧要定位到第一处状态/输入/控制/奖励差异，并保存失败前窗口，不只比较最终平均值。
舞蹈、跳跃、起身与接触切换必须覆盖，不能只挑容易的走路片段。

对齐门槛在看最终成绩前冻结。建议的数值起点（工程目标，不是现有测试成绩）：

| 检查 | 建议门槛 / 方法 |
|---|---|
| 名称、关节/链接映射、frame/clip、布尔 mask | 精确一致 |
| 固定状态的 FP32 obs / goal / reward / action | atol 1e-5、rtol 1e-4；记录 max 与分项 |
| 单次学习更新 | loss/梯度满足同类容差，参数更新 atol 1e-6 / rtol 1e-4；近零量看绝对误差 |
| FK | 由双精度 oracle、有限差分与场景表示精度确定；逐项记录位置、角度和速度单位 |
| 同动作物理短程 | 分开报告米、弧度、m/s、rad/s 的误差；1/8/32 步与接触切换分别建立门槛 |
| 长闭环 | 完整片段回报、失败集合/时间、coverage、分类指标；不要求长期接触动力学逐位一致 |

物理误差阈值由 reference 自身重复、CPU/Warp 数值差和场景表示精度在校准样本上确定，
在独立验证样本上使用；不能在看到失败后放宽阈值。用绝对阈值与相对基线同时约束，
不能单凭“参考也会波动”接受明显回归。

产物：固定权重端到端对照报告，协议与 checkpoint/data/source 身份绑定。
通过条件：输入和学习数学对齐；短程物理误差有解释；完整片段差异满足冻结门槛。

**8. 从头训练 G1：由走路到完整基准（P7/P8）**

以下阶段均保留完整正式网络与 H=32；小网络只用于单元测试，不作为成绩验收。
种子先用 reference 的 seed 0，再以 1、2 做配对独立训练。旧 Gobot checkpoint 不参与初始化。

| 实验 | 训练安排 | 必须交付的验证 |
|---|---|---|
| E0 全网络联通 | 64–256 环境，数百至约 1000 control steps | 实际物理采样、WM/BPTT/critic 更新、保存、恢复、finite 指标 |
| E1 G1 单动作走路 | walk1_subject1；4096 环境；从 20k 到约 40k control steps，按固定检查点评估 | 同一完整长片段从首帧完成、配对曲线、跟踪与接触视频 |
| E2 LAFAN1 default | 全 40 条；单卡 4096 环境；先匹配 reference 的约 28k control-step 检查点 | 40 条完整评估、各类失败和漂移、3 seeds 配对结果 |
| E3 LAFAN1 anchor | 显式 anchor 配方；约 48k control steps | 与 anchor 基线单独比较，不能混入 default 成绩 |
| E4 BONES 工程预热 | 全量紧凑库；4 rank；先 1k，再 5k control steps | 实际 per-rank 显存、吞吐、sampler/梯度同步与分片评估 |
| E5 BONES e8 复现 | 4 rank × 4096 环境；WM / policy batch 各 2048 每 rank；lr×2；100k 停止，保留 320k 调度 | 全 129943 条与固定 129491 子集双报告、每片段 JSON、3 seeds |
| E6 延长学习 | 仅在 E5 达标后延至计划 320k，保持协议 | 单独报告改进，不能用更多算力后的成绩冒充 100k 复现 |

E1 尚未学会完整走路时，对比 reference 相同训练量的行为，按数值/环境/采样/更新链定位问题。
不通过临时更换奖励、加入姿态反馈或挑短窗口来改判 E1/E2 成功。
E2 的目标是复现该仓库的 default 行为，不能拿论文 0.82 或 anchor 的 0.904 强行要求纯本体版本。

E5 每 4k control steps 保存评估候选；使用固定 2000 条子集监控，最终必须完整评估。
视频每 15 分钟或指定 checkpoint 由独立 worker 生成；走、跑、起身、跳跃、旋转以及最差失败片段均覆盖。
训练视频采用确定性动作，参考 ghost 与策略放在同一世界坐标中；不自动重置成“继续成功”。

**9. 结果验收、资源与交付（P9）**

区分“发布成绩目标”和“同硬件重跑的配对基线”。发布成绩提供方向，配对重跑用于判断
Gobot 是否在相同协议和训练量下产生系统性回归。

建议在 P0 完成后冻结以下工程验收值，不能在最终评测之后修改：

- 主目标为 BONES 固定 129491 评估集约 0.93 return、约 0.11% 失败。
- 数值复现工程目标：跨种子平均 return ≥0.92、跨种子平均失败率 ≤0.15%；这是本项目建议容差，
  不是论文或参考仓库规定的门槛。同时列出每个 seed 的原始失败数和最差 seed。
- 同配方配对重跑：跨种子平均 return 差不低于 -0.01，失败率差不高于 +0.03 个百分点。
  不得只选最好的 seed；不能拿不同观测、数据过滤、总样本量或训练时长的模型配对。
- LAFAN1 分 default / anchor，各与实际配对基线比较；建议 mean return 差不低于 -0.02，
  平均失败数差不超过 2/40，并报告全部 per-clip 变化。该门槛仍需 P0 固定。
- 报告走/跑/跳/起身/舞蹈/翻转等分类结果。分类退化需解释，不能被大量容易片段掩盖。
- 以 seed 为主要独立重复单位；clip 配对或 bootstrap 只是补充分析，不能把十几万相关动作
  当成十几万个独立训练重复。三 seed 能提供基本复现证据，但不等同于严格统计等价证明。
- 性能在同型号 GPU、同软件栈、同批量下测量；建议先达到参考有效训练吞吐的 70%，再优化。
  记录 compile 冷启动与稳态、全局环境步、各 optimizer 更新、NVML 总显存和 Torch 显存。
  性能达标与策略质量达标分别报告。

资源按集群规划，不按开发机显存缩小正式算法：

- 训练位：4 张同型号 GPU，至少 32 GB / 卡，优先 40–80 GB 留出峰值余量；有同型 5090 时最便于比较。
- 独立评估位：1 张 GPU；全库评估可用更多卡按 clip 分片。
- 数据位：建议 32 CPU cores、128 GB RAM、约 300–500 GB NVMe 起步，下载后按实际展开量复核。
- 更多资源优先用于不同 seeds / 两框架配对 / 独立评估；直接把 4 卡变 8 卡会改变全局 batch 与 lr，
  应当单列规模实验。
- 参考自报 4×5090、7.1 小时约为 28.4 GPU·h / run；3 seeds × 2 框架是约 170.4 GPU·h 的
  粗略训练占用参照，不含开发、转换、编译、评估与重跑。其他 GPU 的耗时必须由 E4 实测估计。
- 准备可重复安装的训练包/镜像和锁定依赖；作业记录 GPU、驱动、CPU 绑核、网络、源码、数据和环境变量。
  执行前需要集群连接方式、调度器/队列、GPU 配额和数据目录；本计划不假定已取得这些信息。

最终交付：

1. Gobot 内置 SuperMimic 新实现、G1 场景/任务、LAFAN1/BONES 数据流水线与同步多卡训练器。
2. 同状态数学、单次更新、物理/控制、同权重闭环、独立训练五类对照报告。
3. 全部 seeds 的 checkpoint、resolved config、计数器、学习曲线、逐片段评测与视频。
4. 一条从准备资产到训练、恢复、分片评估、报告生成的正式工作流。
5. 新格式 checkpoint 与任务 manifest；删除旧算法文件、旧格式读取和指向旧参考的活动配置/测试。
   旧实验日志可以作为只读历史证据，不再参与正式运行；其他仍在使用的 PPO/AMP 算法不属于此次删除范围。

完成定义：不仅能运行训练，也能给出绑定版本/数据/训练量的 G1 全基准结果，满足冻结的数值与质量门槛。
参考 teacher 的零样本移植通过不能替代 Gobot 从头训练；能跟踪走路动作也不能直接宣称
已完成速度命令控制、复杂地形、蒸馏或真机部署。这些作为对齐后的下一阶段。

**10. 实施顺序与文件清单**

依赖顺序：`P0 → P1/P2/P3 → P4 → P5/P6 → E0/E1 → E2/E3 → E4/E5 → P9`。
P1/P2/P3 可独立开发，但共享 P0 的规范；P5/P6 可在接口固定后并行推进。
同步训练和大量算力不能替代前面的数值与环境验证。

| 当前位置 | 计划处理 |
|---|---|
| `python/gobot/rl/supermimic/{config,networks,geometry,learner,replay,contracts}.py` | 用新数学、replay 和调度替换；拆出 dynamics/world-model/actor-critic 职责 |
| `python/gobot/rl/motion_tracking/{config,env,mdp,motion,dataset}.py` | 替换状态、观测、动作/DR、采样与动作库 |
| `python/gobot/rl/motion_tracking/{evaluate,manifest,training,experiment,runner}.py` | 新评测协议、计数器、manifest 和分布式运行/恢复 |
| `python/gobot/rl/{articulation,kinematics}.py` 与 physics providers | 复用中立 API，仅补缺失的通用状态/随机化/时序能力 |
| `python/gobot/rl/tasks/`、`train.py`、`training/` | 保留注册入口，接入新配方、设备映射、分布式记录和独立 worker |
| `examples/g1/prepare_tracking.py`、场景与 `train/*.json` | 采用固定 mjlab G1 来源，删除旧参考配方，新增目标实验配方 |
| `motion_tracking/validation.py`、`examples/g1/validate_tracking_adaptation.py` 及旧 fork/promote 测试 | 删除旧冻结适应流程；固定权重对照由独立评测工具承担 |
| 旧 Uni-SuperMimic reference runner / probe / fixtures | 退出活动工作流，以新参考 oracle 和新协议替换；实验结果只作历史记录 |
| `examples/g1/evaluate_tracking.py`、record/preview 工具 | 新协议和分片评估，固定窗口加失败片段视频 |
| `examples/locomotion/cluster/` | 同步 4-GPU job 与独立 seed 数组分别表达 |
| `tests/python/test_supermimic_*`、`test_tracking_*` | 替换旧参考 fixtures；增加独立 oracle、终止边界、计数器、多卡和数据测试 |

优先实施 P0 和 P1：取得实际 reference 配置/权重、建立相同依赖环境、从最终 mjlab G1 模型导入并审计。
然后先完成 P3 的纯 Torch 一步更新对照，再连入 Gobot 真实物理环境。

**核对依据**

参考固定 checkout 中的：`README.md`、`docs/EXPERIMENTS.md`、`pyproject.toml` / `uv.lock`、
`src/supermimic/config.py`、`training/loop.py` / `distributed.py`、`world_model/`、`imagination/`、
`policy/nets.py`、`envs/tracking_env_cfg.py` / `actions.py` / `sim_context.py` / `state_io.py`、
`envs/mdp/commands.py` / `motion_library.py` / `rewards.py`、`data/recorder.py` / `replay_buffer.py` / `motion_convert.py`、
`eval/benchmark.py` 与评估合并脚本。Gobot 当前差异由对应源码直接核对。
本计划中的阶段、资源预留与验收容差是实施建议，所有实验成绩仍是参考仓库自报。
