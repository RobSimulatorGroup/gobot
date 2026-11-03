**Gobot：SuperMimic 迁移、命令学生、AMP 与地形技能实施计划**

日期：2026-11-02。本文是修改与验收计划；新增模块、接口、配置和阈值提案均不代表已经实现或通过训练验收。本次编写计划不启动训练、不变更授权预算。

**1. 范围、已知事实与交付顺序**

保留四个产品里程碑：M1 Gobot 动作跟踪复现 → M2 可命令走跑 → M3 地形路线通过 → M4 技能衔接与部署。每个里程碑交付独立 checkpoint、manifest、评估报告和可复现实验配置。

当前参考是 Uni-SuperMimic，commit `faf7818e8abac3ec6abb8f00e6e904d3577f800b`；它与原始 IsaacLab 实现的关系在实验元数据中明确记录。现有本地实验使用 MuJoCo/MuJoCo Warp 3.11，Gobot 主环境固定为 3.17。算法移植保留来源、许可和修改记录，运行时不导入 UniLab。

2026-10-28 的参考训练已结束：12,000 iterations，512 environments，走路 12/12、跑步 4/4、冲刺 2/2，全库 33/40，平均归一化 return 0.7318447。全库 40/40、return ≥0.82 仍是质量提升目标；它不作为走跑学生训练的必要前置条件。18 段走跑完成也不自动代表已通过脚滑、扰动恢复或命令控制验收。

当前预算账本累计 8.916599 GPU 小时。原窗口起点 6.738601 小时，追加 8 小时，有效累计上限 `min(24, 6.738601 + 8) = 14.738601` 小时，剩余约 5.822001 小时。每次启动前重新读取账本，本文数字不能代替实时预算检查。

首轮教师、学生、地形和播放统一使用参考兼容的 29DoF G1。现有包含手指的 43 关节 G1 作为独立物理配置保留；即使两者都控制 29 个身体关节，也不能假设质量、惯量和碰撞等价。迁移到带手指版本或其他机器人需要新的物理身份及验收。

**2. 先冻结实验身份与评估协议**

新建运行身份描述和独立实验配置，至少包含：源代码 revision/overlay digest、原生构建身份、机器人/控制/运动学签名、场景与地形签名、数据及 split 签名、奖励/终止/观测/动作版本、checkpoint 父链、依赖版本、设备、精度、随机种子、批量大小和评估协议版本。

严格区分三种 checkpoint 操作：

- `eval-import`：转换键名或容器后只做冻结推理；参数、归一化统计、历史长度、动作映射不改变。
- `resume`：语义配置相同，恢复算法、optimizer、normalizer、训练进度与 RNG；物理 episode 是否重新开始写入协议，不宣称逐位连续恢复。
- `fork`：明确改变物理、任务、数据或训练方式，建立新实验身份与父 checkpoint；说明哪些权重保留，哪些状态重建。

参考 checkpoint 接入前先做以下对照，不用适应训练弥补移植误差：

| 实验 | 实现与物理 | checkpoint | 用途 |
|---|---|---|---|
| R0 | 独立参考，3.11 | 冻结原始 C0 | 既有基线及必要复核 |
| R1 | 独立参考，3.17 | 同一个 C0 | 版本变化 |
| G0 | Gobot，3.17 | 同一个 C0 | 与 R1 比较移植误差 |
| G1 | Gobot，3.17，显式适应训练 | C0 的新 fork | 适应收益 |

若 R1 已严重退化，先定位版本/物理配置；若必须在参考 3.17 下适应出 C1，则 R1/G0 都用同一个 C1 重新做移植对照，不能比较不同 actor。

统一评估关闭训练随机化与观测/动作噪声，冻结归一化，固定初始帧、初始状态、碰撞配置、solver、控制器、奖励及终止规则。参考随机化配置作为训练协议另行记录，不能在忠实训练复现中静默删除。

产物建议：`run_identity.json`、`physics_contract.json`、`data_manifest.json`、`evaluation_protocol.json`、`paired_evaluation.json`。配置、报告和视频都引用这些文件的 digest。

**3. M1-A：参考兼容的 29DoF 场景与静态编译验收**

建议新增 `examples/g1/g1_29dof.jscn`、`g1_29dof_flat_scene.jscn` 和独立 robot profile。通过导入/转换工具一次性建立 Gobot 资产；训练和播放始终从 `.jscn` 编译。参考 URDF/MJCF 只用于来源记录和对照，不作为 Gobot 的第二份运行模型。

静态比对按稳定关节/链接名进行，覆盖：

- 29 个活动关节和固定链接；浮动根单独统计。常见自由根结构的 `nq=36`、`nv=35` 不等于动作维度 29；最终以编译产物核实。
- 父子拓扑、轴、joint anchor、链接静态变换、零位/初始角度、关节上下限。
- 质量、质心、完整惯量及惯量坐标系、armature、阻尼、摩擦。
- 碰撞几何/缩放、接触参数、排除对、地面摩擦。
- PD、actuator affine 参数、gear、力矩/速度/控制限幅、动作裁剪和控制更新时序。
- 50 Hz 控制、4 子步，即 0.02 s 控制周期和 0.005 s 物理步长；solver/integrator/iteration 等设置逐项记录。

不要用相同的 joint_names 或相同 kp/kd 代替完整物理一致性。先输出机器可读差异报告，所有非零差异有明确解释，再进入动态测试。

复用现有 `robot_physics_digest()`，同时补充控制与运动学签名；地形身份独立记录，允许在相同机器人上评估新地形。纯可视材质改变不应被误判为动力学改变。

通过条件：维度、次序、坐标约定、参数来源可追溯；没有未解释的拓扑、惯量、碰撞或控制差异。失败时修改资产/编译映射，不启动训练。

**4. M1-B：薄 ArticulationBatch 与场景 Torch FK**

现有 `python/gobot/sim/providers/base.py` 已有 `RobotBatchSpec/State/View` 和 `SensorBatchView`。复用这些接口及 Warp provider；CPU 使用已有 typed batch runtime 适配。不要先把两个大型 velocity 环境整体重构成新基类。

建议 `python/gobot/rl/articulation.py` 只提供以下能力：

```python
read_state()                         # 稳定借用视图
set_joint_targets(q_target)           # 按 spec.joint_names，rad/m
step_control()                       # 固定 decimation，更新/聚合传感器
reset(mask, state, targets, *, clear_controller=True)
read_contacts()                      # 区分末子步值、周期峰值、周期冲量
```

它不拥有命令采样、reward、episode timeout、非法接触规则或自动重置策略。状态只按任务实际需要补充链接速度、实测/命令力矩等字段。控制器、接触缓存、历史控制量的 reset 行为由底层契约保证；任务的 observation history、phase 和 reference frame 在任务 reset 中同步清零或初始化。

现有 RobotBatchView pose 使用 xyzw；参考实现使用 wxyz。维持公共接口，在 tracking adapter 做一次显式转换，禁止顺便改写整个 locomotion 栈。速度明确为世界/机体坐标以及链接原点/质心速度，不能只写六维数组。

推荐任务步进顺序：读 s_t → 处理动作 → 固定子步推进 → 刷新状态/传感器 → 算 reward/terminated/truncated → 把 s_t、动作和重置前 s_{t+1} 写入拥有独立存储的 transition → 掩码 reset → 返回新 episode 的 observation。replay 入库复制到预分配存储，不能保留会被下一步覆盖的借用张量。

建议 `python/gobot/rl/kinematics.py` 从 physics snapshot 生成 Torch FK。先核实 snapshot transform 与 initial_position 的语义，再构造 parent→joint 和 joint→child 的两段静态变换，避免初始角度重复计算。必须使用物理拓扑而非编辑器树遍历推断关节关系。

首版支持浮动根、树形结构、固定/转动/移动关节；不支持的球关节、闭环及其他约束明确报错。参考 FK 会跳过无关节 body，不能直接作为通用 FK。

FK 验收使用固定链接、偏移 anchor、非零初始角度和随机姿态/速度：位置 ≤1e-5 m，旋转测地误差 ≤1e-5 rad，线/角速度 ≤1e-4 对应 SI 单位；双精度有限差分验证可微路径，记录绝对与相对梯度误差。动态 CPU/Warp 接触轨迹不套用 FK 的严格误差阈值，应依据短程对照和接触误差单独判断。

复用/扩展 `test_robot_batch_view.py`、snapshot/provider/batch 测试，新增 `test_scene_torch_fk.py`。测试 mask 隔离、控制器清零、固定 storage、失效 view、重置前状态、env=1 与 batch 内同一环境一致性。只有确需补充 native 字段时改 C++，并完成 CMake 构建和相关原生测试。

**5. M1-C：先实现 tracking 推理，再移植训练算法**

建议新增以下包；文件名是实现建议，不要求预先创建空框架：

| 模块 | 职责 |
|---|---|
| `gobot.rl.motion_tracking.config` | MotionTrackingCfg、机器人/数据/参考时间配置 |
| `gobot.rl.motion_tracking.env` | make_motion_tracking_env、批量 step、reset_to_frames |
| `gobot.rl.motion_tracking.mdp` | tracking obs/reward/termination/critic features |
| `gobot.rl.motion_tracking.motion` | clip/frame、未来目标及兼容参考的采样 |
| `gobot.rl.motion_tracking.manifest` | 教师输入/动作/数据协议 |
| `gobot.rl.supermimic` | contracts、networks、replay、geometry、learner、checkpoint |

忠实移植阶段保留参考的 10 帧历史、10 帧未来目标、32 步 BPTT、8 步世界模型拟合窗口和现有网络结构；512 环境/批量是本地配置，不称为上游默认 4096 环境规模复现。未来目标跨度约 0.20 s，32 步 BPTT 跨度约 0.64 s，两者不要混为一个 horizon。

教师单帧观测是 99D，包含参考相对朝向，不能将它视为纯本体感知。reset 历史填充、previous action、clip 尾部目标钳制、episode/clip 边界、reward 的 dt 缩放严格复现；tracking 不继承 velocity 的骨盆高度和禁止身体触地规则。

实现顺序：

1. 同一状态输入下对照 observation、future goal、动作转换、reward 和 termination。
2. 冻结 actor/normalizer，在 CPU/Warp 对照 reset、单步、短程状态和传感器时序。
3. 与参考 3.17 使用同一 checkpoint 跑完 40 段，逐段记录完整完成率、失败时间、累计回报和轨迹误差。
4. 固定 replay minibatch 和初始权重，对照网络 forward、loss、梯度和一次 optimizer update；先用 FP32/关闭编译诊断，再验证 bf16/编译模式。
5. 验证 replay 不跨 episode/clip，mask reset 不污染其他序列；之后才进行少量真实训练与 checkpoint/resume smoke。

M1 忠实度门槛：相对参考 3.17 平均归一化 return 下降 ≤0.02，不新增失败片段；同时报告逐片段差值、最坏组差值和失败提前时间。长轨迹对微小数值差异敏感，对临界失败按预先固定的重复评估协议复核，不能看到结果后调整门槛。任何提前终止片段按完整目标时长归一化，不能仅对存活部分归一化。

产物：独立 Python train/evaluate 入口、冻结推理 checkpoint、教师 manifest、40 段配对报告、固定走/跑/冲刺视频。M1 通过不等于动作质量达标。

**6. M1-D：物理适应 fork 与走跑教师准入**

仅在忠实移植已验收或版本适应确有需要时进行。保留 actor 权重及其 obs/goal normalizer；重新建立 optimizer、critic、replay，世界模型权重是否复用写入 fork 配置。actor 归一化必须一起冻结，参考 collect 的持续统计更新需要显式开关。

先用固定 actor、参考状态 reset 和适度探索收集目标物理数据。记录样本覆盖、终止比例、动作/力矩饱和，避免训练集只有即将跌倒或单一站立状态。

先拟合世界模型与 critic；新增不更新 actor 的 critic warmup 路径，验证有限值、目标尺度及 bootstrap 误差。世界模型在独立 rollout 上报告 1/8/32 步 root、关节、姿态误差，分接触/非接触、走/跑/转向场景统计。误差门槛由冻结参考在目标物理上的基线确定，不能事后以训练 loss 下降代替。

解冻前固定真实模拟 probe bank，包含完整片段和可恢复扰动；小步恢复 actor 更新，每次用真实物理结果决定接受或退回。若想象回报上升而真实成功率下降，回到最近已验收 checkpoint，重新收集数据/修正模型；不能无限延长训练掩盖失配。

覆盖实验的走跑教师准入使用 18 段及其运动活跃区间，报告完整跟踪、跟踪误差、实测足底接触、脚滑和恢复能力。严格留出实验的开发准入仅使用训练/验证 subject，subject5 留到冻结模型后最终测试，不能把全 18 段反复当开发门槛。阈值在使用验证集分析后冻结；未采集脚滑前不能宣称自然步态已验收。舞蹈和全库改进独立报告，不阻塞已通过的走跑分支。

**7. M2-A：统一运动数据与命令参考调度器**

忠实移植时先保留原预处理结果；共享 motion store 在通过数值回归后接替 tracking/AMP 的独立数据读取。数据记录源文件哈希、机器人映射、30→50 Hz 重采样、四元数连续化、速度计算、contact/phase 提取方法及质量标记。推断接触是估计值，不能充当真实接触力标签。

维护两条隔离实验线：`coverage40` 用于覆盖和迁移；`subject_split_v1` 以 subject1–3 训练、subject4 验证、subject5 最终留出。split 在生成窗口、统计量和采样索引前生效。正式泛化教师不能由看过 subject5 的权重或统计初始化；AMP、BC、DAgger、世界模型也遵守相同数据隔离。记录教师数据血缘，避免通过教师间接泄漏。

先输出覆盖表：gait × body-frame vx × vy × yaw-rate，以及站立、加减速、左右转向和相位覆盖；样本数量和连续时长均统计。没有数据的命令标为 unsupported，不通过旋转直行轨迹假装获得倒退/横移示范。

建议新建 `reference_scheduler.py`，输入命令、步态和时间，维护 clip/phase/heading 及锚点，输出教师未来目标和学生可见的 gait、sin/cos phase、heading error。首版选择每个动作族的少量标准片段，限定受支持速度，避免随机切换多个不可辨识的动作风格。

只在明确的切换点进行 SE(2) 参考对齐；每步重定位参考以掩盖跟踪误差不可接受。clip 尾部、loop seam、相位推进、站立相位、命令突变的处理都要有确定规则。时间缩放会改变动力学，未经验证不作为任意调速工具。

学生输入暂定：10 帧本体状态历史 + 速度命令 + heading error + gait + phase sin/cos + 地形扫描/有效性掩码。actor 不读取完整未来动作；critic 可以增加明确版本化的特权信息。G1 需要独立的双足 phase 适配，现有 `NaturalGaitEnv` 是四足实现，不能直接复用。

在相近学生输入附近统计教师 q_target 标签方差。若差异来自隐藏片段/风格，先减少标准片段或增加部署时也可生成的意图输入；不能让模型靠记忆不可观测信息。这项检查是诊断，最终仍以闭环表现验收。

**8. M2-B：BC、DAgger 与 PPO 接入**

第一阶段 BC：固定教师、调度器、观测与动作协议，采集真实物理 rollout。按 clip/episode 划分训练和验证窗口，避免重叠帧泄漏。监督实际送入控制器的命名关节目标角度，记录原始动作、裁剪后动作与目标角度。

```text
q_teacher = q0_teacher + runtime_offset_teacher + scale_teacher * clipped_action_teacher
a_student = (q_teacher - q0_student - runtime_offset_student) / scale_student
```

公式只是位置偏移控制的示意，必须遵守实际控制链的限幅/滤波顺序。教师目标超出学生可表达范围时报告并处理，不能静默裁剪后仍声称等价。初版固定双方 kp/kd、限矩、控制周期；若改变这些参数，相同角度目标也不代表相同控制行为。

BC 按关节记录 rad 误差、尾部误差、动作和力矩饱和，并运行闭环走/跑测试。地形输入在平地阶段保持真实平地观测，地形感知能力留待后续课程；不能把输入中有 height scan 当作已学会使用它。

第二阶段 DAgger：学生控制真实环境，教师从该环境的实际状态查询。独立维护教师历史，每一帧使用当时的参考朝向；previous_action 根据学生真正执行的目标角度转换为教师协议。禁止复制原教师 rollout 历史，禁止用当前参考朝向重算全部过去历史。

先在稳定走跑及轻微偏离上查询，逐轮增加学生访问状态。专家标签有效性由命令覆盖、参考连续性及教师恢复能力共同决定；无效状态进入失败统计和恢复/reset，不能当成可靠监督。固定轮数/时间上限，保留 BC 数据与新访问数据的混合，防止仅覆盖失败边缘。

第三阶段 PPO：使用 BC/DAgger 学生作为 actor 均值网络；新建 critic/optimizer 和保守探索分布，先预热 critic。初始标准差建议从现有支持范围低端 0.1–0.2 做筛选，不直接套用冷启动 0.5；这是待验证配置。

修正 `examples/locomotion/train.py` 的 initialize 校准路径：保留 actor normalizer，critic 单独校准。在固定真实 observation probe bank 上比较导入前后 q_target，确定性 FP32 转换建议最大差 ≤1e-5 rad；任何统计调整另行通过 GuardedPPO 的 KL/真实 rollout 检查。仅检查动作裁剪率 <1% 不足以证明策略未变。

复用已有 `GuardedPPO`、`TransitionPPO`、命令修复和终止前 bootstrap。按能力逐步扩展站立→前进走→跑→转向/停走→倒退/横移；无教师覆盖的命令由 PPO 训练，不启用虚假的 BC/AMP 标签。能力提升按独立验证推进，不按固定 update 数自动升级。

补训练健康处理：记录每种命令的 episode 长度、跌倒率、目标/力矩饱和、观测统计漂移和能力得分。非有限值立即停止并保留诊断；连续验证退化触发恢复最近已验收 checkpoint、降低探索/更新幅度并暂停课程升级。重试次数和退化阈值写入配置，达到次数后结束该试验，不无限自动恢复。每次退回的 optimizer、normalizer、AMP 状态与训练身份必须一致，恢复耗时仍计入实验成本。

M2 产物：独立学生 manifest、调度器配置、BC/DAgger 数据身份、PPO checkpoint、逐命令评估与视频；学生推理不依赖教师或完整未来动作库。

**9. M2-C：AMP 只在有效参考范围生效**

现有 AMP 已有 walk/run 两个 head 和 vx/vy/yaw 匹配采样。优先修正适用范围，不立即扩大 discriminator 结构或引入 terrain-conditioned 世界模型。

把硬编码前向 active 条件、`.02` reward 缩放和 `terrain_level/9` 权重改为显式配置：真实 `step_dt`、参考 support mask、地形类别/难度、训练 warmup。报告每类命令的有效参考比例；unsupported 样本的风格奖励为零，并从对应判别器训练配对中排除。

奖励端支持判定和采样端精确匹配使用同一个语义：桶内存在样本不代表当前命令一定在容差内。对速度桶边缘、正反方向、低样本数和跨 gait 条件加入回归，防止奖励认为支持而 discriminator 无法取得匹配样本。

保留 matched sampling + gait heads 作为首版；这不等于 discriminator 直接输入连续命令。如果以后引入 `D(s_t, s_t+1, command, gait)`，单独做消融，且真实/策略样本必须带同一命令条件，避免靠命令分布辨别真伪。

接触/运动对来自真实重置前 transition，不能把 reset 跳变当作风格样本。奖励、特征、normalizer、replay 与采样器版本进入 checkpoint。

地形相对 AMP 权重建议初始采用：平地 1.0，缓坡/粗糙地面 0.5，上下楼梯/低障碍 0.1–0.25；没有参考支持则为零。数值是消融起点，不是已验证最优参数。新地形先跑无 AMP 短对照，确认通行能力不会被风格奖励压制。

**10. M3：地形路线、课程与能力门槛**

课程按独立能力推进，保留已学地形回放采样，防止新课程遗忘平地命令能力。默认楼梯/障碍区域步行；跑步限制在平地和 ≤5° 缓坡。

| 课程 | 起始配置 | 必须明确的场景参数 |
|---|---|---|
| 平地命令 | 站立、走跑、停走、转向 | 命令保持时间、加减速过程、评估区域 |
| 缓坡 | 正反向 2°、5° | 坡长、坡顶/坡底连接、横坡是否另测 |
| 粗糙地面 | 低幅到 ±3 cm | 空间频率、相关长度、实际可通行性 |
| 上/下楼梯 | 5、10、15 cm | 踏面深度、级数、宽度、平台和进入距离 |
| 跨低障碍 | 5、10 cm | 宽度、间距、边缘、允许接触部位 |
| 静态绕行 | 已知障碍和航点 | 走廊、转弯半径、碰撞体与到达顺序 |

楼梯生成建议从 0.30 m 踏面、5 级开始，随后在训练集中变化踏面/级数；低障碍从单个窄障碍开始，随后变化宽度和间距。以上为初始设计提案，实际资产需验证碰撞和传感器可见性，再冻结训练/验证/测试配置。

路线按必须跨越的顺序 gate 和终点判定。站立存活、退出侧边界、越过未指定路线、自动 reset 后抵达均不是通过。每种高度、上/下行、名义/扰动条件独立报告。

reward 优先使用沿路径的增量进度、命令跟踪、平衡和动作/能量/接触成本；进度由路线投影定义，避免原地绕圈或反复跨一个边界刷分。滑移从实际接触足的切向速度统计，接触阈值和窗口固定。

height scan 的坐标系、采样位置、范围、时序、无命中值/有效性掩码进入 observation contract；上下楼梯边缘和小障碍必须有传感器覆盖测试。航点跟踪器在策略上层产生速度命令，低层 PPO 不承担隐含全局规划职责。

新增结构化能力结果：`capabilities[name] = passed / failed / untested / unsupported`，并记录协议、checkpoint、条件、样本数与失败原因。训练读取本阶段需要的能力集合得到 `stage_passed`；不得把 missing 字段当成通过，也不得把未测试的地形报告为 failed 或 passed。保留完整产品验收的 `passed`，两者语义不同。

现有 assessment 的瞬态/稳态误差拆开：命令变化后默认前 2 s 作为响应窗口，单独报告响应时间、超调与不稳定；稳态只统计后续窗口。停走在 2 s 后速度 <0.1 m/s，并持续检查后续窗口。该响应定义需版本化，不能借删除困难时间段隐藏跌倒或越界。

命令稳态采用平面速度向量误差 ≤max(0.15 m/s, 20%命令速度)、yaw-rate 绝对误差 ≤0.2 rad/s；直线路径偏差 ≤0.5 m。地形各类完成率 ≥90%；沿用现有最终总体 ≥95% 门槛时，必须在协议中明确其覆盖范围和权重。

验证使用固定开发 seeds；最终使用独立地形/命令/seeds。现有 1001/2002/3003 可保留，每地形/难度/seed 至少 10 次形成初步评估；同时报告样本数和置信区间。若要声明统计上可靠的 ≥90%，需要扩大样本并预先规定区间下界标准，不能把 9/10 当作可靠证明。

**11. M4：SkillController、切换与导出播放**

建议 `gobot.rl.skills` 提供 `set_velocity()`、`play_motion()`、`follow_waypoints()`、`stop()`，以及 `step(state, dt) -> JointTargetBatch`。高层命令与物理步进分离，始终输出按稳定关节名排列的 PD 目标；不依赖编辑器、ImGui 或渲染后端。

内部包含命令/航点生成、策略 adapter、manifest 校验、历史/phase 管理和过渡状态机。策略可拥有不同输入协议，统一的是外层调用与控制输出。

首版状态机：ACTIVE → DECELERATE → STABLE_STAND → ALIGN_ENTRY → WARM_HISTORY → ACTIVE；超时或入口条件不满足进入 HOLD/FAILED 并给出原因。`stop()` 表示受控停止，不等于急停；急停/异常暂停与技能停止保持明确语义。

每个技能注册入口/出口姿态、可接受 root 速度和姿态、接触条件、历史初始化方式、允许来源技能、最大过渡时间。舞蹈先只接入已验收入口；从站立到任意舞蹈帧、倒地动作、跳跃动作不自动开放。先测可用短过渡，失败再训练专项 transition，不宣称关节插值能保证稳定。

切换测试覆盖 walk↔stand、run→stand、stand→walk/run、stand↔指定 dance、waypoint→stop，以及转向/扰动时停止。报告每对切换的完成率、时间、最大关节/力矩跳变、脚滑和终止原因。

`PolicyManifest` 增加技能/协议、历史/phase、坐标/时间/normalizer、物理/控制签名、数据与父 checkpoint 身份及准入报告引用。兼容既有 format v1 的读取；新格式显式升级，旧策略不能被误认作新 tracking/student 协议。

导出 actor 及推理需要的确定性预处理，教师仍需未来目标，学生仍需历史/phase/scan。单步 Torch/ONNX 对照不足，增加真实 observation 序列、reset、命令变化、切换后的多步 q_target 对照。最终播放使用同一 adapter，在 Gobot runtime clone 上运行，不修改编辑场景。

**12. 提速实验、性能测量与集群交付**

最终比较采用四组；相同学生架构、机器人、控制、命令/地形课程、随机化、评估频率、调参预算与终点指标：

| 组 | 初始化 | AMP |
|---|---|---|
| A | 冷启动 PPO | 无 |
| B | 冷启动 PPO | 有 |
| C | BC/DAgger 学生 → PPO | 无 |
| D | BC/DAgger 学生 → PPO | 有 |

冷启动/蒸馏的 reset 分布应相同；若蒸馏额外使用 reference-state initialization，明确把它计入方法组成或增加消融，不能把收益全部归因于权重迁移。BC/DAgger 与教师的 GPU 成本计入完整方法。固定同一教师只能测学生训练随机性；如需证明完整方法鲁棒性，另评估教师种子变化。

本地 42/43 用于筛选，最终集群建议至少 3–5 个学生种子并报告离散度。统一定义 time-to-threshold：连续两次独立验证通过该任务门槛的第二次时间；最终留出集只做冻结模型后的确认，不用于选 checkpoint。未达标实验保留为预算内未达标，不从统计中删除。

同时报告端到端 GPU 小时/实际时间、教师复用后的新增任务成本，以及明确任务数 N 的摊销成本 `教师成本/N + 学生/适应成本`。≥50% 提速只在相同质量门槛和明确成本口径下判断。

先测 512 环境，再测 1024/2048 的显存和吞吐；物理 batch、world-model batch、policy batch 分别扫描，保持其他条件可比。profile 分解仿真、传感器、状态转换/复制、采样、世界模型、BPTT、PPO/AMP 更新、评估和编译启动时间。CUDA 异步计时使用正确的 event/同步边界，不能以 Python 调用时间代表 kernel 时间。

microbenchmark 报告稳态时间和启动/编译时间；端到端时间把两者都计入。32 步 BPTT 不为追求吞吐而静默缩短。未验证接触/reset/终止前状态语义的优化不能进入正式比较。地形世界模型/BPTT 放在后续独立阶段。

扩展现有 `package.py`：用明确源码清单包含任务依赖的 Python、provider、native C++/bindings、测试和构建配置、未提交但必要的源文件、场景/运动数据、完整配置、版本与校验和。现有固定 overlay 不覆盖新增 native/provider 代码，不能直接宣称包可复现。只包含明确需要的项目文件，目标机器重新构建 native 库。

在干净 checkout + bundle 上完成 build、契约 smoke、冻结 checkpoint 推理后才交给集群。每 GPU 独立进程先于分布式同步训练；独立 job 账本汇总并受总额度预留控制，不能每个进程各自获得一份完整预算。

**13. 本地剩余额度的建议分配**

以下是上限分配，不是达成全部里程碑的时间保证。先完成不消耗 GPU 的代码、静态模型和 CPU 测试；上一关失败时停止后续 GPU 阶段。未花额度保留在同一个冻结窗口内。

| 阶段 | GPU 小时上限建议 | 目的 |
|---|---:|---|
| 参考 3.17 对照 | 0.60 | 升级兼容、冻结推理和失败定位 |
| Gobot tracking 对照 | 1.00 | 短程/全库冻结评估 |
| 算法 smoke + 单点性能 | 0.40 | backward/checkpoint/resume、512 环境 |
| 条件性世界模型/critic 适应 | 0.80 | 仅在确有需要且前置通过时 |
| 单个学生试验 | 1.80 | BC/少量 DAgger/PPO，优先 seed42 |
| 独立评估与视频 | 0.70 | 同一 checkpoint 的质量验证 |
| 故障/保存余量 | 0.52 | 编译超时、受控停止及断点 |
| 合计 | 5.82 | 不增加原授权 |

不在此额度内预先承诺四组×两种子的完整比较、地形全课程或多机器人训练。接口试验失败时先保存证据并修代码，不能自动加时或另建授权窗口。

统一预算入口读取现有 `budget_window.json`，把有效累计 cap 传给 train/evaluate/record/benchmark；现有默认 `--budget-hours 24` 不得覆盖冻结窗口。受控停止预留以实测保存耗时和最近迭代耗时计算，保存失败仍保留上一份原子 checkpoint。

保留每 15 分钟固定走/跑/冲刺视频；舞蹈在对应教师可评估后增加，地形在对应路线具备后增加。视频标注 checkpoint/hash、任务、命令、起始帧、环境身份、失败原因和时间。预览与完整路线/片段验收分开，失败画面保留终止前状态。明确是否按墙钟每 15 分钟调度；繁忙时跳过过期请求，避免录制任务堆积。同卡重叠使用按同一设备占用窗口计账，同时报告它对训练吞吐的影响；独立录制单独计账。

**14. 建议提交顺序与每个提交的完成定义**

| 顺序 | 修改内容 | 完成定义 |
|---|---|---|
| P0 | 运行身份、冻结协议、有效预算上限、能力结果 schema | 配置/账本/报告回归通过，无训练启动 |
| P1 | 29DoF G1 场景与 profile、静态物理对照 | 无未解释的物理/控制差异 |
| P2 | ArticulationBatch 薄适配、snapshot FK | FK、mask、buffer、控制与时序测试通过 |
| P3 | tracking 环境及冻结 actor 推理 | 单帧/短程/40 段移植对照通过 |
| P4 | SuperMimic 算法与 checkpoint 三种模式 | 固定 batch 等价、真实小训练和恢复通过 |
| P5 | motion store、split、覆盖表、reference scheduler | 数据血缘、目标连续性、历史/phase 测试通过 |
| P6 | BC/DAgger、PPO actor 导入与 normalizer 修复 | q_target 等价、闭环学生报告和命令能力门槛 |
| P7 | AMP 支持域、地形课程、能力推进 | 逐命令/逐地形路线报告，不误升阶段 |
| P8 | SkillController、切换、ONNX/Gobot 播放 | 多步导出等价及已声明切换对验收 |
| P9 | 扩展打包、集群消融与性能报告 | 干净环境可重现；质量/成本同口径 |

P0/P1/P2 可以在冻结数据协议后拆小提交；参考 3.17 独立环境验证可与这些 CPU 工作并行。P3 不能越过模型/FK/时序验收；P6 不能越过教师准入和调度器定义；P7/P8 只依赖相应已验收技能，不等待无关的全库失败动作全部修好。

近期可执行目标是完成 P0–P4，并在剩余额度允许时做 P5–P6 的最小前进走跑学生。M3/M4 保留完整设计和接口预留，按照验证结果推进。每次阶段失败留下失败输入、checkpoint、身份文件、最小复现与诊断报告，而不是只留下“训练未收敛”。
