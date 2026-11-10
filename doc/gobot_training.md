# Gobot 统一训练入口

`gobot-train` 与 `python -m gobot.rl.train` 使用同一入口，支持下列任务；机器人和地形仍由
Gobot 场景描述，训练不依赖编辑器或 NodeScript。配置分开描述任务、算法和仿真后端，
既可以使用任务名称，也可以直接提供 Python 工厂。

| 任务名称 | 工作流 |
| --- | --- |
| `go1-velocity` | Go1 速度控制 PPO |
| `go1-gait` | Go1 步态相位奖励 PPO |
| `g1-velocity` | G1 速度控制 PPO |
| `g1-amp` | G1 速度控制与 AMP 动作先验 |
| `g1-tracking` | G1 SuperMimic 参考动作跟踪 |

```bash
python -m gobot.rl.train --list
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/go1_cpu
python -m gobot.rl.train g1-amp \
  --backend mujoco-warp --device cuda:0 --num-envs 1024 \
  --dataset /data/g1/locomotion.npz --iterations 1000 --output logs/g1_amp
```

共同参数有 `--backend`、`--device`、`--num-envs`、`--seed`、`--output`、`--resume`。
上述任务只支持 `mujoco-cpu` 和 `mujoco-warp`；引擎中的 Newton、IPC 等后端尚未接入这些训练工作流，
不接受请求后静默回退。单独切换后端会选择相应默认 device；CPU 物理也可显式配合 CUDA learner。
切换后端用于新实验，不意味着可跨后端恢复优化器及训练状态。

PPO/AMP 的 `--iterations` 计数为 **PPO 更新次数**；SuperMimic 的 `--stop-iteration` 是
**每 rank 的绝对控制步停止点**，不会缩短学习率调度。速度 PPO、Go1 步态 PPO 和
SuperMimic 支持 `torchrun` 同步多卡；AMP 的判别器尚未接入多卡同步，仍使用单进程训练。

标准速度任务用 `--terrain flat|rough` 选择地形，可直接开始训练，无需通过前置阶段。
`--initialize` 只载入 actor 与观测归一化参数；`--resume` 恢复算法、优化器、课程和随机状态。
AMP、Go1 步态扩展可另外使用 `--terrain mixed`。旧任务名称和实验配置已删除。

## 标准速度训练

`go1-velocity` 和 `g1-velocity` 参照仓库内 mjlab
`dcdf6ce519abfd2ea45e9503ae6fb650b75bb869` 的平地/粗糙地形配置，使用 RSL-RL 5.5.2 的 PPO 更新。
公共训练器接收 `RunConfig` 与 `VelocityTaskCfg`，型号选择只在 `examples/locomotion/presets.py`。
`TransitionPPO` 对 timeout 使用重置前最终状态的价值，不使用上一步状态或重置状态的价值。
默认设置为学习率 `1e-3`、自适应学习率、熵系数 `0.01`、初始动作标准差 `1.0`，
每次更新采样 24 步，优化 5 轮、每轮 4 个 minibatch。没有更新回滚、归一化预热或额外跌倒扣分。

奖励权重、速度误差公式、观测噪声、命令采样和课程按该参考设置。actor 的关节位置包含编码器偏置，
critic 使用真实位置。最后物理子步后刷新状态，再计算奖励、终止与最终观测，随后处理推力和重置；
刚重置的环境保留完整命令持续时间。平地策略不接收地形高度扫描。

速度任务保留 mjlab 的 `ImplicitFast` 积分，物理步长 0.005 秒、控制周期 0.02 秒。
异常处理同样对齐标准 mjlab：非有限的加权、按时间缩放后的奖励项置零；
观测不清零 NaN/Inf，由 RSL-RL 的输出检查报错。没有额外速度阈值中止或数值异常自动重置。
默认八卡配置输出到 `logs/g1_velocity_mjlab_seed42`，详细规则见 `doc/velocity_stability.md`。

`go1-gait` 显式增加步态相位；`g1-amp` 保留动作先验；`g1-tracking` 使用 SuperMimic。
它们的附加奖励和观测不进入标准速度任务。

独立参考数据保存在 `tests/fixtures/mjlab_velocity.json`，由
`tests/support/export_mjlab_velocity.py` 在 mjlab 环境中生成。它记录来源文件哈希、四份有效配置和
16 个奖励项以及机器人关节增益、力矩限制和动作缩放。`tests/python/test_mjlab_training.py`
比较参考参数、NumPy/Torch 奖励和相同非超时 transition 下的 PPO 更新。
Gobot 采用更新后的状态、独立环境 RNG 和正确的终态 bootstrap，不宣称完整训练时序与参考逐位一致。
这些检查不替代多 seed 的训练效果比较。

物理仍从 Gobot 场景编译。G1 行走、AMP 与回放使用同一个 `g1.jscn`：29 个身体关节、固定手掌，
删除了 14 个手指驱动。质量、惯量、关节、碰撞和驱动参数通过 `tests/fixtures/mjlab_g1.xml`
与固定版本的 mjlab 模型逐项核对。网格随仓库提供，不再依赖本机未跟踪的手指 STL 文件。
旧模型 checkpoint 的物理标识不同，不能直接 `--resume`；`--initialize` 也仍要求模型一致。
评测可显式加 `--allow-model-change` 测迁移，报告同时记录训练模型和实际模型标识。
Gobot 的原生 CPU 批处理要求 MuJoCo 3.17，配对运行应使用同一套依赖。

评测的 `--terrain` 只选择评测地形，保留 checkpoint 的观测、控制和求解器设置。例如把粗糙地形策略
放到平地评测，不会删掉它的高度扫描输入：

```bash
python -m examples.locomotion.assessment \
  --checkpoint logs/g1_ppo_8gpu/model_final.pt \
  --terrain flat --speeds -.3 .8 1.5 2.5 --steps 1000 --num-envs 32 \
  --allow-model-change --output logs/g1_transfer/flat.json
python -m examples.locomotion.assessment \
  --checkpoint logs/g1_ppo_8gpu/model_final.pt \
  --terrain rough --speeds .8 --steps 1000 --num-envs 70 \
  --allow-model-change --output logs/g1_transfer/rough.json
```

`velocity_error_after_warmup` 是排除起步 2 秒后的水平速度向量误差（m/s），只统计首次结束前的样本；
`tracking_episodes` 同时给出有有效样本的次数，不能忽略提前跌倒的情况。
`fall_rate` 统计物理跌倒终止，`off_route_rate` 单列偏离路线。
`terrain_pass_rate` 是到达对侧终点且未失败的比例，未及时到达也不算通过；
它与附加速度/轨迹阈值的 `admission` 分开。当前 8 米地形块的起终点距中心 3.25 米，
位于边缘平台内部，避免原先 3 米处的楼梯接缝。各难度、地形类型和每次运行的明细都保留在报告里。

## 修改奖励和训练配置

`--config` 使用所选工作流的 JSON 配置。速度任务包含 `experiment`、`env`、`agent`、
`backend` 等字段；SuperMimic 使用自己的算法配置树。
`--set PATH=JSON` 在配置和普通命令行参数之后应用；`--dry-run` 校验并打印配置，不创建仿真或输出目录。

```bash
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval \
  --set env.rewards.foot_slip=-0.1 \
  --set agent.algorithm.learning_rate=0.0002 \
  --output logs/go1_custom --dry-run
```

移除 `--dry-run` 后启动训练。自定义 `env`（包括奖励）要求 `--no-eval`，因为现有独立验收器
按内置任务配方重建环境；自定义任务应使用匹配的评测程序。完整解析后的任务与算法配置写入运行目录。
速度任务的 `reward_terms` 是加权奖励率，最终 reward 按 `physics_dt × decimation` 积分一次。
标准 PPO 按 RSL-RL 的方式处理超时，真实终止与超时分开传入。AMP 使用重置前的观测进行超时计算，
动作先验及终止惩罚由 `agent.algorithm.motion_prior_cfg` 配置；有限时域超时不算跌倒。
AMP 当前动作先验使用 50 Hz 相邻帧，入口会检查 `physics_dt × decimation = 0.02`，
避免修改控制频率后继续使用不匹配的先验。

Python 可直接修改配置，无需注册任务：

```python
from gobot.rl.training import launch
from gobot.rl.training.locomotion import training_spec

spec = training_spec(robot="go1", recipe="baseline")
cfg = spec.config_factory()
cfg.backend = "mujoco-cpu"
cfg.experiment.num_envs = 4
cfg.experiment.iterations = 10
cfg.env = {"rewards": {"foot_slip": -0.1}}
cfg.agent = {"algorithm": {"learning_rate": 0.0002}}
cfg.no_eval = True
launch(spec, cfg)
```

也可以让可导入模块 `my_training.py` 中的无参 `make_training()` 返回一个 `TrainingSpec`，
再运行 `python -m gobot.rl.train my_training:make_training --backend mujoco-cpu ...`。
`TrainingSpec` 描述配置工厂、运行函数和支持的后端；任务注册只是可选的名称索引。

训练前准备对应机器人场景和 AMP/跟踪数据。自动评测会在项目内生成评测地形，
因此应使用可写项目或源码 checkout；安装包中的只读示例目录不适合作为完整评测工作目录。

## 多 GPU 共同训练 PPO

每张卡运行一个进程，各自产生 `--num-envs` 个环境的采样，RSL-RL 在每个 minibatch
平均梯度，并同步观测归一化统计、KL 与自适应学习率。启动时广播相同模型，环境和采样使用
`seed + rank`。不自动放大学习率，也不改变 PPO 的 rollout、epoch 或 minibatch 配置。

8 卡 Go1 粗糙地形训练：

```bash
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train go1-velocity \
  --terrain rough --backend mujoco-warp --num-envs 1024 \
  --iterations 10000 --hours 12 --budget-hours 96 --seed 42 --no-eval \
  --budget-file logs/go1_8gpu/budget.json --output logs/go1_8gpu
```

`--num-envs 1024` 是**每张卡**的环境数，8 卡共 8192 个环境；默认 rollout 为 24，
每次更新采样 196608 个控制步。`--iterations` 仍是共同策略的更新次数，不自动除以卡数。
`LOCAL_RANK` 自动选择当前可见设备；例如 `CUDA_VISIBLE_DEVICES=2,3` 配合
`--nproc-per-node=2` 使用物理卡 2 和 3，无需逐卡设置 `--device`。
同一入口支持 `g1-velocity`、`go1-gait`，也可在 `mujoco-cpu` 后端用多进程验证。

先将环境数设为 128、更新次数设为 10，用新的输出目录做冒烟测试。没有添加 `--no-eval`
时，只有主进程启动评测，其余进程等待结果并共同检查停止信号。日志中的吞吐量是所有卡的
合计值，episode 指标按所有卡实际完成的 episode 数加权统计；模型、配置、日志与预算
文件都只由主进程写入。

恢复同一训练：

```bash
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train go1-velocity \
  --terrain rough --backend mujoco-warp --num-envs 1024 \
  --iterations 10000 --hours 12 --budget-hours 192 --seed 42 --no-eval \
  --resume logs/go1_8gpu/latest.pt \
  --budget-file logs/go1_8gpu/budget.json --output logs/go1_8gpu
```

`--resume` 中的更新次数表示继续运行的次数；保持相同卡数、每卡环境数、任务和算法配置。
检查点保存共同策略、critic、优化器，以及每个进程的课程和随机状态；恢复时重新开始物理
episode，不保存接触求解器的瞬时状态。更换卡数、每卡环境数或使用旧格式检查点时，
用 `--initialize checkpoint.pt` 和新的输出目录开始训练。初始化保留 actor 与观测归一化，
critic、优化器和 action std 重新初始化。

`--hours` 是本次运行的墙钟小时上限；`--budget-hours` 是预算文件累计的 GPU 小时上限。
例如 8 卡运行 12 小时需要 96 GPU 小时，恢复训练会继续累加同一预算文件的用量。
单卡与多卡使用相同的 PPO 更新规则；更多卡意味着每次更新采样更多，实际吞吐和收敛需要实测。

Slurm 模板 `examples/locomotion/cluster/train_velocity.sbatch` 默认申请 8 卡共同训练一个模型。
设置 `GOBOT_REPO`、`GOBOT_OUTPUT` 后提交。不同随机种子的对照实验分别提交，并使用独立的
输出目录和预算文件。与 mjlab 比较时记录速度误差、跌倒率、地形通过率、训练时间和显存。

### EPA 碰撞缓冲警告

`Warning: EPA horizon = 24 isn't large enough` 表示凸碰撞穿透计算的临时边缓冲不足，
不是 PPO rollout 长度，也不是显存耗尽。仅关闭 `warn_overflow` 会隐藏输出，不能恢复
失败的碰撞计算；`nconmax` 和 `njmax` 也不控制这块缓冲。

Gobot 的 Warp 后端在预热和 CUDA graph 捕获前将该缓冲至少扩为 64，保留 MuJoCo Warp 的溢出警告。
这是对当前固定依赖 MuJoCo Warp 3.17 的内部缓冲分配调整，不修改安装包文件。
更新 Python 代码后重启训练进程即可生效，不需要重编译 Gobot。仅调整缓冲容量且训练签名
不变时，现有 PPO 检查点可以用原来的参数和 `--resume` 继续。采用 mjlab 数值处理的训练
签名与旧版不同，旧运行须使用新的输出目录重新训练。若扩容后仍报 EPA 溢出，应记录触发
姿态和碰撞网格进一步排查。

## SuperMimic 训练与复现

G1 tracking 使用固定版本 SuperMimic-Reproduced，启动时显式选择对应任务。
完整操作入口：[数据准备/评测/视频](../examples/g1/reproduction/README.md)。
mjlab 仅用于独立参考对照，以下配方与恢复协议保持不变。

## 固定目标

- SuperMimic-Reproduced `ba6105349069bf040f1fdad60873d13138d505a2`。
- mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376`。
- G1 29 DoF；50 Hz 控制，200 Hz 物理；93D 观测、term-major 10 帧历史、680D 目标。
- 显式积分世界模型、学习终止头、8 步 replay window、32 步 BPTT、GAE critic。
- 正式 e8 配方每卡 4096 env、WM batch 2048 × 10、policy batch **2048**、UTD 0.4。
- 基础学习率 0.0002；4 卡时乘 sqrt(4) 得到 0.0004。配置文件始终存基础值。
- 320000 控制步调度、250000 衰减结束、100000 停止分开表达。

Gobot 的原生批处理依赖 MuJoCo 3.17。原始发布栈 3.16 复核和同 3.17 栈配对实验分开记录，
不能把 3.17 上的诊断当作原始发布栈复现。

## 训练中的数值异常

CPU/Warp 速度环境采用固定 mjlab 标准任务的异常处理：保留有限的大速度和位置，
不增加物理阈值终止；加权、按时间缩放后的非有限奖励项置零，其他项保留。
观测不进行额外清洗。RSL-RL 检查返回的观测、奖励和 done，发现 NaN/Inf 后报错。
日志中的 `action_abs_max` 是每轮所有 GPU 采样动作的最大绝对值，可用于追查尖峰。

采样阶段的异常会在 PPO 更新前通知所有进程，共同退出并保留上次 checkpoint，
避免其余进程等待 NCCL 超时。相同配置和训练签名可使用 `--resume .../latest.pt`；
恢复重新开始物理回合，不会恢复失败瞬间的物理状态。数值处理语义发生变化时应启动新实验。

## 本地构建与诊断

使用本仓库重新构建的 Gobot Python 包。开发构建可通过
`cmake --build build --target gobot_python` 更新，再设置
`PYTHONPATH="$PWD/build/python:$PWD"`。不要混用新 Python 包与旧原生库。
当前 checkout 的验证使用这个独立构建目录；主 `.venv` 的原生安装没有被覆盖。
项目最低 Torch 版本现为 2.9，配对实验实际锁定使用 2.11。

## 准备场景与数据

先使用 [reproduction 工具](../examples/g1/reproduction/README.md) 导出固定参考的有效 G1 模型，再执行：

```bash
python -m examples.g1.prepare_tracking --reference-export /data/reference_model --project-path /data/g1_project
python -m gobot.rl.motion_tracking.convert --project /data/g1_project \
  --input /data/lafan1/*.csv --input-fps 30 --output /data/lafan1_npz --compact
```

转换遵循参考的端点、插值和速度定义，以 Gobot 场景 FK 输出 50 Hz NPZ；可同时输出 fp16 `library.pt`。
`manifest.json` 记录 CSV/NPZ/packed library 哈希、帧数和场景 FK 身份。
BONES CSV 支持厘米、欧拉角度和关节度数的输入格式，必须显式填写真实输入 FPS。
不能用公开评估片段清单替代原始训练数据清单。

全量 BONES 使用 `examples.g1.reproduction.export_bones_reference` 和 `prepare_bones`，
分别导出固定参考输入、分批转换并按历史规则独立筛选；详细命令见工具 README。
训练库和评估子集清单分开保存，已完成的数据包含 129943 条训练动作与 129491 条评估子集动作。
`library.pt` 是 71D fp16 动作数据，`ckpt_latest.pt` 才是模型及优化器 checkpoint；LAFAN1 单独存储。

## 训练和恢复

```bash
# 完整网络、缩小批次和 rollout 的功能检查；不是成绩实验。
python -m gobot.rl.train g1-tracking --project /data/g1_project --data /data/lafan1_npz \
  --config examples/g1/train/reproduced_smoke.json --output logs/smoke

# 单节点同步四卡，目标 e8 配方；必须有实际 BONES 训练库。
torchrun --standalone --nproc-per-node 4 -m gobot.rl.train g1-tracking \
  --project /data/g1_project --library /data/bones/library.pt \
  --config examples/g1/train/reproduced_bones.json --output logs/g1_e8_s0

# 按相同 rank 数恢复；先重新填充 replay，再继续绝对控制步调度。
torchrun --standalone --nproc-per-node 4 -m gobot.rl.train g1-tracking \
  --project /data/g1_project --library /data/bones/library.pt \
  --config examples/g1/train/reproduced_bones.json --output logs/g1_e8_s0_resume \
  --resume logs/g1_e8_s0/ckpt_latest.pt
```

`--stop-iteration` 是绝对停止点，不改变 LR 调度。恢复要求相同场景、数据、rank 数和算法参数。
每个 rank 保存自己的 sampler 与 RNG；世界模型、actor、critic 及优化器同步。
与参考恢复方式相同，episode/replay 重新开始，warmup 40 步，UTD 累加器从零开始；不承诺逐位连续恢复。
`config.json` 分别保存 base/effective config、软件版本、数据哈希及模型身份；checkpoint 也校验训练源码哈希。
发布 e8 有 92k 恢复历史，更新数 39968；默认不中断 100k 重建配方为约 39984 次更新，两者必须分别记录。
`metrics.jsonl`、TensorBoard 和 checkpoint 都使用控制步计数。

八卡 4090 实测使用独立的 `examples/g1/train/reproduced_bones_nj500.json`，只将约束容量
从 250 改为 500。原配方仍保留。不能在 `--resume` 时修改容量；也不能把八卡增加的全局
样本量与学习率当成和四卡等价。每 rank 保存一份 checkpoint，八卡续训须保留全部八份。
每 500 控制步打印一次；第一次日志前可能包含初始化、编译和 warmup。

## 完整片段评估

```bash
python -m examples.g1.evaluate_tracking --project /data/g1_project --data /data/lafan1_npz \
  --checkpoint logs/g1_e8_s0/ckpt_latest.pt --num-envs 40 --output logs/g1_e8_s0/evaluation.json
```

所有片段从首帧开始，关闭观测/动作/reset 噪声和随机化，足部摩擦固定 1。
报告逐片段失败、实际执行步数、归一化 return 和各奖励项。
`--num-envs` 不超过选定片段数。单世界诊断若触发容量溢出，可对**双方**显式使用
`--nconmax 128 --njmax 500`；报告会记录容量，不能将它视为正式 35/250 配方。
公开 teacher 评估使用 `--release` 并强制验证固定 SHA256；
teacher 评估结果与 Gobot 从头训练结果分别标记。

上述入口适用于 LAFAN1 或有限片段。全量 BONES 使用
`python -m examples.g1.reproduction.evaluate_bones prepare|worker|merge` 分批加载 dense NPZ，
不要把全库一次塞进显存。最终报告保留全量和过滤子集、逐片段失败、实际步数和 coverage。
视频使用 `python -m examples.g1.reproduction.record_tracking --outcome success|failure ...`；
成功视频必须完成整条片段，失败视频在终止处结束。工具 README 提供可执行命令。

## 代码与验证

| 职责 | 位置 |
|---|---|
| 配置、replay、sampler、WM、BPTT、critic、分布式梯度 | `python/gobot/rl/supermimic/` |
| 场景训练环境、数据转换、训练时钟、恢复、评估 | `python/gobot/rl/motion_tracking/` |
| 任务注册和 CLI | `python/gobot/rl/tasks/`、`train.py` |
| 独立参考工具与来源锁定 | `examples/g1/reproduction/` |
| 数学和完整更新 fixture 回归 | `test_supermimic_reproduced_math.py`、`test_reproduced_training.py` |

回归覆盖独立参考的 replay wrap/terminal rows、WM 更新、actor BPTT、critic GAE、梯度和 Adam 状态。
原生场景/物理回归继续执行。最终成绩对齐需要相同训练库、训练量和独立 seeds 的配对训练；
冒烟测试、teacher 移植和数学测试都不能代替这一步。

恢复 checkpoint 包含每个 rank 的 startup COM；恢复时不依赖环境重新构造所消耗的随机数。
多 rank 恢复会比较控制步、learner 更新和三套优化器计数，拒绝混合不同进度的 shards。
配置的正整数维度、采样模式、学习率和 replay/window 要求会在创建仿真前检查。

## 训练终端日志

速度 PPO/AMP 训练每次更新打印对齐的统计表，CPU、CUDA 和 torchrun 共用同一个入口。
八卡由 rank 0 打印全局统计，`metrics.jsonl` 仍然每次更新保存一行 JSON，详细项保存在 `log` 字段。
恢复训练后，总步数包含已完成的更新；`--iterations` 仍表示本次追加的更新次数。

- `Collection time` 是采样耗时；`Learning time` 包含 GAE、PPO 更新和相关进程同步。
  CUDA 在阶段边界等待任务完成后计时。多卡显示各阶段最慢进程的耗时。
- `Mean reward` 和 episode 长度按本轮实际结束的 episode 汇总；跨更新但未结束的累计值保留。
  没有结束的 episode 显示 `—`，JSON 为 `null`。
- `Episode_Reward/*` 使用与 mjlab 相同的归一化：每个 episode 的加权奖励积分除以配置的最长 episode 秒数。
  多卡按实际 episode 数加权；这不等于按每一步即时奖励取平均。
- `Metrics/*` 中的角动量按环境步平均；触地最高点、落地冲击按落地次数加权；滑动速度按接触脚样本数加权。
  没有落地或接触样本时显示 `—`。
- `Metrics/twist/error_vel_xy` / `error_vel_yaw` 是已结束 episode 的时间平均误差，单位分别为 m/s 和 rad/s。
  仓库内 mjlab 对照入口已使用相同口径；上游原入口采用指令时间上限归一化，不能混用其日志值。
- `Episode_Termination/*` 是所有卡在本次更新中的 episode 结束次数，不是百分比，也不是 mjlab 的每步平均结束数。
  返回数据含 NaN/Inf 时，由 RSL-RL 检查报错并结束采样。地形等级按实际环境数量汇总；没有分配该地形的环境时显示 `—`。

日志仅收集数据，不更改奖励、PPO、物理求解器、评测规则或 checkpoint 配置签名。

共同地形、初始分配和统一统计的对照命令见 [mjlab 对照训练](mjlab_comparison.md)。

## 直行偏航与检查点

独立评测新增 `yaw_rate_bias_rad_s`（有符号的身体偏航角速度误差均值），保留原来的绝对误差 `yaw_error`。
零转向的直行场景另外报告 `heading_drift_deg`、`heading_drift_abs_deg`、`heading_drift_max_deg`，
分别为首次结束时的平均航向变化、平均绝对变化、全程最大绝对变化，单位为度。
逐 episode 字段 `heading_change_deg` / `heading_change_max_deg` 正确跨越 ±180°，结束后不再累加。
主动转向或变速序列没有直行航向基准，顶层 heading_drift 字段为 null；通过判据不变。

每个 `--save-interval` 保留 `model_000499.pt` 等编号检查点及 manifest，同时更新 `latest.pt`。
关闭评测也会保存历史，便于排查策略何时开始退步；不再只保留最后的 latest。

## 自定义机器人接入同一个训练器

公共代码不选择 G1/Go1 资源、不依赖关节数量或足名称。机器人、控制增益、碰撞组、
姿态奖励映射和传感器由 `VelocityTaskCfg`/`LocomotionRobotCfg` 显式提供。
例如在用户自己的 Python 模块中定义配置工厂：

```python
from gobot.rl.locomotion.experiment import RunConfig
from gobot.rl.training.locomotion import training_spec

def training():
    return training_spec(
        task_factory="my_project.velocity:make_task",  # make_task(run) -> VelocityTaskCfg
        experiment=RunConfig(num_envs=1024, iterations=18000),
        name="my_velocity",
    )
```

运行 `python -m gobot.rl.train my_project.velocity:training --backend mujoco-warp`。
也可以直接传 `training_spec(env=task_cfg)` 再通过 `launch(spec, cfg)` 启动。
默认关闭训练内评测；需要时显式提供与任务匹配的 `evaluation_command`。
AMP 数据必须携带 `gobot_motion_v2` 的具名 joint/link 特征布局及控制周期，训练时检查与任务一致。

118 八卡从头训练配置为 `examples/locomotion/configs/g1_velocity_8gpu.json`。
每卡 1024 环境、总计 8192 环境、18000 次更新，每 500 次保存；单次上限 8 小时，
总预算 64 GPU 小时。`g1-velocity` 只是任务预设名称，可用其他预设或用户工厂替换。
