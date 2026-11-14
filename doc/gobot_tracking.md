# Gobot 全身动作跟踪

`g1-tracking` 移植 mjlab `dcdf6ce519abfd2ea45e9503ae6fb650b75bb869` 的
tracking MDP，即 BeyondMimic 的 PPO 动作跟踪阶段。每次训练使用一个动作文件。
它与 `g1-supermimic` 的 GenCoord/世界模型训练是独立任务，不包含扩散模型阶段。

训练从 Gobot `.jscn` 场景编译物理模型，使用 `ArticulationBatch` 的命名关节控制、
状态和接触接口；支持 MuJoCo CPU 和 MuJoCo Warp。训练、评估不导入 mjlab，
不依赖编辑器、NodeScript 或外部 MJCF。MJCF 只作为 Gobot 场景编译的运行时产物。

## 准备场景与动作

从仓库根目录执行。准备器复用 `examples/g1` 的机器人场景、驱动和地面，
生成一个独立项目；原示例不变。输出目录必须不存在。
模型网格使用原项目的绝对资源路径，因此生成项目仍需要原网格资源。
示例训练配方和数据集验证使用 `nconmax=128, njmax=500`，以容纳 Gobot 场景中的
复杂接触；MDP 配置类保留上游的 35/250 默认值。容量溢出会明确终止，不能忽略。

```bash
.venv/bin/python -m gobot.rl.tracking.prepare \
  --csv examples/g1/assets/motions/raw/walk1_subject2.csv \
  --input-fps 30 --output logs/g1-tracking-assets
```

CSV 格式为 root xyz、root 四元数 **xyzw**、29 个 G1 关节角度。
关节顺序是 `gobot.rl.tracking.config.G1_JOINTS`，与现有 G1 重定向 CSV 一致。
转换器插值到 50 Hz，通过 Gobot `SceneKinematics` 计算身体位姿和速度。
输入 CSV 必须已经是机器人重定向动作；这里不做人体骨架重定向。

BONES-SEED 原始 CSV 有名称表头，位置为厘米，root 欧拉角及关节角为度。
显式使用 `--csv-format seed`，转换器按关节名称映射列，检查连续 Frame 索引，
将厘米转米、xyz 欧拉角转 xyzw 四元数、关节角转弧度。SEED 默认 120 Hz；
LAFAN 默认 30 Hz；逐动作帧率不同时通过 `--input-fps` 指定。

```bash
.venv/bin/python -m gobot.rl.tracking.prepare \
  --csv /path/to/seed/g1/csv/clip.csv --csv-format seed \
  --output logs/g1-tracking-seed-assets
```

输出 `tracking-source.json` 保存原 CSV 格式、帧率和 SHA256。
SEED/LAFAN 原始数据继续放在未跟踪的本地数据目录。

也可以输入 mjlab/BeyondMimic 的 NPZ：

```bash
.venv/bin/python -m gobot.rl.tracking.prepare \
  --npz /path/to/motion.npz --layout /path/to/motion-layout.json \
  --output logs/g1-tracking-assets
```

NPZ 字段为 `joint_pos[T,J]`、`joint_vel[T,J]`、`body_pos_w[T,B,3]`、
`body_quat_w[T,B,4]`、`body_lin_vel_w[T,B,3]`、`body_ang_vel_w[T,B,3]`。
四元数为 **wxyz**，位置/速度指向连杆原点，使用世界坐标与 SI 单位。
还需要 `fps`、`joint_names` 和 `body_names`；若 NPZ 没有这些字段，
用 JSON sidecar 明确给出**源数组每一列的名称和顺序**：

```json
{"fps": 50, "joint_names": ["完整的源关节名称列表"], "body_names": ["完整的源刚体名称列表"]}
```

不能根据维数猜测 Isaac Lab、mjlab 和 Gobot 的排列是否一致。
加载器按名称重排，拒绝缺失/重复名称、非有限值、非单位四元数、形状错误及帧率不匹配。
NPZ 保留其已有身体轨迹，因此必须与所用机器人几何匹配。

没有动作文件时，可以用 `--standing-smoke` 替代 `--csv` 生成静止合成参考，
仅用于检查数据与训练链路，不代表已准备好真实运动任务。

## 训练与恢复

```bash
# 短 GPU 验证：32 个环境，两次 PPO 更新。
.venv/bin/python -m gobot.rl.train g1-tracking \
  --config examples/g1/train/tracking_smoke.json

# 正式单 GPU 配置；显存不足时减小 --num-envs。
.venv/bin/python -m gobot.rl.train g1-tracking \
  --config examples/g1/train/tracking.json --device cuda:0

# CPU 验证。
.venv/bin/python -m gobot.rl.train g1-tracking \
  --backend mujoco-cpu --num-envs 4 --set updates=2 \
  --output logs/g1-tracking/cpu-smoke
```

`--set` 可修改 dataclass 字段，例如 `sampling_mode="uniform"`、
`state_estimation=false`、`agent.algorithm.learning_rate=0.0003`。
`--dry-run` 校验并打印配置。支持 `torchrun` 同步多进程，所有进程必须使用同一场景和同一动作文件。

输出包含 `config.json`、`contract.json`、TensorBoard 日志、源代码归档、
`model.pt` 和 `status.json`。checkpoint 记录场景/动作/名称布局摘要、观测约定、
PPO 模型与优化器、自适应采样统计、随机数状态和启动随机化参数。

```bash
.venv/bin/python -m gobot.rl.train g1-tracking \
  --config examples/g1/train/tracking.json \
  --resume logs/g1-tracking/walk/model.pt \
  --set updates=1000 --output logs/g1-tracking/walk-resume
```

恢复必须保持任务、PPO、后端、进程数、每进程环境数、模型和动作配置一致。
`updates` 是本次追加更新数。恢复启动新的物理回合，不承诺逐步精确接续物理状态。
输出目录拒绝覆盖；恢复使用新的输出目录。

## 评估

```bash
.venv/bin/python -m gobot.rl.tracking.evaluate \
  logs/g1-tracking/walk/model.pt --steps 500 \
  --output logs/g1-tracking/walk/evaluation.json

# 可选视频，需要可用的渲染设备。
.venv/bin/python -m gobot.rl.tracking.evaluate \
  logs/g1-tracking/walk/model.pt --steps 500 \
  --video logs/g1-tracking/walk/evaluation.mp4
```

评估关闭噪声、随机化和外推扰动，从动作开始运行，默认运行完整动作，在第一次失败或动作末尾停止；训练时的 10 秒回合上限不会截断评估。
`--steps` 可显式限制评估步数。
报告持续时间、是否失败、是否完成动作及锚点位置误差。短 PPO 验证只证明可以更新参数，
不证明策略已学会跟踪；正式效果需长训练和独立评测。

## 对齐的任务语义

- 50 Hz 控制，200 Hz 物理。G1 使用 29 个关节、14 个跟踪刚体，root 为 `pelvis`，
  anchor 为 `torso_link`。骨盆 IMU 包含安装点平移带来的角速度叉乘项。
- actor 160 维，critic 286 维；关闭状态估计后 actor 为 154 维。
  actor 使用带编码器偏差的关节位置，critic 使用真实状态；仅 actor 加观测噪声。
- 动作是默认关节位置加缩放偏移，再减编码器偏差；**不裁剪策略动作**。
  缩放为每个关节 `0.25 * effort_limit / kp`，驱动参数来自已保存的 Gobot 场景。
- 六项指数跟踪奖励、动作变化惩罚、软限位惩罚、自碰撞惩罚；权重与固定上游版本一致，
  每项乘控制步长。自碰撞取 subtree 对 subtree 的第一个接触槽，阈值 10 N，
  累加一个控制周期内的物理子步。不同后端接触排序和数值轨迹不承诺逐位一致。
- 终止检查锚点高度、末端高度和投影重力 z 分量差；最后一项不是 0.8 rad 朝向阈值。
- 相对身体目标保持参考高度，使用真实 anchor 的水平位置和相对 yaw。各环境使用与 mjlab 相同的 2 米网格原点。
- 连杆位姿和速度直接由物理后端的批量状态接口读取，训练热路径不再重复计算 Python FK。
  奖励和终止使用最后一个物理子步积分前的身体状态，关节限位使用积分后的关节角度；
  策略下一步观测使用刷新后的运动学。这与上游 manager 的采样时序相对应。
- 自适应采样统计失败发生时的片段，EMA `alpha=0.001`，基础总权重 `0.1`，
  默认卷积窗口 1。支持 `start`、`uniform`、`adaptive`。多个重置批次的失败会累加，
  不互相覆盖；手动重置不会额外衰减历史统计。
- 普通重置采样帧 k 并初始化状态，然后命令推进到 k+1。调试用
  `env.reset(frames=[...])` 保留指定帧、关闭该次重置扰动。动作末尾重新采样机器人状态，
  保留 episode 时钟及动作历史；末尾本身不记作失败。
- 启动随机化：躯干质心、编码器偏差、共享脚底滑动摩擦。1–3 秒施加随机速度扰动。
  速度扰动只写入浮动基座速度，不重置姿态、关节、控制器和求解缓存。
  各环境的扰动计时器在回合重置时重新采样。

准备器保留 Gobot 的地面和机器人碰撞模型；这是任务语义移植，不是整个 mjlab
仿真场景的逐字节复制。需要严格训练对照时，另外审计机器人几何、碰撞、地面与多 seed 结果。

## 回归验证

本次新增了基座速度批量写入绑定。更新源码后需要重建并安装 Gobot 原生模块；
只更新 Python 文件而保留旧 `_core` 二进制无法使用该接口。

```bash
.venv/bin/python -m pytest -q tests/python/test_tracking.py tests/python/test_articulation_batch.py
GOBOT_RUN_MUJOCO_WARP_GPU_TEST=1 .venv/bin/python -m pytest -q tests/python/test_tracking.py -k gpu
```

`tests/fixtures/mjlab_tracking.json` 由独立上游函数生成，覆盖观测、奖励、终止和采样，
并保存源文件 SHA256。维护者在完整 mjlab 环境中运行
`tests/support/export_mjlab_tracking.py` 重生成，普通用户无需安装 mjlab。
测试还覆盖真实 CPU 物理、帧末重采样、手动/自动重置、PPO 更新、恢复和评估。

## SEED 与 LAFAN 数据集验证

`examples/g1/tracking/validate_datasets.py` 对 manifest 中的每条动作独立执行：
校验源文件 SHA256 → 转换到 50 Hz → 多相位物理重置检查 → PPO 短训练 → 首回合评估。
manifest 是 JSON 数组，每项包含 `name`、`dataset`（`seed` 或 `lafan`）、
`csv`（本地路径）、`fps` 和 `sha256`。

```bash
.venv/bin/python -m examples.g1.tracking.validate_datasets \
  --manifest /path/to/clips.json --output logs/g1-tracking-validation/run \
  --backend mujoco-warp --device cuda:0 --num-envs 64 --updates 10 --seed 42
```

输出 `inputs.json` 保存输入身份，`report.json` 汇总各动作的转换、状态检查、
训练参数变化和评估结果。检查包含源动作不同相位、软限位导致的重置变化，
以及 Gobot FK 与物理引擎连杆坐标的一致性。这里的 `--seed` 是随机种子，
与 `dataset: "seed"` 指 BONES-SEED 数据集不同。

十次 PPO 更新只用于测试链路；不能据此报告全数据集成功率或已收敛的跟踪表现。

## 一个动作、八卡共同训练一个策略

tracking 的多卡使用 RSL-RL 原生 PPO 梯度平均、KL/学习率同步和全局观测归一化。
各进程使用 `seed + rank` 采样独立环境，模型与优化器始终同步。
`num_envs` 默认是**每卡**环境数；`batch_scope="global"` 时将总环境数均分到各卡。
最终只有一套模型 checkpoint，rank 0 写 TensorBoard、metrics.jsonl 和 status.json。
`synchronization.json` 检查各卡 actor/critic（含归一化）和优化器摘要相等；仅全局采样模式要求采样摘要一致。

默认 `sampler_sync="rank_local"`：与原生 mjlab 相同，各卡独立统计失败，每个控制步更新 EMA。
模型和归一化仍跨卡同步；采样分布可以不同。`rollout_global` 保留为显式实验选项，
只在 rollout 结束后合并全卡失败统计；不能把这个变体当作原生 mjlab 配方。
`random_source="torch_global"` 默认使用原生 Torch RNG，各 rank 以 `seed+rank` 初始化。
`init_at_random_ep_len=true` 与原生 runner 一样随机化初始回合时钟，`allow_tf32=false` 使用 IEEE FP32。

选用 LAFAN `dance2_subject1`，原始 6771 帧、30 Hz，约 225.67 秒。
只使用这一条舞蹈；不同 GPU 从同一动作的不同时间点采样，不混入其他动作。
先将它准备为 Gobot 项目：

```bash
.venv/bin/python -m gobot.rl.tracking.prepare \
  --csv /path/to/lafan1/g1/dance2_subject1.csv --csv-format lafan \
  --output logs/dance2_subject1-assets
```

Python API：

```python
from gobot.rl.tracking import train_motion

if __name__ == "__main__":
    train_motion(
        project_path="logs/dance2_subject1-assets",
        motion_file="motion.npz",
        output="logs/dance2_subject1-8gpu",
        num_envs=1024,          # 每卡；8 卡共 8192 个环境
        updates=30000,
        seed=42,
    )
```

保存为 `train_dance.py`，用 `python train_dance.py` 单卡运行，或用
`torchrun --standalone --nproc-per-node=8 train_dance.py` 同步八卡运行。
也可以直接调用仓库示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m examples.g1.tracking.train_dance \
  --project logs/dance2_subject1-assets \
  --output logs/dance2_subject1-8gpu --num-envs 1024 --updates 30000
```

恢复时全部进程使用同一个 `--resume .../model.pt`；文件包含各 rank 的随机数状态、
启动随机化参数及采样状态。恢复后重新开始物理回合，不保证物理轨迹逐位接续。
旧版单卡 checkpoint 与新的多卡运行配置不互通；卡数或批量范围变化应新建实验。

完整动作评估：

```bash
.venv/bin/python -m gobot.rl.tracking.evaluate \
  logs/dance2_subject1-8gpu/model.pt --backend mujoco-warp --device cuda:0
```
