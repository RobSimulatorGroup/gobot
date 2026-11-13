# Gobot 训练

训练使用 `gobot-train` 或 `python -m gobot.rl.train`，不依赖编辑器或 NodeScript。
机器人、驱动和地形来自 Gobot 场景；通用训练器接收任务配置，不按机器人名称分支。

## 任务与入口

| 任务 | 工作流 |
| --- | --- |
| `go1-velocity` / `g1-velocity` | 标准速度控制 PPO |
| `go1-gait` | PPO 加步态相位奖励 |
| `g1-amp` | PPO 加动作先验 |
| `g1-motion-amp` | RSL-RL PPO 加动作判别器，Gobot 场景仿真 |
| `g1-supermimic` | 官方 GenCoord WM/BPTT，支持同步多卡 |

```bash
python -m gobot.rl.train --list
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/go1_cpu
```

官方 SuperMimic 的数据、单卡/多卡命令和恢复语义见
[操作指南](../examples/g1/supermimic/README.md)。该任务的环境数和学习 batch 默认是**全局数量**，
与其他任务的每卡数量口径不同；多卡均分全局 batch，学习率不缩放。

这些工作流支持 `mujoco-cpu` 和 `mujoco-warp`。CPU 仿真可通过 `--device cuda:0`
配合 CUDA learner；Warp 需要 CUDA。CPU 是语义基准，后端不能静默回退。

动作模仿任务 `g1-motion-amp` 使用默认对齐配方，配置、数据与命令见
[训练说明](../examples/g1/mimickit/README.md)。它复用 RSL-RL 的 PPO、runner 和日志，
使用从 MimicKit 导出的 G1 动作数据，没有额外的 Gymnasium 依赖。
该任务的 CPU 后端使用 CPU learner，更新次数通过 `--set updates=2000` 设置。
保留 COM-v2 观测与固定动作噪声的 SGD 配方，checkpoint 校验场景、控制和数据协议。
`--resume` 恢复模型、优化器、判别器、replay 和 RNG，再开始新的物理回合。

速度任务用 `--terrain flat|rough`。`g1-amp` 和步态任务还可使用 `mixed`。
这些通用 PPO/AMP 任务的 `--iterations` 表示本次运行的更新次数；动作跟踪的 `--stop-iteration`
表示绝对控制步停止点，不改变学习率调度。

## 标准速度训练

G1/Go1 预设使用固定 mjlab 配置及原始 RSL-RL PPO；参考版本与成对训练步骤见
[mjlab 对照训练](mjlab_comparison.md)。物理步长为 0.005 秒，decimation 为 4，
使用 Newton + ImplicitFast。每次更新采样 24 步，优化 5 轮，每轮 4 个 minibatch；
学习率 1e-3、自适应调度、熵系数 0.01、初始动作标准差 1.0。

最后物理子步后计算奖励和终止，再处理推力和重置，随后 forward、更新指令和 sense。
奖励使用上一控制步的射线采样；critic 接触数据遵循奖励缓存规则。
奖励率按控制周期积分一次。失败与超时标记独立，PPO 使用采样动作时保存的价值回填超时。
Warp 事件与 PPO 共用 Torch 设备随机流；reset 清零驱动目标。
地形课程统一比较实际位移与累计指令位移，并计入剩余 episode 的指令距离。

训练中，加权并按时间缩放后的非有限奖励项置零；观测不清洗 NaN/Inf，
由 RSL-RL 检查并报错。环境异常会通知所有训练进程共同退出。
SIGINT/SIGTERM 请求保存 checkpoint 并有序结束。

AMP 的先验、终态价值处理及步态任务的附加观测只在对应工作流中启用。
AMP 需要带具名 joint/link 特征布局的动作库，训练时检查机器人身份和控制周期。

## 配置与自定义机器人

JSON 配置包含 `experiment`，可选 `env`、`agent` 和后端设置。
命令行覆盖配置文件，`--set PATH=JSON` 最后应用；`--dry-run` 只校验并打印配置。

```bash
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval \
  --set env.rewards.foot_slip=-0.1 \
  --set agent.algorithm.learning_rate=0.0002 \
  --output logs/go1_custom --dry-run
```

自定义环境需要独立的匹配评测器，使用 `--no-eval`。内置评测器按任务预设重建环境。
运行目录保存实际任务配置、算法配置、源码归档及摘要；实验身份由真实配置、模型和代码记录。

用户模块可以提供配置工厂，不需要注册机器人名称：

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
也可直接传 `training_spec(env=task_cfg)` 并用 `gobot.rl.training.launch` 启动。
关节、控制增益、碰撞组、姿态奖励和传感器由 `LocomotionRobotCfg`、`VelocityTaskCfg` 提供。
内置机器人配置仅位于 `examples/`。

## 多卡训练与恢复

先导出共享训练地形，然后运行八卡配置：

```bash
.venv/bin/python tests/support/export_training_terrain.py \
  --robot g1 --output logs/g1_velocity_terrain.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --config examples/locomotion/configs/g1_velocity_8gpu.json
```

配置为每卡 1024 环境、总计 8192 环境、18000 次更新，每 500 次保存。
输出为 `logs/g1_velocity_gobot_seed42`。`--hours` 限制本次墙钟时间，
`--budget-hours` 限制预算文件累计 GPU 小时；八卡运行 8 小时需要 64 GPU 小时。

每张卡一个进程，使用 `seed + rank`；PPO 同步模型、梯度、归一化统计和自适应学习率。
`--num-envs` 是每卡环境数。卡数增加不自动改变学习率、rollout 或更新次数。
AMP 判别器目前只支持单进程；标准 PPO、步态 PPO 和 SuperMimic 支持多卡。
Slurm 模板位于 `examples/locomotion/cluster/`。

恢复相同配置与相同卡数、环境数：

```bash
.venv/bin/torchrun --standalone --nproc-per-node=8 -m gobot.rl.train g1-velocity \
  --config examples/locomotion/configs/g1_velocity_8gpu.json \
  --iterations 1000 --resume logs/g1_velocity_gobot_seed42/latest.pt
```

`--resume` 恢复 actor、critic、优化器、归一化、每个 rank 的课程和 RNG，再开始新的物理回合。
`--initialize` 只载入 actor 与归一化，开始新的 critic、优化器和探索状态。
两者都验证实际观测/动作布局、关节顺序、控制参数和模型身份。
每次保存保留 `model_000499.pt` 等编号文件、`latest.pt` 及 manifest；正常结束保存 `model_final.pt`。

## 评测、回放与导出

```bash
python -m examples.locomotion.assessment \
  --checkpoint logs/g1_velocity_gobot_seed42/model_final.pt \
  --terrain flat --steps 1500 --num-envs 128 --final \
  --output logs/g1_velocity_gobot_seed42/flat.json
python -m examples.locomotion.assessment \
  --checkpoint logs/g1_velocity_gobot_seed42/model_final.pt \
  --terrain rough --steps 1000 --final \
  --output logs/g1_velocity_gobot_seed42/rough.json
python -m examples.locomotion.record \
  --checkpoint logs/g1_velocity_gobot_seed42/model_final.pt \
  --output logs/g1_velocity_gobot_seed42/walk.mp4 --continuous
python -m examples.locomotion.verify_export \
  --checkpoint logs/g1_velocity_gobot_seed42/model_final.pt \
  --output logs/g1_velocity_gobot_seed42/policy.onnx
```

评测保留训练时的观测、控制和求解器，仅切换评测地形；显式开启终态采集，
只统计首次结束前的回合。`velocity_error_after_warmup` 排除起步 2 秒，
须结合 `tracking_episodes`、跌倒率、偏离路线率和路线完成率阅读。
平地报告直行航向变化和横向偏移。`--allow-model-change` 用于显式模型迁移评测。

粗糙地形每格路线长 6.5 m。严格验收还要求速度误差、角速度误差和路径偏移满足阈值；
这是 Gobot 的任务标准，不是 mjlab 官方通过标准。`--no-eval` 运行产生的
`admitted=false`、`stage_passed=false` 表示没有训练内验收结果。

## 日志与验证

`metrics.jsonl` 每次更新保存一行；rank 0 打印所有卡的统计。
奖励和 episode 时长按实际结束的回合加权；奖励项为 episode 积分除以最长回合秒数。
速度误差单位是 m/s，角速度误差单位是 rad/s，终止项为次数。
没有观测样本的诊断值为 `null`，终端显示 `—`。

训练入口、重置、终止、课程恢复、PPO 更新、模型参数与日志统计由 `tests/python/` 的行为测试覆盖。
本地开发构建方法见 [构建说明](building.md)。短测试验证流程，策略质量用同预算独立评测验证。

## 动作跟踪与 AMP

SuperMimic 使用 `g1-supermimic`，数据、单卡/多卡训练、恢复和完整评估见
[SuperMimic 指南](../examples/g1/supermimic/README.md)。G1 AMP 动作库与回放见
[G1 指南](../examples/g1/README.md)。
